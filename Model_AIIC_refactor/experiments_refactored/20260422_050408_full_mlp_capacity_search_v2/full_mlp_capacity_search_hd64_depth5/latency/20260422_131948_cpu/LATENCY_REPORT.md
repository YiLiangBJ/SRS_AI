# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`686385.595` samples/s, p50=`0.185` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.090` ms, throughput=`10999.133` samples/s

### full_mlp_capacity_search_hd64_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1302336.901` samples/s, p50=`0.098` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.045` ms, throughput=`21823.035` samples/s

### full_mlp_capacity_search_hd64_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`971902.895` samples/s, p50=`0.131` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.074` ms, throughput=`13426.001` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `19,280`
- MACs / sample: `18,944`
- FLOPs / sample estimate: `38,296`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0922635 | 0.0971958 | 0.10467238999999998 | 10765.864200680875 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.096876 | 0.1028332 | 0.10919256999999999 | 10230.036697187641 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0924685 | 0.0987194 | 0.10788511 | 10704.884531762998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.090457 | 0.09470185 | 0.10158753999999998 | 10999.132608402502 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0943435 | 0.09748725 | 0.10089028 | 10559.72230464672 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1176065 | 0.12778655 | 0.12960429 | 16859.085023737593 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.11048250000000001 | 0.11698259999999999 | 0.12382710999999999 | 17942.92606419943 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1162585 | 0.1237873 | 0.12833876 | 17121.358586652394 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.11221049999999999 | 0.1168329 | 0.12483488999999999 | 17732.266536535473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.100266 | 0.1042863 | 0.10938403999999999 | 19818.458952305096 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1180625 | 0.123432 | 0.12713579 | 33683.18372105204 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.116996 | 0.1214181 | 0.12695236999999998 | 34048.389230290195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.115885 | 0.1224604 | 0.13674946999999998 | 34211.70752000701 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1164985 | 0.12067365 | 0.12544155 | 34187.128375231085 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1038125 | 0.12655809999999992 | 0.14171222 | 37551.619394810594 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.117511 | 0.12330385 | 0.12914272 | 67744.848512357 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1178205 | 0.12469559999999999 | 0.13544464 | 67203.27360586388 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1176735 | 0.12541935 | 0.13409336 | 67481.11878296452 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.117562 | 0.1238388 | 0.13190148 | 67569.57639281168 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1186365 | 0.1236364 | 0.13036103999999998 | 69775.21043767373 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1210675 | 0.1275152 | 0.13952019 | 130705.69146496737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.126031 | 0.13403695 | 0.14308484999999999 | 125777.00711410474 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1390685 | 0.14535320000000002 | 0.14962298999999998 | 114324.07817994924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1185155 | 0.12275275 | 0.13095789 | 134119.9917918565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.112024 | 0.1159346 | 0.12046954 | 142149.51748234726 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1284205 | 0.1337661 | 0.13911843999999998 | 247941.12018248465 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.15408650000000002 | 0.16134265 | 0.17908815999999994 | 205832.2570021561 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.1585105 | 0.16586695 | 0.1757849 | 200330.3698211313 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1410895 | 0.1478007 | 0.15366876 | 225261.9444442098 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1782785 | 0.18279925 | 0.18610592 | 179859.956541338 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.14922950000000001 | 0.15733334999999998 | 0.16764559999999998 | 425740.8789340619 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.19634449999999998 | 0.201217 | 0.20809599999999998 | 332590.9971566588 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.17928699999999997 | 0.18593995 | 0.19169876 | 355139.2928444205 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1745265 | 0.18118215 | 0.19927361999999998 | 363954.3273714581 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.18988349999999998 | 0.21732155 | 0.21968217 | 332966.0648270361 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.18481 | 0.19766505 | 0.20507979999999998 | 686385.5953402999 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.238246 | 0.26809185 | 0.27397755 | 520118.5090022761 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2451895 | 0.2528285 | 0.26289754 | 544115.2953104998 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.221574 | 0.22847725 | 0.23370283999999997 | 574480.1896789967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.23560599999999998 | 0.2489781 | 0.25008374 | 541435.8302943838 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.174581 | 0.18483344999999998 | 0.18760481 | 5695.232395666475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.17532350000000002 | 0.18391405 | 0.18533505 | 5679.908312648054 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.179792 | 0.1901012 | 0.20443054 | 5499.775609155146 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1998915 | 0.2167199 | 0.22073369999999998 | 4984.596599587953 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.285382 | 0.32154464999999993 | 0.33139696 | 3406.07761415435 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.23934149999999998 | 0.2487593 | 0.25208363 | 8314.53693683147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.242505 | 0.249855 | 0.25539814 | 8232.785636291832 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2537825 | 0.2700574 | 0.27361088 | 7842.686377324347 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2778215 | 0.29707185 | 0.29986855 | 7178.394296564737 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.4727345 | 0.5540169500000001 | 0.5809239199999999 | 4160.374100839147 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.2799425 | 0.28806535 | 0.29494972999999997 | 14243.187180903646 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.28943850000000004 | 0.29760654999999997 | 0.30075322 | 13796.993248855264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.295674 | 0.316202 | 0.32169606 | 13415.966744501633 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3194415 | 0.33980645 | 0.35123351999999997 | 12482.656129474102 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.489665 | 0.6140732 | 0.6175392 | 7787.819911960254 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.272902 | 0.28291995 | 0.28804522 | 29254.630770357067 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.30657100000000004 | 0.319677 | 0.32120943999999996 | 26048.760544619676 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.3071385 | 0.32206049999999997 | 0.32676478 | 26041.334368389565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.318243 | 0.3398172 | 0.35068 | 25011.065833439687 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5216025 | 0.6660469499999998 | 0.7180862499999999 | 14571.800891349774 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.28934099999999996 | 0.30436949999999996 | 0.31316973 | 54937.15018340424 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.3219785 | 0.3313871 | 0.33403372000000003 | 50225.26344563263 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.3251615 | 0.3403973 | 0.34229276 | 49263.468047160415 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.35225399999999996 | 0.37681855 | 0.38405476 | 45255.90289712451 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.531628 | 0.6433783 | 0.69533879 | 29067.31171875923 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.306459 | 0.31626355 | 0.32026148 | 104108.96616998664 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.35751 | 0.38784055 | 0.39012501 | 87701.98383531848 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.34617549999999997 | 0.36108165 | 0.36594427999999996 | 94129.41877319482 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.342083 | 0.3773118 | 0.3831999 | 92711.92163478467 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5288695 | 0.7023676 | 0.73210181 | 57585.546027946984 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.357665 | 0.36848960000000003 | 0.37653276999999996 | 178362.30076443293 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.4039045 | 0.46116314999999997 | 0.46226339 | 152184.86580481206 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3831445 | 0.4252372 | 0.43002638 | 167464.670711078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3832425 | 0.4107327 | 0.41963976999999997 | 167681.7613962944 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.563824 | 0.7085259999999998 | 0.77269241 | 109696.24287625741 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.43624450000000004 | 0.4440903 | 0.44719216 | 292724.80968084856 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.45955650000000003 | 0.55402955 | 0.64476188 | 262661.39773374924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.457628 | 0.5403165999999999 | 0.55540212 | 276701.3088058376 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.438463 | 0.52260745 | 0.52679805 | 287435.24005446 | - |
| `full_mlp_capacity_search_hd64_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.582778 | 0.6805342999999999 | 0.7126187499999999 | 217210.33508495367 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 1 | ok | 66.928948 | 0.046806 | 0.0501935 | 0.05209796999999999 | 21088.557601708006 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 2 | ok | 67.087394 | 0.0481485 | 0.05178775 | 0.05215706 | 20550.110004738854 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 4 | ok | 66.851906 | 0.048236 | 0.0525447 | 0.05614588999999999 | 20562.947706778945 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.230256 | 0.046882 | 0.0526794 | 0.05493147 | 20976.63873697141 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 1 | 64 | ok | 64.303903 | 0.045199 | 0.0495001 | 0.04966689 | 21823.035261223915 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 1 | ok | 67.208942 | 0.050813 | 0.054269 | 0.05860052999999999 | 38873.62882992709 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 2 | ok | 67.200988 | 0.050539 | 0.05239415 | 0.053727809999999994 | 39513.21302086812 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 4 | ok | 67.969438 | 0.0502285 | 0.05273165 | 0.05745392999999999 | 39513.55650853473 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 8 | ok | 66.641026 | 0.0527975 | 0.05452275 | 0.05974633999999998 | 37597.476156620556 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 2 | 64 | ok | 66.70495 | 0.050506999999999996 | 0.0520524 | 0.053717709999999995 | 39379.970244494485 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 1 | ok | 67.269008 | 0.051331 | 0.05300385 | 0.05425629 | 77639.93239114687 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 2 | ok | 67.63616 | 0.0509515 | 0.0527708 | 0.05752402999999999 | 77801.90933665703 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 4 | ok | 67.706865 | 0.0556315 | 0.0575337 | 0.06226527999999999 | 71550.84671483231 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 8 | ok | 67.664871 | 0.052706 | 0.05735075 | 0.05885732 | 74125.35784016497 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 4 | 64 | ok | 66.668744 | 0.050286 | 0.05311705 | 0.05424022 | 78692.38438711617 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 1 | ok | 67.755368 | 0.0525485 | 0.05407465 | 0.05762831999999999 | 151495.35382686733 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 2 | ok | 67.881201 | 0.052924 | 0.0568138 | 0.058791159999999995 | 149030.9078925651 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 4 | ok | 67.669571 | 0.054475499999999996 | 0.05780535 | 0.060472769999999995 | 145137.72299957584 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 8 | ok | 67.898371 | 0.0523595 | 0.0549895 | 0.05838847 | 151551.25975090277 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 8 | 64 | ok | 67.742172 | 0.0514335 | 0.05245995 | 0.05371127 | 155439.77019784375 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.176078 | 0.0558825 | 0.058509200000000004 | 0.062285469999999996 | 284674.20816089783 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 2 | ok | 67.64966 | 0.055191000000000004 | 0.059789049999999996 | 0.0608669 | 285015.6918951869 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 4 | ok | 68.453117 | 0.062144000000000005 | 0.06681844999999999 | 0.06935448 | 255186.83184910292 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 8 | ok | 67.36637 | 0.054114999999999996 | 0.0572243 | 0.060015469999999994 | 293209.8821992647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 16 | 64 | ok | 64.605951 | 0.054455 | 0.05601035 | 0.05756053999999999 | 292947.15086924745 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 1 | ok | 67.628424 | 0.060239 | 0.0617746 | 0.06651618 | 528284.3437651881 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 2 | ok | 67.334142 | 0.0720665 | 0.07604944999999999 | 0.07915025 | 447980.25303044636 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 4 | ok | 68.100001 | 0.077375 | 0.0793569 | 0.08341922999999998 | 419756.4677912992 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 8 | ok | 68.391353 | 0.06682550000000001 | 0.0697917 | 0.07420922999999999 | 477796.6407909924 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 32 | 64 | ok | 72.864011 | 0.09870899999999999 | 0.10174415 | 0.10382776999999999 | 323891.294793913 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 1 | ok | 68.169718 | 0.0717545 | 0.07469525 | 0.07921882 | 885254.9174527454 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 2 | ok | 68.99584 | 0.108263 | 0.1104821 | 0.1129587 | 589334.9888735396 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 4 | ok | 68.778366 | 0.094772 | 0.0980051 | 0.10171483 | 671045.7997146377 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 8 | ok | 68.089321 | 0.09781100000000001 | 0.1004582 | 0.10757823999999999 | 651668.913906187 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 64 | 64 | ok | 77.291437 | 0.13232149999999998 | 0.1373149 | 0.14196471 | 481225.73005702824 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 1 | ok | 68.830887 | 0.097788 | 0.10215705 | 0.10395746 | 1302336.901125972 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 2 | ok | 68.947132 | 0.160382 | 0.16460825 | 0.16702536 | 843563.5180921967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 4 | ok | 69.463748 | 0.13263599999999998 | 0.13809675 | 0.14083577 | 962852.9815269134 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 8 | ok | 69.194443 | 0.109931 | 0.11422009999999999 | 0.11714342999999999 | 1161468.6408003971 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `fp32` | 128 | 64 | ok | 76.82409 | 0.15195399999999998 | 0.16211535 | 0.16787373 | 835624.445827577 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 1 | ok | 66.907915 | 0.104439 | 0.10706175 | 0.11072958999999999 | 9542.037541810823 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 2 | ok | 67.186712 | 0.10769 | 0.11242635 | 0.11462175 | 9252.03068195423 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 4 | ok | 66.740269 | 0.110232 | 0.1186469 | 0.12143658 | 8969.580028117838 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 8 | ok | 69.756849 | 0.118282 | 0.1283668 | 0.13126805 | 8472.469641870402 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 1 | 64 | ok | 65.436988 | 0.23531400000000002 | 0.2448969 | 0.24766239999999998 | 4353.046335827599 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 1 | ok | 66.602428 | 0.1476225 | 0.15626725 | 0.16177431999999997 | 13455.277617360967 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 2 | ok | 66.891405 | 0.1566125 | 0.16268265 | 0.16628429 | 12706.67728267198 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 4 | ok | 65.335984 | 0.1688265 | 0.18058969999999996 | 0.18517339 | 11756.696114447204 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 8 | ok | 66.760162 | 0.18925799999999998 | 0.20592555 | 0.20728641 | 10646.02530113081 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 2 | 64 | ok | 66.748788 | 0.37302199999999996 | 0.44949619999999996 | 0.4860836399999999 | 5197.644863949827 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 1 | ok | 66.876245 | 0.16494799999999998 | 0.1726787 | 0.17389635 | 24137.576461807836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 2 | ok | 67.178587 | 0.18352049999999998 | 0.19073345 | 0.19213839 | 21689.656799138054 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 4 | ok | 67.008608 | 0.1840945 | 0.19613374999999997 | 0.20168738 | 21656.095784911657 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 8 | ok | 67.342682 | 0.219255 | 0.23569215 | 0.24410315 | 18201.205775278995 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 4 | 64 | ok | 65.47475 | 0.41412150000000003 | 0.50253375 | 0.5840547999999999 | 9050.554724337526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 1 | ok | 67.420498 | 0.1781395 | 0.1856426 | 0.19774949 | 44500.2428044498 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 2 | ok | 67.374553 | 0.19177 | 0.203866 | 0.20657249 | 41025.245500197234 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 4 | ok | 67.52369 | 0.196913 | 0.2075445 | 0.21069353999999998 | 40429.24129778269 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 8 | ok | 67.517659 | 0.2352795 | 0.24723375 | 0.2515717 | 34427.20320544804 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 8 | 64 | ok | 64.996759 | 0.4391025 | 0.54465935 | 0.5612474900000001 | 18391.32293705565 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 1 | ok | 67.77674 | 0.188015 | 0.1957155 | 0.19658777 | 84445.72989611064 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 2 | ok | 67.065887 | 0.2067785 | 0.22670559999999998 | 0.22979163 | 75178.43837280406 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 4 | ok | 68.263947 | 0.2184685 | 0.22634885 | 0.23131462 | 74060.59915950478 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 8 | ok | 67.139521 | 0.242913 | 0.2642228 | 0.26722382 | 65840.09908934913 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 16 | 64 | ok | 65.228723 | 0.440048 | 0.5269962 | 0.7638043099999992 | 35578.59783678567 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 1 | ok | 67.720527 | 0.20823999999999998 | 0.21512409999999998 | 0.21765794 | 153199.95936371078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 2 | ok | 67.858034 | 0.25839500000000004 | 0.26536709999999997 | 0.26865562 | 125939.27082421586 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 4 | ok | 68.866243 | 0.23929099999999998 | 0.26888185 | 0.27324367 | 131056.6929137728 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 8 | ok | 68.657295 | 0.25487550000000003 | 0.28387484999999996 | 0.2868928 | 124402.75984412647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 32 | 64 | ok | 69.155233 | 0.41538699999999995 | 0.5707883 | 0.59104174 | 72979.68956994345 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 1 | ok | 67.957278 | 0.2502625 | 0.2577221 | 0.25953503 | 255175.21247124195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 2 | ok | 68.573903 | 0.308125 | 0.37356675 | 0.38150504 | 192485.93845105724 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 4 | ok | 68.890904 | 0.28135750000000004 | 0.31658415 | 0.31983852 | 226792.30771520498 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 8 | ok | 68.651975 | 0.2906245 | 0.3305863 | 0.33344983 | 215365.84128156674 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 64 | 64 | ok | 73.972837 | 0.4081375 | 0.51419405 | 0.5190750000000001 | 153112.50204249684 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 1 | ok | 68.342004 | 0.346496 | 0.3525598 | 0.35625074 | 369149.6344005469 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 2 | ok | 69.737869 | 0.43678300000000003 | 0.56081865 | 0.5625184900000001 | 306418.74007221236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 4 | ok | 69.277668 | 0.3711795 | 0.4340867499999999 | 0.44843438 | 357138.9509355813 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 8 | ok | 69.537135 | 0.35301400000000005 | 0.37333215000000003 | 0.37650325 | 375508.0065934512 | - |
| `full_mlp_capacity_search_hd64_depth5` | `jit` | `bf16` | 128 | 64 | ok | 75.70333 | 0.5328135 | 0.6583327 | 0.6752616499999999 | 245792.05914985904 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1383.731296 | 0.0736685 | 0.079079 | 0.08292793999999999 | 13426.001385026304 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1376.275121 | 0.078729 | 0.08703315 | 0.08874051 | 12562.971896631865 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1397.671307 | 0.074583 | 0.08131115 | 0.08359177 | 13298.661064206732 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1395.000343 | 0.079475 | 0.0828858 | 0.08686545999999999 | 12536.35857395411 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1647.39199 | 0.081109 | 0.084026 | 0.087418 | 12316.359993408285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1364.029547 | 0.0826965 | 0.08526575 | 0.08992717999999998 | 24071.050036973134 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1373.68136 | 0.081277 | 0.08287725 | 0.08863901999999999 | 24569.59596036531 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1316.302994 | 0.07758699999999999 | 0.07980575 | 0.08709481 | 25630.91785854706 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1447.574002 | 0.0779565 | 0.0798324 | 0.08416379999999998 | 25527.448133330883 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1638.906768 | 0.0786245 | 0.0810116 | 0.08597249999999998 | 25306.787862560857 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1306.328208 | 0.085698 | 0.08949304999999999 | 0.09418585 | 46342.134245284746 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1314.491398 | 0.081626 | 0.08374045 | 0.08498652999999999 | 48926.98235008038 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1398.543774 | 0.080987 | 0.0826284 | 0.08651824999999999 | 49152.71773978109 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1447.105322 | 0.079542 | 0.08217945 | 0.0854465 | 50058.76899479989 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1594.180492 | 0.08206150000000001 | 0.08392445 | 0.08821184999999998 | 48646.389878437534 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1381.429663 | 0.0853785 | 0.0878153 | 0.09380730999999998 | 93233.26917330526 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1367.203948 | 0.0875225 | 0.08901305 | 0.09403231 | 91140.65030221101 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1404.223772 | 0.0826475 | 0.0841393 | 0.09049336999999998 | 96407.28607705256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1341.833388 | 0.0815195 | 0.08446215 | 0.09020124999999998 | 97474.98365466368 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1578.28852 | 0.0862515 | 0.088338 | 0.09183463999999998 | 92429.36490165861 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1372.699452 | 0.085181 | 0.08742605 | 0.09169775999999999 | 187228.3579815566 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1347.279103 | 0.09057899999999999 | 0.0970185 | 0.1006463 | 175256.7675948484 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1395.62248 | 0.10155549999999999 | 0.10371514999999999 | 0.10983093999999997 | 156845.09233666642 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1408.1696 | 0.08561099999999999 | 0.08759065 | 0.09231524999999999 | 186047.37696454403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1631.995272 | 0.08847150000000001 | 0.08983175 | 0.09775421999999999 | 180216.71059448985 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1384.171695 | 0.0906045 | 0.09340675 | 0.10065664999999997 | 350927.13853353256 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1373.341122 | 0.116686 | 0.12215564999999999 | 0.12602873999999997 | 272690.3975485133 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1396.603268 | 0.1310545 | 0.13498585 | 0.14324392 | 243058.18235622425 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1419.53782 | 0.1051645 | 0.1082715 | 0.11424620999999999 | 302813.2868414 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1625.935144 | 0.1466365 | 0.15264495 | 0.16120948999999998 | 217499.31521699973 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1375.485617 | 0.1117805 | 0.11641625 | 0.12326524999999999 | 569005.2188447417 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1326.793648 | 0.158937 | 0.1627845 | 0.16791037 | 401685.4722415255 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1376.057279 | 0.150439 | 0.15425305 | 0.16120126999999998 | 423936.74015963334 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1424.985993 | 0.14347100000000002 | 0.1483998 | 0.15382348 | 444259.6447727647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1542.006768 | 0.196521 | 0.20985944999999998 | 0.23244813999999994 | 325295.84895049897 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1355.643253 | 0.1312755 | 0.1347409 | 0.13921308 | 971902.894752028 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1364.613577 | 0.2198695 | 0.22641804999999998 | 0.22943513 | 580015.7039251837 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1327.541791 | 0.215422 | 0.21952895 | 0.22183809 | 594209.1716557025 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1328.635974 | 0.18024800000000002 | 0.184958 | 0.18979009 | 708046.6067253585 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1570.310843 | 0.205564 | 0.2184185 | 0.2188688 | 621735.0410063403 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1381.572801 | 0.14284550000000001 | 0.14672215 | 0.1543033 | 6975.351062443621 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1367.94099 | 0.15073150000000002 | 0.15701849999999998 | 0.15848236 | 6601.501049308592 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1413.563562 | 0.15288049999999997 | 0.16239355 | 0.16487965999999998 | 6517.1230238616645 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1376.390669 | 0.164312 | 0.1747329 | 0.17954816999999998 | 6048.1029822591 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1575.856016 | 0.27302550000000003 | 0.2857151 | 0.28820878 | 3673.608145182757 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1372.519814 | 0.196908 | 0.20334249999999998 | 0.21069196 | 10106.766874586443 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1370.639924 | 0.22160600000000003 | 0.23162675 | 0.23487963 | 9006.92443343518 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1384.099211 | 0.2245605 | 0.2427761 | 0.25061594 | 8778.36960570285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1334.177382 | 0.2822855 | 0.40515645 | 0.40956006 | 6253.600119418747 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1567.728737 | 0.454311 | 0.55475685 | 0.7414094699999993 | 4334.275258032409 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1312.245617 | 0.2245325 | 0.23477465 | 0.23696527 | 17727.30930999285 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1329.36696 | 0.262042 | 0.26752755 | 0.27226817000000003 | 15299.785137467421 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1431.473203 | 0.246048 | 0.2588977 | 0.26212735 | 16141.995321726916 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1355.727561 | 0.29115 | 0.3214225 | 0.32571107 | 13701.068176378236 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1546.318132 | 0.495225 | 0.6036069 | 0.6591257399999998 | 7775.339344966536 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1314.198462 | 0.22992649999999998 | 0.2352246 | 0.23834751 | 34748.5309624179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1394.14504 | 0.26474600000000004 | 0.2825918 | 0.2883869 | 29690.99762271605 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1383.499143 | 0.2764675 | 0.29023875 | 0.30131177 | 28791.28084850784 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1419.654766 | 0.334701 | 0.4801416 | 0.49340487999999993 | 21290.28947767894 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1644.018384 | 0.49051049999999996 | 0.5801726 | 0.58458391 | 15751.401923994366 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1362.407946 | 0.2543515 | 0.26687415 | 0.27539525 | 62449.298975394355 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1375.75912 | 0.28880300000000003 | 0.29594415 | 0.30229813 | 56077.772018902135 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1326.473203 | 0.277208 | 0.28588355 | 0.29132705 | 57985.85551026538 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1385.468336 | 0.30023 | 0.32175985 | 0.32418133 | 53110.33366434349 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1507.415482 | 0.5168980000000001 | 0.6240162 | 0.6367590399999999 | 30844.53414034973 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1393.307601 | 0.26409550000000004 | 0.27471275 | 0.27615335 | 120617.44067883496 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1381.46381 | 0.3324825 | 0.340976 | 0.34450415 | 99076.21338638535 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1320.281699 | 0.31465350000000003 | 0.429023 | 0.43408038 | 90006.11985361179 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1409.061793 | 0.31514050000000005 | 0.33571305 | 0.33941166 | 101502.50993503394 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1551.258615 | 0.5017755 | 0.60805095 | 0.7068261499999999 | 60707.40057360905 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1364.13759 | 0.3313825 | 0.3413857 | 0.3438835 | 192623.59964147932 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1370.857394 | 0.37269399999999997 | 0.4277394 | 0.43486219 | 171674.75916445468 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1314.684585 | 0.330213 | 0.4226102 | 0.4986361499999999 | 178306.088813037 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1396.872101 | 0.35276549999999995 | 0.3790324 | 0.38155458 | 180766.40890129952 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1625.273319 | 0.5142279999999999 | 0.6248745 | 0.62951914 | 122005.45707533475 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1369.31681 | 0.41784299999999996 | 0.42936625 | 0.43064051 | 305554.8486947293 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1326.398812 | 0.44162500000000005 | 0.531875 | 0.53905934 | 275954.025886816 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1385.148703 | 0.4069715 | 0.44888659999999997 | 0.45126248 | 309020.4717371325 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1421.974559 | 0.4055145 | 0.45709425 | 0.4678447 | 309297.36771991925 | - |
| `full_mlp_capacity_search_hd64_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1553.024005 | 0.6021000000000001 | 0.9123467 | 3.53819662999999 | 175721.649000509 | - |
