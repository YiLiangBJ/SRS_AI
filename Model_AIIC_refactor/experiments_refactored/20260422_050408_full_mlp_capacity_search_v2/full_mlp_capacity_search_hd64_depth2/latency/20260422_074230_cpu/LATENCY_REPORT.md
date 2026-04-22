# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth2

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1456221.764` samples/s, p50=`0.087` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.055` ms, throughput=`17832.305` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 1 | 1 | ok | 0.055837 | 0.061150499999999997 | 0.06626830999999998 | 17608.634710985956 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 1 | 2 | ok | 0.055379 | 0.05981515 | 0.06441077999999999 | 17832.305003744783 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 1 | 4 | ok | 0.055636000000000005 | 0.06091055 | 0.06689812999999999 | 17648.84862441344 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 1 | 8 | ok | 0.058826 | 0.06305145 | 0.06714691999999998 | 16803.898504453035 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 1 | 64 | ok | 0.055693000000000006 | 0.05997915 | 0.060421739999999995 | 17793.76527817171 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 2 | 1 | ok | 0.06777549999999999 | 0.0727137 | 0.07806825999999999 | 29268.10993570382 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 2 | 2 | ok | 0.0762145 | 0.0796534 | 0.08562852999999998 | 26076.965642554685 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 2 | 4 | ok | 0.06950200000000001 | 0.0733277 | 0.07964757999999998 | 28571.83674052486 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 2 | 8 | ok | 0.0698475 | 0.07421495 | 0.07854151999999999 | 28368.512653065856 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 2 | 64 | ok | 0.0710665 | 0.07445775 | 0.08016559999999999 | 28167.181234122512 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 4 | 1 | ok | 0.0674255 | 0.0702366 | 0.07676481999999998 | 58968.99204007061 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 4 | 2 | ok | 0.0679155 | 0.07131095 | 0.07766918999999999 | 58414.16636997138 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 4 | 4 | ok | 0.0719725 | 0.07518485 | 0.07942175 | 55271.32277816882 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 4 | 8 | ok | 0.06958049999999999 | 0.07216739999999999 | 0.07959207999999998 | 57115.343865783514 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 4 | 64 | ok | 0.062240500000000004 | 0.06450535 | 0.06542613 | 64070.15938733551 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 8 | 1 | ok | 0.0694065 | 0.0726588 | 0.07843409999999998 | 114456.81086698743 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 8 | 2 | ok | 0.06984399999999999 | 0.07402004999999999 | 0.07735946 | 113050.33255168449 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 8 | 4 | ok | 0.0719395 | 0.07595579999999999 | 0.08339909999999998 | 110049.50026521929 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 8 | 8 | ok | 0.07145399999999999 | 0.0745782 | 0.07973404999999999 | 111031.84670943243 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 8 | 64 | ok | 0.066445 | 0.07269175 | 0.07949502999999998 | 120823.2169060678 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 16 | 1 | ok | 0.07467599999999999 | 0.07758335 | 0.08419456999999998 | 213289.4241505482 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 16 | 2 | ok | 0.069312 | 0.07330895 | 0.07926829999999999 | 228366.4207731231 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 16 | 4 | ok | 0.0718995 | 0.07505200000000001 | 0.08206405999999998 | 220859.09217526604 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 16 | 8 | ok | 0.0741715 | 0.07732699999999999 | 0.08578689999999997 | 213676.5833969021 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 16 | 64 | ok | 0.0676185 | 0.0721147 | 0.07386645 | 234306.44705904406 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 32 | 1 | ok | 0.07058049999999999 | 0.07561309999999999 | 0.08056996999999998 | 448511.3208462175 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 32 | 2 | ok | 0.07144400000000001 | 0.07467170000000001 | 0.08478987999999997 | 442952.0651117388 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 32 | 4 | ok | 0.06970499999999999 | 0.07397465 | 0.07720023999999999 | 455434.5870688458 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 32 | 8 | ok | 0.070039 | 0.07341285 | 0.07776850999999999 | 454688.7073228469 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 32 | 64 | ok | 0.069822 | 0.0735635 | 0.07758195 | 454764.43912622693 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 64 | 1 | ok | 0.077837 | 0.08302939999999999 | 0.08858654999999999 | 815301.1646067572 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 64 | 2 | ok | 0.09311900000000001 | 0.0978884 | 0.10683008999999999 | 681656.5788593584 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 64 | 4 | ok | 0.0877645 | 0.09098295 | 0.09758556999999998 | 724480.6775343297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 64 | 8 | ok | 0.0785065 | 0.08167825 | 0.08916174999999998 | 809140.8643444999 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 64 | 64 | ok | 0.0899585 | 0.0931172 | 0.10100527999999998 | 716808.6246413717 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 128 | 1 | ok | 0.0869425 | 0.09207485 | 0.10068143999999997 | 1456221.7643719418 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 128 | 2 | ok | 0.1157995 | 0.12292845 | 0.12647474 | 1096853.5385180686 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 128 | 4 | ok | 0.116593 | 0.12196915 | 0.13231146 | 1089063.2524533703 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 128 | 8 | ok | 0.1055085 | 0.10901595 | 0.11563483999999997 | 1207552.8658154048 | - |
| `full_mlp_capacity_search_hd64_depth2` | `fp32` | 128 | 64 | ok | 0.1219915 | 0.12635785 | 0.12968127000000002 | 1044044.6550949542 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 1 | 1 | ok | 0.07362350000000001 | 0.07636530000000001 | 0.08348081999999998 | 13489.13463694129 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 1 | 2 | ok | 0.07343 | 0.07731905 | 0.07936301999999999 | 13518.588464696611 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 1 | 4 | ok | 0.0705305 | 0.0747782 | 0.07798335999999999 | 14050.512717259566 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 1 | 8 | ok | 0.07444999999999999 | 0.07846449999999999 | 0.08000138999999999 | 13341.391533819762 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 1 | 64 | ok | 0.0717535 | 0.07631825 | 0.08007161 | 13802.976860413533 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 2 | 1 | ok | 0.113414 | 0.12079305 | 0.12691009 | 17479.061395377943 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 2 | 2 | ok | 0.1161165 | 0.1214851 | 0.12672511 | 17133.3241612938 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 2 | 4 | ok | 0.1172675 | 0.12828989999999998 | 0.1298241 | 16835.07068456953 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 2 | 8 | ok | 0.139893 | 0.1519959 | 0.15985676 | 14185.262363520047 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 2 | 64 | ok | 0.229357 | 0.24664125 | 0.24910981 | 8696.081684730168 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 4 | 1 | ok | 0.11189299999999999 | 0.1180097 | 0.12458404 | 35503.23168166383 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 4 | 2 | ok | 0.1201275 | 0.13669845 | 0.1391798 | 32536.75229526452 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 4 | 4 | ok | 0.121776 | 0.1328164 | 0.13630053 | 32549.651645490674 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 4 | 8 | ok | 0.1296055 | 0.13886755 | 0.14313397 | 30677.674429805564 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 4 | 64 | ok | 0.225987 | 0.24862725 | 0.25213989000000003 | 17492.91995930797 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 8 | 1 | ok | 0.114627 | 0.12200585 | 0.12667035 | 69139.86891772252 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 8 | 2 | ok | 0.122765 | 0.1402198 | 0.14407565 | 63543.25111101403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 8 | 4 | ok | 0.1346345 | 0.1579176 | 0.15987149 | 57742.44645795562 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 8 | 8 | ok | 0.1352295 | 0.14405695 | 0.14872623 | 59272.90519921179 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 8 | 64 | ok | 0.2334115 | 0.25253185 | 0.25795394 | 33926.35470478813 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 16 | 1 | ok | 0.119764 | 0.12470455 | 0.12649684 | 133136.60181471845 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 16 | 2 | ok | 0.128286 | 0.14378015 | 0.1563839 | 121625.85181805561 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 16 | 4 | ok | 0.13582450000000001 | 0.16292135 | 0.16825739999999997 | 114385.32756526255 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 16 | 8 | ok | 0.14004 | 0.2053153 | 0.21483887999999995 | 106822.2853719966 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 16 | 64 | ok | 0.225358 | 0.24374634999999997 | 0.2515444 | 70173.75196420717 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 32 | 1 | ok | 0.1305635 | 0.13770675000000002 | 0.14732916999999998 | 242888.7487555747 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 32 | 2 | ok | 0.1416485 | 0.15278619999999998 | 0.16482665999999996 | 222948.75648240928 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 32 | 4 | ok | 0.1434415 | 0.16584745 | 0.16958835 | 217766.8041455722 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 32 | 8 | ok | 0.155215 | 0.21418369999999998 | 0.22294507 | 196425.30498097744 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 32 | 64 | ok | 0.24045850000000002 | 0.25637445 | 0.26086611 | 132465.07992818736 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 64 | 1 | ok | 0.1371845 | 0.14546199999999998 | 0.14870114 | 463005.96785754693 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 64 | 2 | ok | 0.1659515 | 0.19834239999999997 | 0.20698841999999998 | 375541.89521911694 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 64 | 4 | ok | 0.1700205 | 0.18043645 | 0.18224623 | 374709.5854341953 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 64 | 8 | ok | 0.170162 | 0.21688365 | 0.23849313999999996 | 361999.42690965725 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 64 | 64 | ok | 0.24447 | 0.2661673 | 0.27434873 | 259465.77780177834 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 128 | 1 | ok | 0.1623305 | 0.1741818 | 0.18830935 | 779435.2820879805 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 128 | 2 | ok | 0.216416 | 0.24683539999999998 | 0.26657737 | 593528.0037177111 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 128 | 4 | ok | 0.2116475 | 0.24737019999999998 | 0.26150882 | 603178.543582053 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 128 | 8 | ok | 0.20560299999999998 | 0.22362374999999998 | 0.23708374 | 627357.6763142188 | - |
| `full_mlp_capacity_search_hd64_depth2` | `bf16` | 128 | 64 | ok | 0.2889215 | 0.9238828 | 1.1808164699999995 | 326352.4670895209 | - |
