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

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1869701.114` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.040` ms, throughput=`24502.765` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3148058.607` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`46076.749` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2115434.641` samples/s, p50=`0.060` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.038` ms, throughput=`25638.436` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.041048 | 0.04880945 | 0.05024553 | 23341.177525060255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0406855 | 0.049773099999999994 | 0.05130437 | 23437.14111877662 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040533 | 0.04830179999999999 | 0.05125427999999999 | 23504.84505371092 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0407705 | 0.0468508 | 0.04745654 | 23755.700180258253 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0403835 | 0.0441222 | 0.0460781 | 24502.76538210102 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043796 | 0.0517925 | 0.05498889 | 43719.172518477906 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.043246999999999994 | 0.05088315 | 0.05378692999999999 | 43647.918430770034 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.043718 | 0.05091729999999999 | 0.053587609999999994 | 43884.3244315444 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.04308 | 0.04543124999999999 | 0.047207650000000004 | 46128.50386349284 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0435845 | 0.0471859 | 0.047509039999999995 | 45495.96746492374 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.045689499999999994 | 0.05234955 | 0.0534472 | 85963.16134683644 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.043819 | 0.0491804 | 0.051839539999999996 | 88747.56332471446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.047703499999999996 | 0.05350775 | 0.05384908 | 84365.54578602537 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0437825 | 0.04902715 | 0.05132004999999999 | 88867.7186234746 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.043230500000000005 | 0.04700864999999999 | 0.056628969999999994 | 90694.76269488604 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044594999999999996 | 0.05153774999999999 | 0.05328439 | 172390.0894661467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0444625 | 0.0519 | 0.05298156 | 172887.44508662526 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046592999999999996 | 0.054815949999999995 | 0.05734029999999999 | 167499.41793952265 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0440495 | 0.05078529999999999 | 0.051760719999999996 | 175987.16701578122 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0430905 | 0.05367195 | 0.06477090999999996 | 179505.33714243656 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0451955 | 0.0519848 | 0.05366621 | 341708.7573547471 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045308 | 0.052060499999999996 | 0.05609079 | 341413.4516899966 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0453545 | 0.05242329999999999 | 0.054898779999999994 | 339214.8956044958 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045423500000000006 | 0.05200045 | 0.05409042 | 340426.11136359384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.044402 | 0.049556949999999995 | 0.05551178 | 354088.3707199015 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0496495 | 0.055680799999999996 | 0.057233139999999995 | 633808.9541359995 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0489585 | 0.05450815 | 0.055484219999999994 | 638947.5256357727 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.056195499999999995 | 0.0630021 | 0.06548399999999999 | 557386.2209249057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.04796 | 0.0560278 | 0.057469839999999994 | 643042.6204629827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0487345 | 0.065592 | 0.07087451999999998 | 612869.2585162971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0525075 | 0.06070895 | 0.06376721999999999 | 1182183.3152557802 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0679525 | 0.07537864999999999 | 0.0767216 | 946867.422286596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062393 | 0.0706996 | 0.07299336000000001 | 1002184.4489159966 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.061476 | 0.0714779 | 0.07957219999999998 | 1009574.871176669 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.19220500000000001 | 6.00418695 | 6.0149159999999995 | 50737.03953243028 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.062313 | 0.08024455 | 0.1527007899999997 | 1869701.1136699398 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.09180150000000001 | 0.1037801 | 0.1064802 | 1383013.482652689 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.099628 | 0.11419159999999999 | 0.15345773999999984 | 1243705.199892575 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.080472 | 0.1026458 | 0.14358984999999996 | 1538470.7840792313 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2772835 | 0.2979607 | 0.30099602000000003 | 461752.09958318935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0735635 | 0.0834196 | 0.11734227999999988 | 13200.556852290258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.07112550000000001 | 0.0797814 | 0.08292298999999999 | 13696.611294006529 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.07301250000000001 | 0.08318279999999999 | 0.11478103999999989 | 13218.407213654933 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.07605200000000001 | 0.0874679 | 0.13905444999999983 | 12383.069767948702 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.16567949999999998 | 0.38234309999999994 | 0.41533352 | 4592.892443828466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0841575 | 0.10699284999999997 | 0.1885246899999998 | 22102.060685628025 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.075485 | 0.08980969999999999 | 0.09283102 | 25296.461885061985 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.076819 | 0.09190704999999999 | 0.13894885999999984 | 24442.078994355103 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0760365 | 0.08972949999999999 | 0.1413869499999998 | 24660.4137721506 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.202733 | 7.932269349999997 | 8.75714937 | 2118.0926905743836 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0752465 | 0.09538465 | 0.15845617999999995 | 49197.611554354255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.07613 | 0.09163235 | 0.12009428 | 49121.65567487751 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.08926600000000001 | 0.1069789 | 0.14856190999999985 | 42185.69128848601 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08665049999999999 | 0.10709654999999998 | 0.15232930999999988 | 43486.996083561135 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.19765749999999999 | 0.22432465000000001 | 0.22831707999999998 | 19857.210768803685 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.110122 | 0.1300084 | 0.13199115 | 69944.31558175747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10594400000000001 | 0.12506185 | 0.13298827 | 72490.15085381619 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1183205 | 0.1240671 | 0.12623814 | 67223.97583432516 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.10205349999999999 | 0.12252344999999999 | 0.13023911 | 75329.53848228307 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.17612250000000002 | 0.19567210000000002 | 0.20314401 | 45040.52577607359 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0957355 | 0.10594835 | 0.11090544999999999 | 164291.2547149023 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1011875 | 0.1245926 | 0.19731272999999983 | 148481.5717665261 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1058645 | 0.12521244999999998 | 0.16384289999999987 | 144950.48672561563 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1015945 | 0.12239254999999997 | 0.13415912 | 151914.38712799017 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.19760699999999998 | 0.22236785 | 0.22695263999999998 | 79902.31941451576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1093415 | 0.1269703 | 0.16271474999999985 | 277395.20355953526 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.10737250000000001 | 0.13381274999999995 | 0.18954691999999984 | 282537.75410739257 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1250235 | 0.14988215 | 0.2391411199999997 | 244098.38751656437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1118805 | 0.1310551 | 0.14416101999999997 | 278614.05527807336 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.192618 | 0.21593359999999998 | 0.22723186999999997 | 164942.21301111596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10054099999999999 | 0.11645604999999999 | 0.11800582999999999 | 618272.3037386347 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1260345 | 0.15554495 | 0.23809360999999976 | 486566.1370343581 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.1270455 | 0.14913595 | 0.19343536999999983 | 482278.4522056779 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1318345 | 0.1514916 | 0.19723870999999982 | 469755.0418294845 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.33588450000000003 | 0.3516228 | 0.35957195 | 190462.6880320568 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.113811 | 0.13307005 | 0.23879677999999968 | 1070129.6060719155 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1398255 | 0.16252984999999998 | 0.18639412999999996 | 893237.8406254338 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.14918900000000002 | 0.15732715 | 0.16681817999999998 | 861061.301106679 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.15030349999999998 | 0.17961239999999998 | 0.18213766 | 824534.5405893232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.6450255 | 0.6806326 | 0.68758922 | 197999.38941938287 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.635407 | 0.021329 | 0.023151049999999996 | 0.02643208999999999 | 46076.74911947332 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 49.896552 | 0.021983000000000003 | 0.022778 | 0.024978639999999996 | 45893.959253507215 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 47.467707 | 0.022297499999999998 | 0.02293305 | 0.025575539999999994 | 44554.9009945545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 47.647194 | 0.021753500000000002 | 0.023760550000000002 | 0.02481266 | 45950.43418565262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 128 | ok | 53.539693 | 0.021772 | 0.023572749999999996 | 0.02476531 | 45792.15855076977 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 49.668774 | 0.022995500000000002 | 0.0239355 | 0.025252549999999995 | 87724.68484906969 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 49.595538 | 0.0230195 | 0.0255698 | 0.02605942 | 84875.37748324136 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 48.565594 | 0.0231535 | 0.02371415 | 0.023991639999999998 | 87145.21006352887 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 49.644632 | 0.022891500000000002 | 0.02545975 | 0.02912314999999999 | 85398.91970366576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 128 | ok | 52.381263 | 0.0229065 | 0.026467399999999995 | 0.02880825 | 84771.84082900078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 49.767025 | 0.0232625 | 0.02416165 | 0.025639659999999995 | 173307.67224399638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 48.619453 | 0.023202 | 0.0255303 | 0.027788589999999995 | 168363.2065782872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 48.417568 | 0.0230525 | 0.0239619 | 0.02663710999999999 | 173925.14261861696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 47.78395 | 0.023240999999999998 | 0.02384745 | 0.024933719999999996 | 172768.56441416772 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 128 | ok | 52.583798 | 0.0234565 | 0.02641685 | 0.02690538 | 167744.56528576545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 49.288987 | 0.023695 | 0.026161399999999998 | 0.028228099999999992 | 332061.813306547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 49.697045 | 0.0234345 | 0.028045749999999994 | 0.030308279999999993 | 332890.58885016263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 48.618227 | 0.0236015 | 0.026400649999999998 | 0.03143069 | 330015.5767352219 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 49.492685 | 0.023853 | 0.025628599999999998 | 0.027719859999999995 | 328207.82119237905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 128 | ok | 55.408998 | 0.0237075 | 0.02452485 | 0.02742445999999999 | 338421.21427223785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 48.638935 | 0.024633000000000002 | 0.025858950000000002 | 0.02634281 | 645406.9613594852 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 48.340416 | 0.024477 | 0.026490649999999998 | 0.03153904 | 638008.9018192026 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 48.50502 | 0.025141999999999998 | 0.029542299999999997 | 0.03143509 | 624155.4396706956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 49.481862 | 0.0246455 | 0.0262241 | 0.02768845 | 638757.2339256741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 128 | ok | 54.500932 | 0.024933999999999998 | 0.026376849999999997 | 0.028699899999999997 | 642374.7309052105 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 50.47529 | 0.027183 | 0.02805485 | 0.029859549999999995 | 1178250.960090431 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 48.830918 | 0.0266515 | 0.0280032 | 0.029953499999999994 | 1186955.9476136994 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 49.171076 | 0.033447500000000005 | 0.034733499999999994 | 0.03910719 | 953488.2473634561 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 48.468631 | 0.027434 | 0.035374499999999996 | 0.03552545 | 1115381.3175023547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 128 | ok | 54.883501 | 0.0269815 | 0.02825095 | 0.02892211 | 1174592.1045386973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 50.226567 | 0.031501 | 0.03307995 | 0.035781299999999995 | 2024109.6763828148 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 49.459342 | 0.0433005 | 0.04679605 | 0.0492482 | 1466578.733050131 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 49.542173 | 0.0414375 | 0.0438678 | 0.04614367 | 1537688.9975949584 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 49.484535 | 0.0398875 | 0.0417902 | 0.04257845 | 1601885.4191383256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 128 | ok | 85.094327 | 0.109519 | 0.12349324999999998 | 0.13919846999999996 | 580287.6993877603 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 49.420922 | 0.0402995 | 0.0416213 | 0.04717986999999998 | 3148058.6070135795 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 51.835841 | 0.063982 | 0.0680422 | 0.06968796 | 1992098.8379838464 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 51.666709 | 0.071975 | 0.0765185 | 0.07893992 | 1785772.5824196828 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 51.170419 | 0.053956500000000004 | 0.058397399999999995 | 0.06189852 | 2342493.8262477163 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 128 | ok | 141.341956 | 0.2398325 | 0.26923474999999997 | 0.27497823 | 528390.0244917033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 48.235119 | 0.0410885 | 0.042604449999999995 | 0.04587409999999999 | 24269.653322564078 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 48.336123 | 0.044087 | 0.04724435 | 0.057504070000000004 | 22325.544843759137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 48.086194 | 0.0454595 | 0.050422449999999994 | 0.05283632 | 21760.550385248785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 48.080698 | 0.045711 | 0.050147349999999986 | 0.05529542999999999 | 21638.352161584826 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 128 | ok | 53.547479 | 0.1227265 | 0.15040335 | 0.15287697 | 8044.086743570281 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 48.647865 | 0.046673 | 0.048957249999999994 | 0.05215553999999999 | 42511.132602850375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 48.375255 | 0.0486395 | 0.05251975 | 0.057826459999999996 | 40643.59952723365 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 49.401737 | 0.0486285 | 0.05126375 | 0.05627537 | 40837.94563897704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 49.440869 | 0.0515455 | 0.0558518 | 0.06471995999999999 | 38275.753840971905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 128 | ok | 49.990118 | 5.9977595 | 6.01035515 | 6.01305719 | 336.80504224362085 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 49.627267 | 0.0489845 | 0.05265645 | 0.05637549999999999 | 80673.0715707289 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 49.325781 | 0.052721000000000004 | 0.05560225 | 0.06212023 | 75144.97344000914 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 49.323692 | 0.052824499999999996 | 0.05736019999999999 | 0.06196370999999999 | 75191.27721031339 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 49.581169 | 0.05374 | 0.0596395 | 0.062334469999999996 | 73704.07959450963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 128 | ok | 53.013242 | 0.12835 | 0.15681009999999998 | 0.16591823 | 30709.583309341477 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 50.59844 | 0.06424550000000001 | 0.0694296 | 0.07358164999999998 | 123181.45675622574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 48.62836 | 0.0668115 | 0.0738986 | 0.07657995 | 117958.08654289915 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 48.475231 | 0.0676175 | 0.0726139 | 0.07656120999999999 | 117046.74671493676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 49.465239 | 0.068165 | 0.07472369999999999 | 0.07871726999999999 | 115281.71536886835 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 128 | ok | 56.380906 | 0.149026 | 0.1673249 | 0.18443168999999998 | 53050.278268603375 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 51.237689 | 0.0652715 | 0.07711359999999999 | 0.07859793999999999 | 240680.33109189747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 50.081354 | 0.0734825 | 0.08288585 | 0.09096989 | 211708.59687745696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 49.810511 | 0.07751 | 0.086008 | 0.08978416 | 203810.1284462382 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 48.460265 | 0.0739315 | 0.0822003 | 0.08549148999999999 | 212840.79136334636 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 128 | ok | 52.678468 | 0.164489 | 0.17672715 | 0.18081395 | 96683.08143966425 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 50.087205 | 0.0682585 | 0.0780364 | 0.08040573 | 463565.893718827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 48.926875 | 0.07893549999999999 | 0.0871075 | 0.09066416 | 398967.5716664202 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 49.145871 | 0.078242 | 0.088264 | 0.09516859999999999 | 405331.73362409126 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 48.473832 | 0.081885 | 0.08859334999999999 | 0.09103836 | 389378.24811425334 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 128 | ok | 55.783618 | 0.151225 | 0.1761633 | 0.18443773 | 207527.78505205901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 50.333425 | 0.069914 | 0.0814766 | 0.08690224999999999 | 889584.741842508 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 49.519004 | 0.0897135 | 0.09913564999999999 | 0.1027145 | 703011.1726050606 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 49.591037 | 0.0903675 | 0.0995441 | 0.10063512999999999 | 695017.8945388536 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 49.386275 | 0.0969865 | 0.1066742 | 0.11351422999999998 | 655598.1216294317 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 128 | ok | 73.612244 | 0.376505 | 0.41162899999999997 | 0.42502550999999994 | 169375.81518727302 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 51.347987 | 0.0803475 | 0.0901674 | 0.09377002999999999 | 1566653.0074107582 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 50.459007 | 0.103535 | 0.1140193 | 0.11620918 | 1218999.0912742713 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 52.41979 | 0.1092465 | 0.1177203 | 0.11976663 | 1159135.7556252044 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 50.560129 | 0.1139685 | 0.120873 | 0.12560404 | 1119722.0849785083 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 128 | ok | 97.778786 | 0.38653899999999997 | 0.41450215 | 0.43372053 | 328816.74421175924 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 503.064711 | 0.038504 | 0.04100965 | 0.04148188 | 25890.54410014048 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 508.984618 | 0.042463 | 0.044802049999999996 | 0.04546292 | 23591.371031400588 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 519.960074 | 0.0406425 | 0.04402885 | 0.045320799999999994 | 24303.48638372872 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 540.361692 | 0.0383265 | 0.04101685 | 0.04500570999999999 | 25944.644506480974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 128 | ok | 805.918293 | 0.038267499999999996 | 0.0421357 | 0.045710350000000004 | 25638.43550164419 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 504.729351 | 0.0411085 | 0.0446015 | 0.046731909999999995 | 48123.40380684999 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 512.529558 | 0.0413255 | 0.04638515 | 0.04812338 | 47518.67008547658 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 511.54498 | 0.0423815 | 0.04449025 | 0.04515035 | 47020.720150541536 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 544.802326 | 0.0410575 | 0.043450050000000004 | 0.04716439999999999 | 48350.966124829625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 128 | ok | 744.03044 | 0.04315 | 0.045459 | 0.04672393 | 46445.63190443347 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 496.40565 | 0.0396475 | 0.041558700000000004 | 0.05343158999999999 | 99542.65128865426 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 501.440273 | 0.043840500000000004 | 0.04846905 | 0.05041522 | 90516.37328047177 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 511.090214 | 0.0421905 | 0.04778445 | 0.051646969999999987 | 92390.4030392747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 523.795979 | 0.039642 | 0.04253695 | 0.04335015 | 100824.33980222297 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 128 | ok | 719.517508 | 0.0401415 | 0.04258765 | 0.04549749999999999 | 98773.38274636427 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 500.13107 | 0.039830500000000005 | 0.04202425 | 0.04374802 | 200267.05611933515 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 509.28624 | 0.0421765 | 0.04563565 | 0.04710694 | 187402.609206528 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 512.13007 | 0.04023 | 0.04685925 | 0.04835711 | 190671.40167314155 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 534.838622 | 0.0419095 | 0.048423499999999994 | 0.04940323 | 185948.4346395901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 128 | ok | 671.867837 | 0.0398465 | 0.044508849999999996 | 0.04614838 | 199186.7206197097 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 496.547388 | 0.044274499999999994 | 0.04722355 | 0.04918118999999999 | 357450.3601982598 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 505.116304 | 0.041605 | 0.0450857 | 0.048656239999999996 | 381852.097227181 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 518.766927 | 0.04588250000000001 | 0.053800799999999996 | 0.057713949999999986 | 340438.85973408323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 557.806709 | 0.0421025 | 0.05044485 | 0.0509368 | 362386.45979231637 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 128 | ok | 799.108904 | 0.0414065 | 0.04791855 | 0.04924225 | 379406.2103105535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 499.272458 | 0.044693 | 0.052871999999999995 | 0.053299849999999996 | 690632.4769058974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 508.410204 | 0.0456875 | 0.0549672 | 0.055862129999999996 | 677370.2454959112 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 505.949253 | 0.055356 | 0.0601378 | 0.06405409 | 572062.7438417446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 536.841537 | 0.0439215 | 0.05050154999999999 | 0.05242748 | 720072.6553309229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 128 | ok | 819.572556 | 0.0443005 | 0.0475048 | 0.051474009999999994 | 717747.8865689182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 499.195578 | 0.0500035 | 0.0518735 | 0.05290668999999999 | 1277524.1581814445 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 508.396519 | 0.0786895 | 0.08438715 | 0.08644202999999999 | 806673.6107819994 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 515.1501 | 0.0624 | 0.0687586 | 0.07167459999999999 | 1017792.9287564033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 535.315235 | 0.059056 | 0.06594974999999999 | 0.06851564 | 1064734.8826778615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 128 | ok | 837.859509 | 0.1427765 | 6.00956385 | 7.750604029999999 | 43062.732995458646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 497.47614 | 0.060311500000000004 | 0.06306295 | 0.06700979999999998 | 2115434.6408372098 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 512.12214 | 0.1094065 | 0.11393384999999999 | 0.12160688999999998 | 1163760.5686279607 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 507.720129 | 0.1232455 | 0.1313415 | 0.13396469 | 1035289.2913719799 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 544.051486 | 0.077636 | 0.08285565 | 0.08673417 | 1645553.021518177 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 128 | ok | 724.155866 | 0.1527905 | 0.16793324999999998 | 0.17751339 | 830687.1405087828 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 496.654826 | 0.061505 | 0.066108 | 0.06710467 | 16136.86644645223 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 510.213473 | 0.079003 | 0.0897373 | 0.09115865 | 12512.26514791124 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 508.130273 | 0.07931250000000001 | 0.08601095 | 0.08932228999999998 | 12505.733878983514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 542.608959 | 0.07974200000000001 | 0.0876444 | 0.08985003999999999 | 12329.887622938228 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 128 | ok | 730.823121 | 0.1654745 | 0.18980129999999998 | 0.23014136999999998 | 5979.988566261862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 492.684904 | 0.075851 | 0.0816258 | 0.08481044 | 26199.052013501943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 523.757256 | 0.07126650000000001 | 0.07877955 | 0.07965194 | 27679.331621643465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 509.02104 | 0.073023 | 0.07776304999999999 | 0.1893586999999996 | 25730.58125897676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 545.162438 | 0.07316349999999999 | 0.0800541 | 0.08764131 | 26838.411026507492 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 128 | ok | 704.49584 | 0.15045799999999998 | 0.19817074999999998 | 0.2277251599999999 | 12593.36722460321 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.21505 | 0.0749585 | 0.08177 | 0.08715548 | 52737.72054914733 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 515.909382 | 0.0832965 | 0.09088875 | 0.09502552999999998 | 47551.93146435222 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 515.775375 | 0.07961850000000001 | 0.08615445 | 0.09148181999999999 | 49672.558494404875 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 547.715774 | 0.075955 | 0.08266605 | 0.08689906 | 52020.98942881474 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 128 | ok | 663.973964 | 0.1598165 | 0.19556744999999998 | 0.22037628999999992 | 24582.759889244833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 496.770844 | 0.1027685 | 0.11275769999999999 | 0.13180557999999998 | 76589.47086249318 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 510.791223 | 0.10323099999999999 | 0.10986344999999999 | 0.11138724 | 77032.47785556989 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 501.84847 | 0.10765849999999999 | 0.1124757 | 0.11544283 | 74031.1634182409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 553.477891 | 0.09618399999999999 | 0.10306309999999999 | 0.10847177000000001 | 82539.50959975766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 128 | ok | 800.837356 | 0.17565150000000002 | 0.1938275 | 0.22694092 | 44774.27664077856 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 502.270545 | 0.090958 | 0.10140714999999999 | 0.10583131999999999 | 172595.53918199206 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 509.427383 | 0.09335199999999999 | 0.10837595 | 0.11078671 | 167857.86299734973 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.552392 | 0.0971405 | 0.11120484999999998 | 0.11555749 | 161851.64758907811 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 546.810391 | 0.1020005 | 0.11741909999999998 | 0.12180197999999999 | 152993.77257972935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 128 | ok | 720.194447 | 5.9926425 | 6.028228449999999 | 6.330925379999999 | 3783.444275900409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 535.92367 | 0.103045 | 0.1089329 | 0.11301214999999998 | 307230.63458103535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 556.430933 | 0.11257700000000001 | 0.128902 | 0.13156073 | 276803.8223839833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 516.707376 | 0.11696500000000001 | 0.1222573 | 0.12904199 | 272931.1181561456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 542.789642 | 0.10635249999999999 | 0.1170494 | 0.12306857 | 296592.394370009 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 128 | ok | 677.538396 | 0.2099705 | 0.24892745 | 0.25342043 | 149157.6693517533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 508.503977 | 0.1070315 | 0.1126519 | 0.11775824999999998 | 592929.4643697568 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 514.015118 | 0.137184 | 0.14952215 | 0.15310367 | 462729.790529454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 505.518882 | 0.136327 | 0.14386615 | 0.14649882 | 467663.82336805057 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 545.706406 | 0.1393135 | 0.14744995 | 0.14872071 | 462016.73771136545 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 128 | ok | 705.488922 | 0.4414675 | 0.47804245 | 0.5125282999999999 | 144755.79606731073 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 497.525039 | 0.1118945 | 0.1202915 | 0.12394703 | 1133644.2588889224 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 503.152291 | 0.1390965 | 0.14878525 | 0.15232803 | 923720.0631709059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 509.509376 | 0.154682 | 0.16646429999999998 | 0.16917315 | 845064.6573493886 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 539.012883 | 0.155693 | 0.1655098 | 0.17642444999999998 | 824854.3036644411 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 128 | ok | 850.041331 | 0.39087150000000004 | 0.4264554 | 0.43587381999999997 | 324782.65189704025 | - |
