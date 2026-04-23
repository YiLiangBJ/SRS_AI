# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`997599.371` samples/s, p50=`0.128` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.078` ms, throughput=`12640.985` samples/s

### full_mlp_capacity_search_hd32_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1817519.869` samples/s, p50=`0.070` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.041` ms, throughput=`24004.935` samples/s

### full_mlp_capacity_search_hd32_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1426545.288` samples/s, p50=`0.089` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.065` ms, throughput=`15199.384` samples/s

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
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0782045 | 0.0829873 | 0.09058362999999998 | 12640.984904641466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.08138300000000001 | 0.08443555 | 0.09301364999999999 | 12180.20565546437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.08253250000000001 | 0.08536669999999999 | 0.09376224999999998 | 12044.91017026444 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.07875 | 0.10178519999999992 | 0.117684 | 12280.109292972706 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.07934250000000001 | 0.0822619 | 0.09211968999999998 | 12499.15943152823 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0995985 | 0.1040449 | 0.10954421999999998 | 19953.528232745935 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0970785 | 0.10400519999999999 | 0.10911760999999998 | 20419.98185072013 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.09914300000000001 | 0.1042001 | 0.11119629999999998 | 20035.113539990187 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.10130700000000001 | 0.10605735 | 0.11240719999999998 | 19585.231795135343 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.088301 | 0.09257015 | 0.09579156999999999 | 22517.263422934782 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1010505 | 0.10494415 | 0.11256060999999998 | 39283.26118635056 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.100543 | 0.10812885 | 0.11353657999999998 | 39372.589905340414 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.101272 | 0.10515635 | 0.11198668999999999 | 39274.62130919201 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0995425 | 0.10370114999999999 | 0.11022845999999999 | 39922.50243826684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1020615 | 0.10533094999999999 | 0.11027333 | 39011.122656236126 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.107018 | 0.1149997 | 0.12037777999999999 | 73953.83062354173 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1021415 | 0.10651039999999999 | 0.11087965 | 77891.9327325269 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.103465 | 0.109776 | 0.11566428999999999 | 76667.817322825 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.114802 | 0.11925469999999999 | 0.12361387999999998 | 76449.19482752393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1018295 | 0.10636754999999999 | 0.10929797 | 78027.70920009114 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.1020165 | 0.11064419999999998 | 0.11374925999999999 | 155292.04221458876 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1049165 | 0.111636 | 0.12573737999999998 | 150747.82225927268 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1038625 | 0.11328925 | 0.11726087999999998 | 152594.52658692587 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.105096 | 0.10985084999999999 | 0.11349999 | 151563.0029405117 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.09389549999999999 | 0.1003987 | 0.1041358 | 169055.27684914775 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.10328999999999999 | 0.10785975 | 0.11567287999999999 | 307619.42555148475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.104935 | 0.11515524999999999 | 0.12296307999999999 | 301330.75193322514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.11962 | 0.12540235 | 0.14236136 | 265106.11949426366 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1073755 | 0.11249374999999999 | 0.11598624 | 295791.9163027193 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.09587 | 0.10143364999999999 | 0.10611918 | 331481.25090250943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.1097465 | 0.1171865 | 0.12381562 | 577524.0281579059 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.136184 | 0.1424354 | 0.14944335999999997 | 466973.9939264195 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1260885 | 0.13542475 | 0.14077188 | 503052.3487278357 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.12202650000000001 | 0.1281607 | 0.13545456999999997 | 520144.6327169348 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.118558 | 0.12473295 | 0.12688096000000001 | 540676.8057084657 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.12750450000000002 | 0.13504804999999998 | 0.13787453 | 997599.3706394971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.160935 | 0.1686793 | 0.17576573999999998 | 790520.7666717432 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1721625 | 0.1794787 | 0.18588209 | 741756.621076064 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.146783 | 0.15433745000000001 | 0.15951412 | 869599.7178148916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.15211999999999998 | 0.1587077 | 0.16678201999999998 | 838229.1152596467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.143747 | 0.1547992 | 0.1583141 | 6890.093093425804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.155459 | 0.16522155 | 0.17214168 | 6385.028996970686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1608275 | 0.17393625 | 0.1772637 | 6159.729169027896 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.16568149999999998 | 0.1837417 | 0.19061909 | 5959.945116057414 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2680935 | 0.29623554999999996 | 0.29833271 | 3652.939480217397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.14944600000000002 | 0.157804 | 0.16317410999999998 | 13270.398028337342 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15840100000000001 | 0.1670568 | 0.17715706 | 12518.082369983442 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1655395 | 0.1745766 | 0.18507940999999997 | 11981.224941268036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.17859049999999999 | 0.19097969999999997 | 0.19834764 | 11171.798804840964 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.2751935 | 0.3000895 | 0.30715898999999997 | 7203.336354892822 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1523525 | 0.16074335 | 0.16510803 | 26068.41054774814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.15927750000000002 | 0.1786955 | 0.18415653999999998 | 24580.03474387911 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.164463 | 0.1715545 | 0.17363868999999998 | 24304.939492853133 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.18218800000000002 | 0.1999454 | 0.20527277 | 21758.746417694387 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.27710049999999997 | 0.30500385 | 0.31087548 | 14214.545104350753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.2099655 | 0.21776605 | 0.21959448 | 37996.5050814626 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2351185 | 0.24309195 | 0.24860828999999998 | 34018.15855285393 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.245527 | 0.2557797 | 0.26518317999999996 | 32436.115246490554 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.26336950000000003 | 0.28531075 | 0.29516097999999996 | 30180.816288465827 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.466038 | 0.5627508499999999 | 0.5678197700000001 | 17155.680769150928 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.218973 | 0.2277906 | 0.23196691 | 72801.32491131207 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.250173 | 0.26283025 | 0.2685817 | 63711.66140819815 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.24283 | 0.25338745 | 0.25886285 | 65983.92004860376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.271861 | 0.29659445 | 0.30974147999999996 | 58417.10102215323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.451738 | 0.5422544999999999 | 0.58318763 | 34536.69242960019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.235013 | 0.2451489 | 0.24802845999999998 | 135621.16059672972 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.26500049999999997 | 0.27163265 | 0.2827404 | 121101.11185958327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.25327849999999996 | 0.26786814999999997 | 0.27329546 | 126708.68643480561 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.282204 | 0.29670185 | 0.31912893999999997 | 114099.23538536766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.455037 | 0.5552298999999999 | 0.56924562 | 68884.34950592916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2599455 | 0.26943435 | 0.27255165000000003 | 245123.6855816869 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.296425 | 0.30361755 | 0.30896907 | 218477.77114986014 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.28713500000000003 | 0.3040916 | 0.3132018 | 223541.4583481488 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.3105065 | 0.33338009999999996 | 0.33446841 | 205512.19676354257 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4718755 | 0.56117205 | 0.5724844299999999 | 133792.4476922747 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.296936 | 0.30613250000000003 | 0.31028596999999997 | 429648.8882432221 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.351448 | 0.37971259999999996 | 0.40080090999999995 | 370021.8902637643 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.32064950000000003 | 0.3694794 | 0.3760415 | 389198.66670266754 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.330527 | 0.3617177 | 0.37429109 | 386492.7545894203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.463981 | 0.5329811499999999 | 0.7080232 | 264086.1270688363 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.323616 | 0.0457585 | 0.05023275 | 0.051227659999999994 | 21515.12986962692 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.271355 | 0.044775 | 0.0488735 | 0.05369306999999999 | 21889.070568174604 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.530536 | 0.0460515 | 0.05109345 | 0.053100459999999995 | 21305.353438769263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 8 | ok | 63.086606 | 0.046489 | 0.05121555 | 0.05170644 | 21345.655753486706 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 1 | 64 | ok | 60.337652 | 0.041096 | 0.04467445 | 0.04523135 | 24004.93541472127 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.593992 | 0.045713000000000004 | 0.047605749999999995 | 0.04888478 | 43590.70593840546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 2 | ok | 63.008193 | 0.044738 | 0.047619049999999996 | 0.04901009 | 44257.14819329044 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.674639 | 0.0463405 | 0.04802195 | 0.05013778 | 43336.212302370615 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 8 | ok | 62.728729 | 0.051200499999999996 | 0.05424895 | 0.057773649999999996 | 39718.09683589754 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 2 | 64 | ok | 63.411757 | 0.044649999999999995 | 0.047143899999999996 | 0.049488569999999996 | 44457.05048809396 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 1 | ok | 62.505591 | 0.049481 | 0.052340849999999994 | 0.056901379999999994 | 80198.47518639128 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 2 | ok | 62.969319 | 0.046361 | 0.049651549999999996 | 0.053667579999999986 | 85277.7367970876 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 4 | ok | 61.660139 | 0.0460295 | 0.0491212 | 0.05166243 | 86004.60038607466 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 8 | ok | 62.428527 | 0.04836650000000001 | 0.05127629999999999 | 0.056878219999999986 | 81888.57994147424 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 4 | 64 | ok | 60.679356 | 0.045833 | 0.048898199999999996 | 0.05002832 | 86560.22644155238 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 1 | ok | 62.659985 | 0.0487335 | 0.0501486 | 0.055628719999999986 | 163192.43570422023 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 2 | ok | 62.990941 | 0.0486645 | 0.05127795 | 0.053339700000000004 | 162916.6644605035 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 4 | ok | 62.891288 | 0.0522615 | 0.0559066 | 0.06023774999999999 | 152971.27581868315 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.309317 | 0.052106 | 0.05449145 | 0.05701946999999999 | 153650.83998993586 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 8 | 64 | ok | 61.532003 | 0.047924999999999995 | 0.0493239 | 0.05168341 | 166463.85819942702 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.252589 | 0.048911 | 0.052264 | 0.054879979999999995 | 323373.039500421 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 2 | ok | 62.67411 | 0.053818 | 0.055900649999999996 | 0.05797862 | 303059.6906366678 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.099938 | 0.0552165 | 0.05820085 | 0.05926816 | 287902.0326963141 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 8 | ok | 63.761105 | 0.053365499999999996 | 0.0552265 | 0.0576184 | 300826.4077453775 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 16 | 64 | ok | 64.230959 | 0.0483565 | 0.051122049999999995 | 0.054125749999999986 | 328880.4293534005 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.062627 | 0.051473500000000005 | 0.05525814999999999 | 0.059404889999999995 | 614393.5513252853 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 2 | ok | 63.33926 | 0.0531275 | 0.05970879999999998 | 0.06918266999999997 | 592290.7436802577 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 4 | ok | 63.29335 | 0.06357550000000001 | 0.06541855 | 0.06942609 | 501956.2174963741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.869248 | 0.055299 | 0.057569199999999994 | 0.05975629 | 581390.0673140688 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 32 | 64 | ok | 61.167172 | 0.050799 | 0.05383945 | 0.057456419999999994 | 625429.00520826 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 1 | ok | 64.060358 | 0.058964 | 0.06299835 | 0.06824476 | 1073917.759377987 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.296651 | 0.07461100000000001 | 0.0835239 | 0.08521313 | 827906.2354793007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 4 | ok | 63.996932 | 0.0675775 | 0.07176729999999999 | 0.07503665 | 939047.5944535153 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.171373 | 0.067684 | 0.07428894999999999 | 0.07585723 | 935992.1751054161 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 64 | 64 | ok | 68.130798 | 0.091322 | 0.09740879999999999 | 0.10703114999999998 | 695137.0169367963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 1 | ok | 64.139531 | 0.069963 | 0.07511744999999999 | 0.07699073 | 1817519.8691840076 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 2 | ok | 64.414961 | 0.09397749999999999 | 0.10114104999999998 | 0.10376337 | 1350129.2221335967 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 4 | ok | 65.188812 | 0.11097950000000001 | 0.1172343 | 0.12044811999999999 | 1197728.9561829576 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 8 | ok | 65.366183 | 0.0861035 | 0.0911927 | 0.09319445 | 1479471.1815195556 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `fp32` | 128 | 64 | ok | 68.945518 | 0.1101665 | 0.12103475 | 0.12466220999999998 | 1146532.5987881152 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 1 | ok | 62.821756 | 0.0905735 | 0.09599284999999999 | 0.0983956 | 10960.477613772498 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 2 | ok | 62.05441 | 0.09114749999999999 | 0.09644029999999999 | 0.10058206999999998 | 10905.705778017413 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 4 | ok | 62.084257 | 0.093303 | 0.10343975 | 0.10765667999999999 | 10599.640756975467 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 8 | ok | 61.949561 | 0.1050335 | 0.11591049999999997 | 0.1214262 | 9541.757114572649 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 1 | 64 | ok | 62.284578 | 0.1993285 | 0.2237172 | 0.23973380999999996 | 4906.386152215724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 1 | ok | 62.509049 | 0.0939855 | 0.10212099999999999 | 0.10451134999999999 | 21071.351811504115 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.807754 | 0.098633 | 0.10499915 | 0.10599625 | 20145.446091692804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.884739 | 0.1015195 | 0.1115372 | 0.11524096999999998 | 19353.945930881255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 8 | ok | 62.891316 | 0.109213 | 0.11745795 | 0.12339620999999999 | 18303.248588682258 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 2 | 64 | ok | 62.100876 | 0.20171699999999998 | 0.22465875 | 0.23290629 | 9803.052749050477 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 1 | ok | 62.70858 | 0.0997905 | 0.10733089999999999 | 0.10877060999999999 | 39717.76555794525 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 2 | ok | 62.81361 | 0.105712 | 0.1216888 | 0.1234328 | 36879.22808825273 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.52903 | 0.100944 | 0.10719554999999999 | 0.11019052 | 39406.55307333677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.256082 | 0.1231875 | 0.13269635 | 0.13987897 | 32688.86153194388 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 4 | 64 | ok | 63.720498 | 0.1916115 | 0.21385579999999998 | 0.2408436599999999 | 20409.08583928845 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.394044 | 0.1442185 | 0.1520712 | 0.15494549999999999 | 55176.59199296609 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.394876 | 0.1545045 | 0.1621516 | 0.16580864 | 51738.71843711818 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 4 | ok | 62.931859 | 0.15509 | 0.1666413 | 0.17333467 | 51193.25723370324 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 8 | ok | 63.143487 | 0.1883195 | 0.20249535 | 0.21362019999999998 | 42162.08877733737 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 8 | 64 | ok | 60.69594 | 0.384743 | 0.441225 | 0.46762013999999996 | 21517.58468124415 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 1 | ok | 62.447541 | 0.141458 | 0.15041745 | 0.1519809 | 112197.90363826946 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 2 | ok | 63.207545 | 0.1663945 | 0.17606855 | 0.18172407 | 95948.37593581149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 4 | ok | 62.602089 | 0.164115 | 0.17398365 | 0.17787313 | 97232.64941103145 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.827363 | 0.19065949999999998 | 0.20344445 | 0.20716713 | 83913.45501903156 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 16 | 64 | ok | 61.051971 | 0.3651295 | 0.43250849999999996 | 0.4927708199999999 | 41959.24498534574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.385598 | 0.1588325 | 0.16648015 | 0.17051092999999998 | 200345.69649930956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 2 | ok | 63.082784 | 0.1835055 | 0.1905015 | 0.19176171 | 173707.03071607143 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.935132 | 0.184008 | 0.19522425000000002 | 0.19898825 | 174391.46371224275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 8 | ok | 62.925361 | 0.2017545 | 0.2121978 | 0.21505949 | 159669.14955520665 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 32 | 64 | ok | 60.839953 | 0.388953 | 0.45820025000000003 | 0.47380131999999997 | 83774.17310309748 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.256226 | 0.1704995 | 0.18341365 | 0.18672717 | 371495.43397283956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 2 | ok | 63.921269 | 0.2347245 | 0.2451976 | 0.25050911 | 278980.2261431149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 4 | ok | 63.884122 | 0.212844 | 0.22355845000000002 | 0.22676137 | 301712.482335677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 8 | ok | 63.695727 | 0.206694 | 0.2299115 | 0.24670078 | 303122.8186405756 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 64 | 64 | ok | 66.585116 | 0.39840200000000003 | 0.49268595000000004 | 0.50139187 | 159171.7102129792 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 1 | ok | 63.754559 | 0.212405 | 0.2204438 | 0.22818852999999997 | 600073.6777962518 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 2 | ok | 64.232675 | 0.26226550000000004 | 0.300169 | 0.3042459 | 467973.33107731194 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 4 | ok | 64.974769 | 0.265147 | 0.29269205 | 0.29972232 | 484339.5613851203 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 8 | ok | 64.543048 | 0.25477550000000004 | 0.2783772 | 0.29448925 | 498231.7831863432 | - |
| `full_mlp_capacity_search_hd32_depth4` | `jit` | `bf16` | 128 | 64 | ok | 72.847665 | 0.42443200000000003 | 0.5240126 | 0.5284196 | 296367.7126826164 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1372.197283 | 0.0653745 | 0.0670975 | 0.07479615999999997 | 15199.383999365275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1357.460542 | 0.066341 | 0.06981615 | 0.07289471 | 14999.45251998302 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1320.901168 | 0.07182749999999999 | 0.07645335 | 0.08012401 | 13840.72317224946 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1379.835448 | 0.0661935 | 0.07195800000000001 | 0.07353116999999999 | 14949.26965343108 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1564.061989 | 0.066279 | 0.06878255 | 0.07080513999999999 | 15024.020403821632 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1362.446868 | 0.070157 | 0.0724054 | 0.07614930999999998 | 28337.450292570007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1323.768735 | 0.074988 | 0.07728934999999999 | 0.08165848 | 26529.231100642086 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1345.177633 | 0.0725185 | 0.0740272 | 0.08033981 | 27494.74987751089 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1414.16465 | 0.0695655 | 0.07206975 | 0.07716323999999998 | 28646.277201430592 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1576.858424 | 0.072823 | 0.0744663 | 0.07829979999999999 | 27393.845424748422 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1322.566958 | 0.07020799999999999 | 0.072413 | 0.07598187999999999 | 56716.88095722214 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1365.708645 | 0.074306 | 0.07608325 | 0.07670769000000001 | 53778.28771545263 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1365.269564 | 0.0696205 | 0.0712411 | 0.07230069 | 57388.72684544211 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1380.578321 | 0.0741105 | 0.07616325 | 0.07965150999999998 | 53728.589492890766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1647.761649 | 0.0728785 | 0.07441195 | 0.07498938999999999 | 54796.742443255236 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1376.952816 | 0.070601 | 0.0726478 | 0.07466481999999999 | 113004.08651027842 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1367.630189 | 0.070016 | 0.07216209999999999 | 0.07901485999999999 | 113515.17328942569 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1403.368796 | 0.074742 | 0.07641025 | 0.07958617999999999 | 106693.26885503452 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1347.820687 | 0.070227 | 0.07205310000000001 | 0.07328703 | 113578.05808211524 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1633.87862 | 0.071138 | 0.0734812 | 0.07891536 | 111924.24071994149 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1374.679988 | 0.07272500000000001 | 0.07591165 | 0.07978472 | 218860.88381496407 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1310.773305 | 0.07051199999999999 | 0.07257225 | 0.07538260999999999 | 226573.24674081465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1352.874761 | 0.07729349999999999 | 0.07946805 | 0.08107782 | 206838.1199552506 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1401.744051 | 0.07199 | 0.0750986 | 0.08252098999999997 | 221088.65157478684 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1577.846272 | 0.070477 | 0.07254419999999999 | 0.07426732 | 225905.74082963882 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1396.986616 | 0.0753805 | 0.07865865 | 0.08600795999999998 | 420342.91575066943 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1310.361331 | 0.078492 | 0.0805766 | 0.08352894 | 406685.3996128863 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1404.267559 | 0.087614 | 0.08955724999999999 | 0.09527803 | 363662.3159016348 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1395.798836 | 0.078813 | 0.08026865 | 0.08506552999999999 | 404148.2789724429 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1608.626756 | 0.0810225 | 0.08297645 | 0.08636836999999999 | 394110.6081274475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1361.031788 | 0.0841355 | 0.08677595 | 0.09045044999999999 | 756625.4975699317 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1374.18236 | 0.104019 | 0.10535894999999999 | 0.10966849999999999 | 614781.1840070823 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1412.718004 | 0.0949565 | 0.0970635 | 0.09877045 | 672348.8549268673 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1403.403868 | 0.094624 | 0.0978152 | 0.10569135999999998 | 671878.9072645437 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1498.367226 | 0.129995 | 0.13496334999999998 | 0.13623529 | 490152.67949620885 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1375.323648 | 0.089336 | 0.09201725000000001 | 0.09640193999999999 | 1426545.2884664035 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1363.255982 | 0.124487 | 0.12989565 | 0.13389024 | 1022511.2236583534 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1393.182627 | 0.1367345 | 0.1415743 | 0.14972686 | 932466.5473983674 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1439.710623 | 0.1128275 | 0.11735444999999999 | 0.12462327999999999 | 1129015.0155468895 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1598.004984 | 0.1387945 | 0.1437313 | 0.14466065 | 920574.4211780446 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1425.116412 | 0.12617 | 0.13281754999999998 | 0.13905235 | 7879.866704174831 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1361.595895 | 0.1223755 | 0.1277232 | 0.13396097 | 8121.729785501868 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1333.211058 | 0.132067 | 0.14596815 | 0.14866373 | 7454.6700153044385 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1392.392797 | 0.136877 | 0.15008095 | 0.15319872999999998 | 7277.29476755229 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1549.249921 | 0.2513615 | 0.2778303 | 0.29699350999999996 | 4014.7069958437346 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1393.443759 | 0.12535849999999998 | 0.1332087 | 0.13566361999999998 | 15830.166376631634 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1315.855751 | 0.1266965 | 0.13071885 | 0.13758644999999997 | 15696.312763472417 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1375.794855 | 0.1279765 | 0.13933664999999998 | 0.14247996 | 15481.074386562428 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1417.505392 | 0.14019900000000002 | 0.1521767 | 0.16054058999999998 | 14138.034437141521 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1535.66082 | 0.2522485 | 0.27061235 | 0.2744348 | 7911.264097378169 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1374.400051 | 0.127382 | 0.13304844999999998 | 0.13767947 | 31278.590586708164 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1367.189657 | 0.1375595 | 0.1535628 | 0.15575782999999999 | 28515.18847256397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1378.676845 | 0.1405085 | 0.1523961 | 0.15340502 | 28072.91771940108 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1350.041575 | 0.1489925 | 0.15826455 | 0.16744218 | 26755.67029576119 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1583.841728 | 0.25397349999999996 | 0.26902509999999996 | 0.26974561 | 15837.145994255232 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1362.105953 | 0.1873325 | 0.1977861 | 0.19941674999999998 | 42411.89220975183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1368.972862 | 0.1971635 | 0.20393275 | 0.20756022 | 40461.27880906272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1405.802313 | 0.2068955 | 0.2183084 | 0.22159952 | 38509.57916154248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1387.873171 | 0.24078349999999998 | 0.2598872 | 0.26388153999999997 | 33060.67915726676 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1548.778458 | 0.439556 | 0.5182658 | 0.5570255000000001 | 18436.5016922404 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1334.296015 | 0.188125 | 0.19581965 | 0.19988985 | 84690.70478228296 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1367.099209 | 0.220111 | 0.2320333 | 0.23669208 | 72649.88079062686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1399.774006 | 0.223358 | 0.2405432 | 0.24818705999999996 | 70848.80051209513 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1389.432103 | 0.247785 | 0.2671548 | 0.26916216 | 64239.649868212364 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1600.059376 | 0.42095550000000004 | 0.49817155 | 0.50793333 | 37168.282371901376 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1367.990268 | 0.204721 | 0.2126698 | 0.21380638 | 155848.50745832513 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1364.852185 | 0.225957 | 0.23149409999999998 | 0.23400428 | 142535.1534003409 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1389.192042 | 0.2363135 | 0.25055425 | 0.26099861999999996 | 134680.18002699665 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1364.470246 | 0.24755100000000002 | 0.26066754999999997 | 0.26617711 | 129929.00922552809 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1583.008721 | 0.435291 | 0.5140123 | 0.51613243 | 72106.62478713898 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1396.568598 | 0.22107500000000002 | 0.2293639 | 0.23289917 | 288120.14033611736 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1371.956701 | 0.2589915 | 0.27797485 | 0.28235694 | 242510.64602789914 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1431.123674 | 0.2606195 | 0.27706815 | 0.28090169 | 246635.6774076247 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1398.89784 | 0.268977 | 0.2878128 | 0.29379422 | 238283.71980842884 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1578.888548 | 0.427023 | 0.5019971 | 0.5069496 | 144763.41857365327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1337.22099 | 0.25970649999999995 | 0.2684317 | 0.27164245000000004 | 491646.1634852551 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1364.64498 | 0.315135 | 0.3575437 | 0.35942461000000003 | 395089.1163628195 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1324.354247 | 0.289293 | 0.3112506 | 0.3128546 | 439472.0622116651 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1435.886057 | 0.3109465 | 0.33888925 | 0.34597470999999996 | 410739.5294632129 | - |
| `full_mlp_capacity_search_hd32_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1545.158586 | 0.4503505 | 0.5304656 | 0.53819261 | 277585.34751863853 | - |
