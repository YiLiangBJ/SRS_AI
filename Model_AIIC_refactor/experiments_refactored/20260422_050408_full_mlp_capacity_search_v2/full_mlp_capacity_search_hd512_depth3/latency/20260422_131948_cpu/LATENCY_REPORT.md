# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`467789.820` samples/s, p50=`0.278` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.081` ms, throughput=`11918.911` samples/s

### full_mlp_capacity_search_hd512_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`588636.573` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.043` ms, throughput=`22620.964` samples/s

### full_mlp_capacity_search_hd512_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`503296.791` samples/s, p50=`0.233` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`15565.452` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,672`
- MACs / sample: `86,016`
- FLOPs / sample estimate: `172,760`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.08609 | 0.09284479999999999 | 0.09794396999999999 | 11502.836369391965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.108386 | 0.11489065 | 0.12018531999999998 | 9178.660422240408 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0820755 | 0.09903994999999999 | 0.09994866 | 11646.408515481222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.08081250000000001 | 0.1006572 | 0.10455278 | 11918.9113552422 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0855105 | 0.1018638 | 0.10433211999999999 | 10982.161016930539 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09582750000000001 | 0.1020498 | 0.10705964999999999 | 20701.524998540543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.094994 | 0.10133465 | 0.10797108999999998 | 20852.261104506735 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.10119349999999999 | 0.10763745 | 0.11857248 | 19557.266512151225 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.1001595 | 0.1071005 | 0.11432565999999998 | 19786.40574992951 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.10117000000000001 | 0.12363465 | 0.12566351 | 18760.32521398965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.10921249999999999 | 0.11541494999999999 | 0.12673806999999995 | 36290.43525659334 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.108252 | 0.11505485 | 0.12329185999999999 | 36597.17960175681 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.10241600000000001 | 0.10908694999999999 | 0.11278115 | 38688.101938506035 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1011405 | 0.1061582 | 0.12160438 | 39144.113948515704 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.0964605 | 0.10182465 | 0.10223894 | 41263.94773013214 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1093775 | 0.11690225 | 0.11899420999999999 | 72455.64310813697 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1250755 | 0.1314036 | 0.14008388 | 63700.36368130135 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.11525099999999999 | 0.1204064 | 0.12463007999999999 | 69105.79008045123 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1109685 | 0.11623254999999999 | 0.12118415999999999 | 71691.73129078811 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.127829 | 0.13235539999999998 | 0.13374146 | 62441.129722383615 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.19593 | 0.2081855 | 0.21456023 | 80894.07365971665 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.28793749999999996 | 0.295168 | 0.30308142 | 59146.35543854435 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.228551 | 0.23959155 | 0.25046474 | 73496.25063470904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1641145 | 0.17245055 | 0.17601665 | 96947.21696653854 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.143104 | 0.16380495 | 0.18503646999999993 | 107977.19305728245 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2188725 | 0.22926939999999998 | 0.23643689999999998 | 145110.80434283987 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.288316 | 0.34625975 | 0.35384171999999997 | 104683.8635647355 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.2554765 | 0.26329525000000004 | 0.27041595 | 132561.21946379816 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.19227450000000001 | 0.20026449999999998 | 0.20910378 | 165740.6082825347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.1838695 | 0.1902512 | 0.19326723999999998 | 173671.97922014768 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.265244 | 0.28073845 | 0.28629263 | 239436.7967666455 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2898535 | 0.317198 | 0.32258289 | 223681.77688329926 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.27262450000000005 | 0.32627665 | 0.33783184 | 219789.397799029 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2365825 | 0.2430436 | 0.24717418 | 283874.62606171326 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.188362 | 0.19311625 | 0.19446077 | 340641.5493910553 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3561195 | 0.36728845 | 0.37186792999999996 | 357660.70673532115 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.46797 | 0.5960113 | 0.61142577 | 295816.15654480073 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.360267 | 0.4343825 | 0.44352463 | 335001.2031254147 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.279997 | 0.3469634 | 0.35582481 | 418966.59140761057 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.2777645 | 0.28716955 | 0.29344137 | 467789.8202795366 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.208202 | 0.2156286 | 0.21853968 | 4786.874925085408 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.214706 | 0.22510305 | 0.22869506 | 4630.797619695931 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.216252 | 0.22754745 | 0.2300997 | 4598.809938359391 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2240025 | 0.23476315 | 0.24078424999999998 | 4481.022734289601 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.37662399999999996 | 0.44026034999999997 | 2.452926959999992 | 2219.9402223376687 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.21869349999999999 | 0.2273868 | 0.23587064999999996 | 9107.980758844167 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.23811700000000002 | 0.24457769999999998 | 0.2516428 | 8376.153626698622 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2162265 | 0.22920965 | 0.23567561999999997 | 9241.709424205459 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.22554849999999999 | 0.24412649999999997 | 0.25279929 | 8823.766355843614 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.37381600000000004 | 0.5851081999999997 | 0.71549413 | 4553.44003743292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.236415 | 0.24357905 | 0.24921641 | 16842.73358576661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.285629 | 0.29222905 | 0.29494873 | 14379.767207386658 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.2338895 | 0.2448564 | 0.25392403999999996 | 17222.548828724593 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.230972 | 0.2498097 | 0.2561501 | 17181.411739904008 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.38746749999999996 | 0.5404702 | 0.57339469 | 9532.256919107838 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.25358250000000004 | 0.26276279999999996 | 0.26689674 | 31424.657263047473 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.30663149999999995 | 0.33963745 | 0.34640033 | 26058.373884506156 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.27360799999999996 | 0.28604835 | 0.29427222 | 29051.656605351694 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.256196 | 0.26914945 | 0.27348485 | 31525.04617631138 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.36904400000000004 | 0.5316535499999999 | 1.4113460099999968 | 19061.005319354754 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.304847 | 0.31671095 | 0.32244729 | 52269.08254366183 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.35785 | 0.42010274999999997 | 0.42300877 | 45703.84173353757 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.32664649999999995 | 0.33774865 | 0.34015650999999997 | 51359.02397310842 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.30821 | 0.3312597 | 0.33516537999999996 | 53448.4756027084 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.430534 | 0.6566736 | 0.673855 | 35223.09648882965 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.382218 | 0.3912324 | 0.39319182 | 83438.65725506424 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.4652885 | 0.5934673 | 0.60087219 | 71544.275129949 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.37088 | 0.4419189 | 0.45831795 | 85973.72062775001 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.3370115 | 0.38265340000000003 | 0.39016569 | 93008.88810373886 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.4239495 | 0.61540615 | 0.6220252 | 71730.93010807161 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.5604025 | 0.57080775 | 0.59741548 | 113677.65252237211 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.560435 | 0.75770425 | 0.76358934 | 117554.66604225546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.4705965 | 0.6292014 | 0.64983788 | 137952.6787823607 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.401054 | 0.49610129999999997 | 0.5011769 | 156540.60907994234 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.41721600000000003 | 0.5373128 | 3.5755523399999984 | 115792.43858007206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9098029999999999 | 0.9195407 | 0.9209595399999999 | 140468.63981648124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.5884020000000001 | 0.8822035999999993 | 0.98889502 | 195844.70176126203 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.4798095 | 0.7910286 | 0.81514347 | 237127.04870823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.453446 | 0.57737105 | 0.6536612799999998 | 263295.7460828484 | - |
| `full_mlp_capacity_search_hd512_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.4778435 | 0.5955993999999999 | 0.63631785 | 253495.0330618916 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 1 | ok | 58.821057 | 0.046273999999999996 | 0.0504366 | 0.05522146999999999 | 21343.487141616173 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.998861 | 0.0567255 | 0.06112405 | 0.06706179 | 17534.550955931114 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.90477 | 0.0535835 | 0.05725134999999999 | 0.059984939999999994 | 18520.41857627616 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.647718 | 0.052516 | 0.054602 | 0.05636707 | 18959.73256159638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 1 | 64 | ok | 58.970331 | 0.0434905 | 0.0485288 | 0.052345099999999985 | 22620.964476489804 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 1 | ok | 59.192596 | 0.048962 | 0.052077799999999994 | 0.055072239999999995 | 40594.69605939166 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 2 | ok | 58.812904 | 0.048016500000000004 | 0.052653599999999995 | 0.05543707 | 40488.69859200551 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 4 | ok | 59.157712 | 0.0539785 | 0.0564021 | 0.05894081 | 37504.74903884705 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 8 | ok | 59.424646 | 0.0528855 | 0.05804364999999999 | 0.06175106999999999 | 37304.37121430578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 2 | 64 | ok | 58.26257 | 0.0455435 | 0.0495491 | 0.053173109999999996 | 43433.78844704661 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 1 | ok | 59.162593 | 0.049571000000000004 | 0.055977099999999995 | 0.059241459999999996 | 78889.27062530397 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 2 | ok | 59.760741 | 0.0563975 | 0.0632715 | 0.06659328 | 69425.32837312504 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 4 | ok | 59.111233 | 0.0570615 | 0.0583563 | 0.06417853999999999 | 69752.71268299625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 8 | ok | 60.498069 | 0.055760500000000005 | 0.058398 | 0.06145936999999999 | 71330.46656188222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 4 | 64 | ok | 64.384908 | 0.074514 | 0.07921755 | 0.08396984999999998 | 53230.322014171514 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 1 | ok | 59.551546 | 0.056225 | 0.06340794999999999 | 0.06715797999999999 | 139707.19467603823 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 2 | ok | 59.555171 | 0.0593145 | 0.060909 | 0.06426545 | 134817.72138502292 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 4 | ok | 59.388101 | 0.057638499999999995 | 0.059649549999999996 | 0.062317399999999995 | 137989.4182814591 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 8 | ok | 59.831527 | 0.061913499999999996 | 0.06720775 | 0.06986908 | 128034.70252577259 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 8 | 64 | ok | 64.365009 | 0.07802200000000001 | 0.082166 | 0.08556990999999999 | 101607.66182734775 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.69833 | 0.0633405 | 0.06717649999999999 | 0.07422003999999999 | 250079.0093370124 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 2 | ok | 60.495979 | 0.08778949999999999 | 0.09113989999999998 | 0.09407023 | 181693.14390006152 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 4 | ok | 60.496039 | 0.084534 | 0.08762745 | 0.09267185999999998 | 188399.26797464426 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 8 | ok | 59.460277 | 0.072255 | 0.07717874999999999 | 0.08460936 | 219526.55804858453 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 16 | 64 | ok | 65.113195 | 0.0862935 | 0.09029459999999999 | 0.09815125999999999 | 184822.68458183753 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 1 | ok | 60.491555 | 0.085863 | 0.0926588 | 0.09365425999999999 | 369239.0629081822 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 2 | ok | 60.787854 | 0.126095 | 0.16238055 | 0.1670073 | 232171.51322282062 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 4 | ok | 60.272876 | 0.130917 | 0.1367573 | 0.1398486 | 243182.78849496145 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 8 | ok | 60.631021 | 0.12826500000000002 | 0.1346846 | 0.1381566 | 249819.3103769164 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 32 | 64 | ok | 66.248519 | 7.334543 | 16.99893225 | 31.60054750999994 | 3420.5238253465827 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 1 | ok | 60.479297 | 0.128172 | 0.13504624999999998 | 0.13710318 | 493914.8913641914 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 2 | ok | 61.251551 | 0.1959695 | 0.23705775 | 0.23867702999999998 | 307173.2439817562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 4 | ok | 60.731272 | 0.1772955 | 0.21196199999999998 | 0.2178269 | 347379.0629645178 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 8 | ok | 60.794893 | 0.152867 | 0.1571189 | 0.18121310999999993 | 416824.98417693283 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 64 | 64 | ok | 65.890385 | 4.5601255 | 9.55953415 | 14.079766539999987 | 13107.560196593085 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 1 | ok | 61.417926 | 0.2161555 | 0.22831705 | 0.27323463999999986 | 583966.5416369969 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 2 | ok | 63.12721 | 0.2726065 | 0.35867745 | 0.363006 | 415180.4234645703 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 4 | ok | 62.667125 | 0.259815 | 0.332056 | 0.3355845 | 481147.76596702024 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 8 | ok | 62.782047 | 0.2719485 | 0.3190654 | 0.32366052 | 459256.71740089974 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `fp32` | 128 | 64 | ok | 68.426315 | 0.2263165 | 0.23242015 | 0.237681 | 588636.5732963363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.629737 | 0.107682 | 0.11421185 | 0.12022474999999998 | 9206.68397891522 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.840754 | 0.12277 | 0.13037985 | 0.13503969 | 8073.566336457803 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 4 | ok | 59.11837 | 0.129247 | 0.1385705 | 0.1444049 | 7672.1664847851725 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 8 | ok | 58.229189 | 0.14501799999999998 | 0.15593055 | 0.16367366 | 6913.318189479395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 1 | 64 | ok | 59.34529 | 0.313861 | 0.341761 | 0.8917183099999979 | 3142.754006649942 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 1 | ok | 59.761089 | 0.124749 | 0.13558974999999998 | 0.13885440000000002 | 15860.569102596311 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 2 | ok | 59.084003 | 0.137661 | 0.14355725 | 0.15036918999999999 | 14638.999348271751 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 4 | ok | 59.191657 | 0.1315605 | 0.14404999999999998 | 0.15706257999999998 | 15012.123039960921 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 8 | ok | 59.53802 | 0.1454725 | 0.16375769999999998 | 0.17130292 | 13615.912535734962 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 2 | 64 | ok | 59.257047 | 0.2903425 | 0.3488557 | 0.3812491999999999 | 6544.314597244227 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 1 | ok | 59.300818 | 0.1345815 | 0.1406798 | 0.14221278 | 29636.711706071394 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 2 | ok | 59.918431 | 0.17436400000000002 | 0.19198674999999998 | 0.19635562999999998 | 22505.504002322567 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 4 | ok | 59.778635 | 0.1486135 | 0.15872205 | 0.16793075 | 26610.48660734125 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 8 | ok | 59.562078 | 0.158466 | 0.17132514999999998 | 0.17464565999999998 | 25189.012049163914 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 4 | 64 | ok | 64.429515 | 0.2922015 | 0.41925585 | 0.42837372 | 12811.581669829526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 1 | ok | 60.263064 | 0.1625095 | 0.17352145 | 0.17826077999999998 | 48687.26357008367 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 2 | ok | 59.795667 | 0.200735 | 0.2290447 | 0.2345191 | 38108.096855920056 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.993557 | 0.196122 | 0.2085938 | 0.21280517 | 40496.34343329012 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 8 | ok | 59.232924 | 0.17093999999999998 | 0.1880312 | 0.19993815999999995 | 46278.65791429262 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 8 | 64 | ok | 60.098585 | 0.30271800000000004 | 0.36605265 | 0.37163839 | 25229.358521483904 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 1 | ok | 60.33411 | 0.212918 | 0.22069095 | 0.22445714 | 74770.21943283611 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 2 | ok | 60.404313 | 0.2578575 | 0.33280794999999996 | 0.33923976 | 56260.15539758173 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.997612 | 0.2256935 | 0.26676995000000003 | 0.27488764 | 67933.28272303772 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.990106 | 0.211959 | 0.2259964 | 0.23109015 | 75020.43603565384 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 16 | 64 | ok | 64.779277 | 0.34856149999999997 | 0.3983808 | 0.49570668999999984 | 49164.16615669086 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 1 | ok | 60.270705 | 0.302044 | 0.31023795000000004 | 0.31581052 | 105527.17717241381 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 2 | ok | 61.062446 | 0.3739085 | 0.5033188 | 0.51293533 | 84586.87111313414 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 4 | ok | 60.471302 | 0.3108095 | 0.3462558 | 0.35476747999999997 | 103609.37504249603 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 8 | ok | 60.188674 | 0.258518 | 0.30376064999999997 | 0.31230672 | 119546.90527431981 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 32 | 64 | ok | 65.129524 | 0.33286550000000004 | 0.3997245 | 0.5589500399999997 | 95129.97549095094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 1 | ok | 60.277919 | 0.4827165 | 0.49293175 | 0.50340025 | 132196.18723887898 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 2 | ok | 61.533573 | 0.5101525 | 0.6797133000000001 | 0.68117809 | 135025.64917694905 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 4 | ok | 61.344588 | 0.353648 | 0.4641954 | 0.46680767 | 162397.00414066686 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 8 | ok | 61.217579 | 0.3588255 | 0.37827649999999996 | 0.38552971999999996 | 189923.03962328762 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 64 | 64 | ok | 60.326646 | 0.3695045 | 0.4331056 | 0.44919374999999995 | 174262.81926309917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 1 | ok | 61.42984 | 0.840041 | 0.8526117 | 0.8840405499999998 | 152080.21375824543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 2 | ok | 63.255137 | 0.5178905 | 0.69776925 | 0.70062858 | 229181.76416458792 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 4 | ok | 62.987333 | 0.440794 | 0.57936625 | 0.6741034299999996 | 264784.1068304577 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 8 | ok | 62.32086 | 0.46667250000000005 | 0.60871115 | 0.6258543099999999 | 287825.80925936525 | - |
| `full_mlp_capacity_search_hd512_depth3` | `jit` | `bf16` | 128 | 64 | ok | 68.645864 | 0.443962 | 0.5344868 | 0.54357187 | 284636.5045071746 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1432.067654 | 0.0642405 | 0.06745075 | 0.07378422999999999 | 15407.902651638413 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1362.257816 | 0.081647 | 0.0840726 | 0.08900259999999999 | 12187.38147771513 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1358.966588 | 0.0670095 | 0.06876455000000001 | 0.06974709999999999 | 14875.856514628473 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1377.252437 | 0.063934 | 0.0664051 | 0.07166196999999999 | 15565.452416100656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1398.40085 | 0.065835 | 0.06835225 | 0.07908657999999998 | 15077.607461123898 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1315.637419 | 0.0683035 | 0.07027304999999999 | 0.07103688 | 29164.756375124096 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1380.130396 | 0.066993 | 0.07346049999999998 | 0.07889349999999999 | 29366.773082063395 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1369.33282 | 0.0751105 | 0.0769122 | 0.0823501 | 26487.905092776535 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1375.45675 | 0.06666449999999999 | 0.0698308 | 0.07645246999999998 | 29830.247990783646 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1375.7768 | 0.064084 | 0.06732305 | 0.07240422999999999 | 30978.723193336355 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1380.339546 | 0.06970000000000001 | 0.0716154 | 0.07750356 | 57014.02174343747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1325.549887 | 0.083837 | 0.08663554999999999 | 0.09162936999999999 | 47432.25488196483 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1373.114506 | 0.0756325 | 0.07786889999999999 | 0.08083201999999999 | 52595.786130806766 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1391.868401 | 0.073782 | 0.0772575 | 0.08211154999999999 | 53849.04926811065 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1428.862635 | 0.097996 | 0.10289605 | 0.10597287 | 40628.61404218347 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1303.42697 | 0.083708 | 0.0855062 | 0.09016726999999998 | 95337.65498923398 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1320.671461 | 0.096377 | 0.09818265 | 0.10509689999999997 | 82786.85386154105 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1416.313796 | 0.0852205 | 0.08749455 | 0.09360759999999997 | 93350.0896744299 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1359.731781 | 0.0841715 | 0.0863244 | 0.08915118 | 94731.73157763858 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1399.106232 | 0.11591950000000001 | 0.1202958 | 0.12321119999999999 | 68742.0935851425 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1369.05459 | 0.14988400000000002 | 0.15854164999999998 | 0.16994891999999998 | 105817.36240700638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1323.309086 | 0.2635425 | 0.27148585 | 0.27558444 | 64487.27617675576 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1355.483943 | 0.2066935 | 0.21300525 | 0.21811274 | 79577.16672460879 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1359.224709 | 0.133027 | 0.13689825 | 0.14856557999999997 | 119416.88733912306 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1433.02031 | 0.1647785 | 0.1701423 | 0.17496764 | 96659.40271255282 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1399.898252 | 0.168914 | 0.1762733 | 0.18564962999999998 | 188088.07647379057 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1357.954622 | 0.237262 | 0.30156510000000003 | 0.30925417 | 121882.25199399363 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1315.250943 | 0.2148725 | 0.23363889999999998 | 0.23652158 | 144965.6159679626 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1362.199345 | 0.15931 | 0.16249465 | 0.16647974 | 200721.11569828555 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1390.148006 | 0.199141 | 0.207207 | 0.20966215 | 161106.49552189393 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1383.768524 | 0.20671699999999998 | 0.2132346 | 0.21910819999999998 | 307964.5898465037 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1397.072678 | 0.27313849999999995 | 0.27802055 | 0.28625357 | 252511.88167040714 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1303.534405 | 0.2545965 | 0.2805898 | 0.28324058999999996 | 245677.96792996582 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1376.684508 | 0.19692900000000002 | 0.20422605 | 0.20805129 | 324165.72405352216 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1396.380031 | 0.22265200000000002 | 0.2289712 | 0.23566134 | 291502.11220608617 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1361.996097 | 0.27925 | 0.28704155000000003 | 0.29086174000000004 | 457248.41620579833 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1366.014043 | 0.401819 | 0.4220595 | 0.42665010999999997 | 339694.9528707753 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1333.966554 | 0.325715 | 0.4139846 | 0.41945699 | 386174.6809276953 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1408.291352 | 0.2325275 | 0.286973 | 0.29444553 | 503296.7905785986 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1409.174935 | 0.29395150000000003 | 0.30384675 | 0.30967743 | 455937.163879917 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1375.719137 | 0.16568349999999998 | 0.1716304 | 0.17679734 | 6005.007696017862 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1372.504345 | 0.18381799999999998 | 0.19104754999999998 | 0.1942092 | 5441.309222529414 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1361.471111 | 0.1688605 | 0.180255 | 0.18177843000000002 | 5894.136359901625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1318.345816 | 0.1904505 | 0.2064347 | 0.21496544999999997 | 5220.778363757941 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1396.173016 | 0.34598 | 0.4259968 | 0.42829367 | 2691.949890752597 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1316.246081 | 0.1825115 | 0.19120565 | 0.19323476 | 10895.278073645106 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1395.698508 | 0.2104855 | 0.21717229999999998 | 0.22136506 | 9643.939950657747 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1350.518798 | 0.18462699999999999 | 0.19474814999999998 | 0.21215570999999994 | 10771.76113878197 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1350.761839 | 0.2018735 | 0.2145162 | 0.22154656 | 9903.541486628534 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1434.437645 | 0.351615 | 0.4147223 | 0.42031237 | 5409.893277412349 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1365.084595 | 0.194694 | 0.2015217 | 0.20458399 | 20460.6824967557 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1387.362074 | 0.26162799999999997 | 0.2688215 | 0.27277105 | 15468.251028058636 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1328.5596 | 0.206752 | 0.21917315 | 0.22174705 | 19369.011827880764 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1315.602657 | 0.20829599999999998 | 0.22019555 | 0.22492824 | 19244.74757915509 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1415.332102 | 0.35471600000000003 | 0.41476575 | 0.43308108999999995 | 11035.641424862428 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1307.312471 | 0.225537 | 0.23348639999999998 | 0.23738162999999998 | 35314.76535942883 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1389.176829 | 0.305982 | 0.31752615 | 0.32167318 | 27125.349086289396 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1381.510066 | 0.2485575 | 0.26042485 | 0.26486983000000003 | 32485.882244686756 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1377.511307 | 0.2260675 | 0.2412882 | 0.24284414 | 35271.67345132002 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1390.706584 | 0.3635425 | 0.42478615 | 0.43551234999999994 | 21564.730365932985 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1360.364923 | 0.2760185 | 0.28401879999999996 | 0.2861916 | 57809.821440636646 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1368.54806 | 0.3223405 | 0.40460425 | 0.40984733 | 45293.89880425806 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1298.913341 | 0.2961905 | 0.30549565 | 0.30843971 | 55629.624082119895 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1369.602721 | 0.26649100000000003 | 0.29131619999999997 | 0.29718994 | 60272.81131905315 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1410.733567 | 0.3822755 | 0.44834135 | 0.45616041 | 39797.09055410086 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1299.583457 | 0.36443800000000004 | 0.37578629999999996 | 0.37671731 | 87402.3401758076 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1423.126721 | 0.37706649999999997 | 0.47852025 | 0.48330955999999997 | 77438.11484653217 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1363.143973 | 0.3351545 | 0.40470045 | 0.41080785 | 94548.49341998623 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1363.808819 | 0.311878 | 0.3341254 | 0.34092268 | 104744.91862760065 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1448.411309 | 0.3867125 | 0.4577089 | 0.47734447999999996 | 78964.36653995518 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1388.328891 | 0.5487139999999999 | 0.5553939999999999 | 0.5591721000000001 | 116530.61838356446 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1363.390764 | 0.44541 | 0.7494133 | 0.75674487 | 123342.9786257796 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1388.015311 | 0.45536350000000003 | 0.6015457 | 0.6117754999999999 | 139998.57201456546 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1323.459127 | 0.3657765 | 0.4467296 | 0.45048042 | 175368.26650934672 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1390.409096 | 0.45927799999999996 | 0.5233557999999999 | 0.5281157599999999 | 141840.26215626454 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1335.054315 | 0.9425005 | 0.9561786499999999 | 0.97393345 | 135630.91538873027 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1364.630165 | 0.634029 | 0.87351005 | 0.88295174 | 195745.6517703206 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1379.119865 | 0.552175 | 0.77182195 | 0.7944271999999999 | 236867.08052033625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1373.841087 | 0.4505 | 0.5399023 | 0.6074032899999997 | 265219.45252074517 | - |
| `full_mlp_capacity_search_hd512_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1401.280392 | 0.5021845 | 0.5869272999999999 | 0.6130633199999999 | 252911.4654172419 | - |
