# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth3

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1150314.054` samples/s, p50=`0.111` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.068` ms, throughput=`14530.250` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `5,552`
- MACs / sample: `5,376`
- FLOPs / sample estimate: `11,000`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 1 | 1 | ok | 0.0722245 | 0.07597414999999999 | 0.07959974999999998 | 13732.607995289165 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 1 | 2 | ok | 0.06944349999999999 | 0.07404799999999999 | 0.07884237999999999 | 14257.659285856656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 1 | 4 | ok | 0.06804 | 0.0721629 | 0.07427395999999999 | 14530.250091758531 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 1 | 8 | ok | 0.0698175 | 0.0737802 | 0.0761829 | 14207.540339469288 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 1 | 64 | ok | 0.0723045 | 0.0760347 | 0.07990763999999999 | 13694.225473779119 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 2 | 1 | ok | 0.0853825 | 0.08830045 | 0.09425916999999999 | 23266.927969779055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 2 | 2 | ok | 0.0869585 | 0.08964704999999999 | 0.09381125 | 22880.31080614199 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 2 | 4 | ok | 0.08276 | 0.0855976 | 0.08869907 | 24044.073748944767 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 2 | 8 | ok | 0.083142 | 0.08662035 | 0.09023528 | 23894.108955225307 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 2 | 64 | ok | 0.0838955 | 0.0868684 | 0.09218170999999999 | 23678.059838719262 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 4 | 1 | ok | 0.083448 | 0.0864965 | 0.09095044999999999 | 47633.20148463162 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 4 | 2 | ok | 0.087102 | 0.09130605 | 0.09489027 | 45547.91868785556 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 4 | 4 | ok | 0.088398 | 0.0915377 | 0.09889592999999998 | 45017.37558154009 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 4 | 8 | ok | 0.084292 | 0.08788135 | 0.09164673999999999 | 47231.602287237576 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 4 | 64 | ok | 0.08416599999999999 | 0.08688834999999999 | 0.09202165 | 47191.55950643292 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 8 | 1 | ok | 0.0906515 | 0.09399695 | 0.10183055999999997 | 87722.35698569314 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 8 | 2 | ok | 0.08937500000000001 | 0.0933218 | 0.09552524 | 88948.80084785998 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 8 | 4 | ok | 0.0869055 | 0.0899711 | 0.09471540999999999 | 91477.74986689987 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 8 | 8 | ok | 0.0901975 | 0.09370365 | 0.10062747999999999 | 88160.7982431316 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 8 | 64 | ok | 0.08936 | 0.09324244999999999 | 0.09685994999999999 | 88943.2035385164 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 16 | 1 | ok | 0.0877445 | 0.09148885 | 0.10056697999999996 | 181089.01045494765 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 16 | 2 | ok | 0.0886585 | 0.0929234 | 0.09850728999999998 | 179166.79824228413 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 16 | 4 | ok | 0.088963 | 0.0923982 | 0.10358204999999997 | 178341.1773727848 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 16 | 8 | ok | 0.0870255 | 0.09102455 | 0.09436803999999999 | 182558.08142409907 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 16 | 64 | ok | 0.0878475 | 0.09093735 | 0.09515313999999998 | 181467.66053479884 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 32 | 1 | ok | 0.088672 | 0.09149275 | 0.09583346 | 358929.17071209755 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 32 | 2 | ok | 0.0893615 | 0.09220185 | 0.10256575999999996 | 355065.7226652654 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 32 | 4 | ok | 0.0989865 | 0.10435875 | 0.10980717 | 320364.89561610675 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 32 | 8 | ok | 0.089458 | 0.09343879999999999 | 0.09852517 | 355257.3761978002 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 32 | 64 | ok | 0.0910195 | 0.09393185 | 0.09964497999999998 | 349403.6444110375 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 64 | 1 | ok | 0.0970245 | 0.1008628 | 0.11161932999999996 | 654102.9836089968 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 64 | 2 | ok | 0.11271149999999999 | 0.11658924999999999 | 0.12228702 | 564417.1000037745 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 64 | 4 | ok | 0.11114299999999999 | 0.11418365 | 0.11929870999999999 | 574053.3277602323 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 64 | 8 | ok | 0.10757900000000001 | 0.11079345 | 0.11504419 | 591998.9492018651 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 64 | 64 | ok | 0.1442835 | 0.15052865 | 0.15476156 | 440766.85719732335 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 128 | 1 | ok | 0.11070050000000001 | 0.114479 | 0.11940051 | 1150314.05371032 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 128 | 2 | ok | 0.146478 | 0.15037915 | 0.16243975999999996 | 869143.3994262567 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 128 | 4 | ok | 0.14432050000000002 | 0.1500928 | 0.15601183999999998 | 886404.7396061425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 128 | 8 | ok | 0.133841 | 0.1401624 | 0.14498476 | 952011.6228719007 | - |
| `full_mlp_capacity_search_hd32_depth3` | `fp32` | 128 | 64 | ok | 0.1777085 | 0.1866485 | 0.18859551 | 716672.6501367613 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 1 | 1 | ok | 0.1277905 | 0.13727109999999998 | 0.14032122 | 7752.744510320376 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 1 | 2 | ok | 0.136852 | 0.143089 | 0.14864659 | 7264.69014788875 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 1 | 4 | ok | 0.1386645 | 0.14635365 | 0.15605613999999998 | 7148.784506467934 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 1 | 8 | ok | 0.1517075 | 0.15918245 | 0.16402467 | 6614.331430654754 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 1 | 64 | ok | 0.25670000000000004 | 0.272892 | 0.28176735999999997 | 3918.0700213965806 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 2 | 1 | ok | 0.133554 | 0.14268295 | 0.14646447999999998 | 14838.712836629182 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 2 | 2 | ok | 0.14381100000000002 | 0.1504189 | 0.15570341 | 13824.132428553083 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 2 | 4 | ok | 0.149035 | 0.1600435 | 0.16205907 | 13280.967364413276 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 2 | 8 | ok | 0.161357 | 0.17327545 | 0.17766815 | 12325.149269882808 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 2 | 64 | ok | 0.250817 | 0.27172155000000003 | 0.2741691 | 7985.41990092809 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 4 | 1 | ok | 0.13473600000000002 | 0.140509 | 0.14508019 | 29514.51427890065 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 4 | 2 | ok | 0.150414 | 0.16543405 | 0.1702875 | 26014.297457882854 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 4 | 4 | ok | 0.1440235 | 0.15578305 | 0.15711632 | 27539.27617143475 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 4 | 8 | ok | 0.155215 | 0.1675717 | 0.17715283 | 25663.188107062713 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 4 | 64 | ok | 0.33208099999999996 | 0.42346265 | 0.44188357999999994 | 11035.866732418363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 8 | 1 | ok | 0.1738345 | 0.18121425 | 0.18474234 | 45855.42609549759 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 8 | 2 | ok | 0.18498199999999998 | 0.1902259 | 0.19373927 | 43159.62555140468 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 8 | 4 | ok | 0.1883785 | 0.20587095 | 0.21067887999999999 | 41918.601830187115 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 8 | 8 | ok | 0.198663 | 0.21160009999999999 | 0.21736845999999999 | 40318.032673532085 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 8 | 64 | ok | 0.3701235 | 0.42033434999999997 | 0.42419236 | 21174.40222280405 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 16 | 1 | ok | 0.169404 | 0.1770147 | 0.18598960999999997 | 93839.26460045119 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 16 | 2 | ok | 0.1898785 | 0.196442 | 0.20040892 | 84262.8313596345 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 16 | 4 | ok | 0.188095 | 0.19860055 | 0.206653 | 84516.08309367741 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 16 | 8 | ok | 0.2082585 | 0.2180704 | 0.22296422 | 76636.39784406485 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 16 | 64 | ok | 0.373254 | 0.5192143 | 0.5382841199999999 | 39301.558252745075 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 32 | 1 | ok | 0.18074600000000002 | 0.18960079999999999 | 0.19435175999999998 | 175976.67605135616 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 32 | 2 | ok | 0.2110135 | 0.21736394999999997 | 0.22116735 | 152264.10055977994 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 32 | 4 | ok | 0.19999499999999998 | 0.21331745000000002 | 0.21599027 | 159249.52070870812 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 32 | 8 | ok | 0.2105505 | 0.22501249999999998 | 0.23030393999999998 | 150676.3862980921 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 32 | 64 | ok | 0.361856 | 0.40879519999999997 | 0.41749584 | 86868.67107438303 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 64 | 1 | ok | 0.1961075 | 0.20220975 | 0.20620836999999997 | 325478.5526416297 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 64 | 2 | ok | 0.2400175 | 0.2484054 | 0.25356033 | 265429.36766927637 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 64 | 4 | ok | 0.22366 | 0.23584455000000001 | 0.23881468 | 285312.40281546285 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 64 | 8 | ok | 0.2415715 | 0.2587737 | 0.26019213 | 264857.84210549406 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 64 | 64 | ok | 0.3853695 | 0.43442264999999997 | 0.44507526 | 165114.67472151088 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 128 | 1 | ok | 0.227989 | 0.23875255 | 0.25263767 | 556960.9633893024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 128 | 2 | ok | 0.2964715 | 0.3054485 | 0.30872979 | 446128.42434062733 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 128 | 4 | ok | 0.2875225 | 0.29857975 | 0.30538095 | 458766.4315440916 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 128 | 8 | ok | 0.27841249999999995 | 0.3054445 | 0.31372998 | 456402.3443391668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `bf16` | 128 | 64 | ok | 0.41396299999999997 | 0.5264787999999999 | 0.5613877599999999 | 293667.99493312585 | - |
