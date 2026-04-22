# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth5

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`709212.863` samples/s, p50=`0.179` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.091` ms, throughput=`10949.223` samples/s

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

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 1 | 1 | ok | 0.091773 | 0.0945946 | 0.10011825999999999 | 10836.443392587007 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 1 | 2 | ok | 0.0947885 | 0.09816265 | 0.10494120999999998 | 10481.536459081235 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 1 | 4 | ok | 0.0907015 | 0.0940506 | 0.10162652999999998 | 10949.223196411107 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 1 | 8 | ok | 0.09389700000000001 | 0.09738335 | 0.10257159 | 10614.192363980186 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 1 | 64 | ok | 0.094982 | 0.09826599999999999 | 0.1022912 | 10475.04320955324 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 2 | 1 | ok | 0.1147415 | 0.11995014999999999 | 0.13085405 | 17296.417704336418 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 2 | 2 | ok | 0.11786250000000001 | 0.12447129999999999 | 0.13393381 | 16800.524848396264 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 2 | 4 | ok | 0.11890400000000001 | 0.12882265 | 0.133704 | 16646.807025884787 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 2 | 8 | ok | 0.119316 | 0.12773020000000002 | 0.13607692 | 16629.84818278997 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 2 | 64 | ok | 0.1069375 | 0.13734415 | 0.13983512 | 17257.205357810035 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 4 | 1 | ok | 0.11533399999999999 | 0.12374895 | 0.13353526 | 34247.73061694033 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 4 | 2 | ok | 0.124761 | 0.12979805 | 0.13750721999999999 | 31907.24436434308 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 4 | 4 | ok | 0.113885 | 0.12004515 | 0.12839453 | 34776.179638442496 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 4 | 8 | ok | 0.121209 | 0.12846325 | 0.13526897 | 32709.877614357414 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 4 | 64 | ok | 0.12303150000000002 | 0.13698855 | 0.13905417 | 33759.547622061866 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 8 | 1 | ok | 0.1170535 | 0.12136185 | 0.12665389 | 68029.21242410704 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 8 | 2 | ok | 0.117826 | 0.12496394999999998 | 0.13306273 | 67310.38957739004 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 8 | 4 | ok | 0.1222995 | 0.12676755 | 0.13113445999999998 | 65117.90247421982 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 8 | 8 | ok | 0.122113 | 0.12714175 | 0.13443914999999998 | 65147.343747353385 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 8 | 64 | ok | 0.1108605 | 0.11713 | 0.12020558999999999 | 71726.6983268313 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 16 | 1 | ok | 0.122672 | 0.1281692 | 0.13262633999999998 | 129684.90782158857 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 16 | 2 | ok | 0.1212705 | 0.1274525 | 0.13375923999999997 | 130792.19360252874 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 16 | 4 | ok | 0.13616 | 0.14563405 | 0.15364943999999997 | 116410.8102570729 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 16 | 8 | ok | 0.11968200000000001 | 0.12506035 | 0.13110466999999998 | 132783.65327001325 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 16 | 64 | ok | 0.10554350000000001 | 0.11211549999999999 | 0.11624236999999998 | 150429.97588043372 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 32 | 1 | ok | 0.130996 | 0.13837149999999998 | 0.1423704 | 242544.15478440328 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 32 | 2 | ok | 0.155266 | 0.16189800000000001 | 0.17384049 | 204655.16308522265 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 32 | 4 | ok | 0.160895 | 0.16823169999999998 | 0.17467549999999998 | 197625.16311795692 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 32 | 8 | ok | 0.141461 | 0.15077895 | 0.15751290999999998 | 224344.77304721897 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 32 | 64 | ok | 0.1673895 | 0.17134685 | 0.17482435 | 190563.9179369836 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 64 | 1 | ok | 0.146544 | 0.1560218 | 0.16079315 | 433803.30951256095 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 64 | 2 | ok | 0.1972625 | 0.20445314999999997 | 0.21094557 | 327566.1893549587 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 64 | 4 | ok | 0.1878265 | 0.19548145 | 0.20194801 | 338372.86850206455 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 64 | 8 | ok | 0.18644850000000002 | 0.19383989999999998 | 0.2014761 | 341392.54445624386 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 64 | 64 | ok | 0.1890935 | 0.19429 | 0.19687158 | 337795.61338616454 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 128 | 1 | ok | 0.179149 | 0.18693265 | 0.19751904 | 709212.8634812888 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 128 | 2 | ok | 0.2653435 | 0.27711790000000003 | 0.28548125 | 494187.3147521477 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 128 | 4 | ok | 0.226983 | 0.2540153 | 0.26173205 | 542530.7670045672 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 128 | 8 | ok | 0.2213415 | 0.23314165 | 0.23874104999999998 | 579731.2547069195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `fp32` | 128 | 64 | ok | 0.218123 | 0.22382244999999998 | 0.22588015 | 585201.7117150067 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 1 | 1 | ok | 0.1693625 | 0.17751455 | 0.18324949 | 5868.536335394832 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 1 | 2 | ok | 0.1810755 | 0.1885378 | 0.19093285000000002 | 5494.695695510339 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 1 | 4 | ok | 0.18468 | 0.19572199999999998 | 0.19702534 | 5357.534531185191 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 1 | 8 | ok | 0.1999245 | 0.2102751 | 0.21302675999999998 | 5022.543687340628 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 1 | 64 | ok | 0.30974999999999997 | 0.31931655000000003 | 0.32284975 | 3300.6128181807258 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 2 | 1 | ok | 0.23918650000000002 | 0.24709585 | 0.25177805999999997 | 8314.522419236679 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 2 | 2 | ok | 0.2484685 | 0.25742525 | 0.26564192999999997 | 8023.858461704371 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 2 | 4 | ok | 0.2492725 | 0.26480360000000003 | 0.27457804999999996 | 7976.418516298417 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 2 | 8 | ok | 0.2862635 | 0.30695799999999995 | 0.31343684 | 6981.347654218319 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 2 | 64 | ok | 0.456669 | 0.58923605 | 0.6014036 | 4180.901241761116 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 4 | 1 | ok | 0.270386 | 0.28074585 | 0.28460913 | 14746.759067892752 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 4 | 2 | ok | 0.2945455 | 0.30258995 | 0.30784612 | 13583.292170298353 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 4 | 4 | ok | 0.2973395 | 0.3107307 | 0.3140707 | 13434.634056689858 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 4 | 8 | ok | 0.314942 | 0.33830015 | 0.34340898999999997 | 12593.740721955077 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 4 | 64 | ok | 0.5590065 | 0.65470165 | 0.7311357099999997 | 7009.715606024262 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 8 | 1 | ok | 0.2714265 | 0.282696 | 0.28855178 | 29306.126892745364 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 8 | 2 | ok | 0.297651 | 0.30404275000000003 | 0.30785876 | 27062.561267101595 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 8 | 4 | ok | 0.30237250000000004 | 0.31636385 | 0.32156428 | 26553.273567225184 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 8 | 8 | ok | 0.3166445 | 0.33576075 | 0.34189548999999997 | 25246.610468974097 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 8 | 64 | ok | 0.52658 | 0.6287184499999999 | 0.6868622399999998 | 14767.252775237504 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 16 | 1 | ok | 0.2787405 | 0.28793905 | 0.29048179999999996 | 57231.18530148637 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 16 | 2 | ok | 0.3175515 | 0.3333196 | 0.33945209 | 50358.96182734039 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 16 | 4 | ok | 0.31369400000000003 | 0.33380784999999996 | 0.33710079 | 50973.09875598241 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 16 | 8 | ok | 0.33563 | 0.3644739 | 0.37386843999999997 | 47266.04063189733 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 16 | 64 | ok | 0.519182 | 0.6750527 | 0.682495 | 29657.73747571356 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 32 | 1 | ok | 0.30153450000000004 | 0.31306425 | 0.3183878 | 105613.64232540647 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 32 | 2 | ok | 0.3519355 | 0.3796623 | 0.38494466 | 91196.15152240575 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 32 | 4 | ok | 0.3306935 | 0.3586814 | 0.35992182 | 95761.63779213883 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 32 | 8 | ok | 0.34498249999999997 | 0.37630525 | 0.38633053999999994 | 92544.5155025656 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 32 | 64 | ok | 0.5002935 | 0.6354499999999998 | 0.6738743399999999 | 60564.601383477195 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 64 | 1 | ok | 0.345472 | 0.3578475 | 0.36534663 | 184107.88768245163 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 64 | 2 | ok | 0.4006615 | 0.46926055 | 0.47881229 | 154293.07709624866 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 64 | 4 | ok | 0.3882675 | 0.42554525 | 0.4400005 | 161667.2991057727 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 64 | 8 | ok | 0.3768995 | 0.42093165 | 0.42592894 | 166142.37498136738 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 64 | 64 | ok | 0.5617645 | 0.6572591 | 0.66576094 | 115376.45860180895 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 128 | 1 | ok | 0.448556 | 0.4615783 | 0.47251755 | 284494.59135774773 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 128 | 2 | ok | 0.504778 | 0.615932 | 0.61858828 | 253113.3135309078 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 128 | 4 | ok | 0.456109 | 0.5290036499999999 | 0.53228321 | 287819.46667862317 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 128 | 8 | ok | 0.4470495 | 0.49728954999999997 | 0.50549949 | 285208.7404892355 | - |
| `full_mlp_capacity_search_hd64_depth5` | `bf16` | 128 | 64 | ok | 0.5405314999999999 | 0.69751025 | 8.25356761999999 | 143955.21913020907 | - |
