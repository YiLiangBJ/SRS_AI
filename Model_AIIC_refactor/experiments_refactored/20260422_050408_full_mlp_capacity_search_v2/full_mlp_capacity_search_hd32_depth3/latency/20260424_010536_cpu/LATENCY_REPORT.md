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

### full_mlp_capacity_search_hd32_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2301540.666` samples/s, p50=`0.053` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.034` ms, throughput=`29306.520` samples/s

### full_mlp_capacity_search_hd32_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3508554.569` samples/s, p50=`0.036` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.020` ms, throughput=`50049.449` samples/s

### full_mlp_capacity_search_hd32_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2602205.695` samples/s, p50=`0.049` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.033` ms, throughput=`30251.017` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0349755 | 0.04072485 | 0.042286620000000004 | 27745.623682776517 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.035019999999999996 | 0.041125 | 0.042665659999999994 | 27325.62969181062 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035067 | 0.04263595 | 0.06473907999999992 | 26932.588270211425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.034423499999999996 | 0.03803165 | 0.039422139999999994 | 28625.604572768578 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0339915 | 0.035383399999999995 | 0.03573325 | 29306.519821464677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0390645 | 0.044404049999999994 | 0.04678323 | 50543.74965882969 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0367595 | 0.042693749999999996 | 0.04643556 | 52186.51041329629 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038136 | 0.04305805 | 0.045782359999999994 | 51252.981642207036 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.037295499999999995 | 0.04183605 | 0.04832674999999999 | 51941.76497078535 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036060999999999996 | 0.03930089999999999 | 0.06080231999999995 | 53352.22713536953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038082000000000005 | 0.044183099999999996 | 0.04936408 | 102098.63749368266 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.039155499999999996 | 0.0444202 | 0.04525579 | 101195.88233954759 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.039815500000000004 | 0.0461346 | 0.04936469999999999 | 98021.29316551436 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.037095500000000003 | 0.044657749999999996 | 0.04746185 | 102589.35532849112 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.036498 | 0.0398223 | 0.08904993999999986 | 102965.29760574794 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0376535 | 0.0436434 | 0.046404219999999996 | 205646.64559474037 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0374165 | 0.045158699999999996 | 0.046231709999999995 | 206889.73863102094 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0370165 | 0.044983149999999986 | 0.0478723 | 206603.67320670592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.039150000000000004 | 0.04334435 | 0.04587429999999999 | 204135.06188099232 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.037224999999999994 | 0.0402835 | 0.050697599999999995 | 210616.21565836787 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0405355 | 0.0459432 | 0.05026625999999999 | 389687.88436012034 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.038428500000000004 | 0.0439952 | 0.06254855999999993 | 394213.148092575 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0387715 | 0.0489635 | 0.11693126999999989 | 369988.04938600486 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.038734500000000005 | 0.049193449999999986 | 0.10455454999999986 | 374554.1050427624 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.038639 | 0.0435252 | 0.06781787999999994 | 397480.9643881899 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0410045 | 0.0471992 | 0.04830238 | 758108.8028277459 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.040476 | 0.04644865 | 0.04894867 | 758025.1171622571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.048179 | 0.0538463 | 0.05691996999999999 | 646283.1649294441 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.040752 | 0.04620335 | 0.04881728 | 761774.8974222488 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.0429035 | 0.048016249999999996 | 0.050042949999999996 | 737891.659057159 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0458645 | 0.05536539999999999 | 0.07620432999999993 | 1338232.379034248 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0593395 | 0.06649775 | 0.06838682 | 1048253.7403003771 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.059202500000000005 | 0.0647146 | 0.0658824 | 1086234.824960047 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.0534555 | 0.0607566 | 0.06207725 | 1167487.7997524925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.1125365 | 6.00095645 | 6.00885531 | 42661.00811055088 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.05301 | 0.06074825 | 0.08232826999999993 | 2301540.665706253 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0810005 | 0.0930194 | 0.09458277 | 1555722.5796602888 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.0831915 | 0.09414890000000001 | 0.09556685999999999 | 1515448.218673495 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0693505 | 0.08213539999999998 | 0.13295418999999986 | 1749033.8637551812 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.20309300000000002 | 0.21581365 | 0.22949816999999995 | 628051.8782627417 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0475825 | 0.05314955 | 0.05458614 | 20653.296823812092 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.050702 | 0.059231349999999995 | 0.06188687 | 19041.07617115947 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.052122 | 0.05871515 | 0.09312229999999987 | 18378.52858559579 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.053556 | 0.0602042 | 0.06433626999999999 | 18233.309046565682 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.114552 | 0.12772909999999998 | 0.13322273999999998 | 8621.703448577919 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.0514205 | 0.06057449999999999 | 0.0632979 | 38440.7952170425 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0542605 | 0.06304284999999998 | 0.06821237999999999 | 35672.49793099512 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.054516999999999996 | 0.062403349999999996 | 0.06677825999999999 | 35327.462615595876 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0568105 | 0.06697464999999998 | 0.10919705999999985 | 33333.333333333336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1422675 | 0.15809289999999998 | 0.16351735 | 13923.39958179677 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0540275 | 0.06095425 | 0.062287779999999994 | 72439.38543874181 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0593375 | 0.07631594999999998 | 0.14029215999999997 | 62229.65105967761 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.057338 | 0.06994310000000001 | 0.11080659999999984 | 65916.18888415757 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.060302499999999995 | 0.0717671 | 0.07599299 | 63582.89620092196 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.15837299999999999 | 0.30942005 | 0.32693403 | 21114.608511404265 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.07212099999999999 | 0.08128715 | 0.08377425 | 109520.62821032341 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.069388 | 0.09877294999999994 | 0.15778480999999983 | 106797.10152666457 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.069825 | 0.08153534999999999 | 0.12720143999999986 | 108453.1062325289 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.06774 | 0.07487085 | 0.08114783999999998 | 116144.79423062349 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1848675 | 0.20815519999999998 | 0.21390792999999997 | 42600.187653826615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.06503500000000001 | 0.0751889 | 0.07680463 | 237446.50033539318 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.07568 | 0.08953625 | 0.09275834999999999 | 205328.79556695127 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0773905 | 0.0889005 | 0.08984527 | 203176.87359553983 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.07723 | 0.08882709999999999 | 0.09550467999999998 | 202659.65464261602 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 5.9996410000000004 | 6.00489555 | 6.0103884899999995 | 2667.2625775808747 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07271749999999999 | 0.08737914999999999 | 0.15050634999999993 | 413572.62645500025 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.07839299999999999 | 0.0918551 | 0.09537873999999999 | 395975.3069798567 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0806135 | 0.0963887 | 0.17143096999999982 | 371211.353128024 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.085427 | 0.10592354999999998 | 0.12196268999999997 | 359644.3117756539 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.2087705 | 0.22738974999999997 | 0.23095146 | 153540.79958098716 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.08078099999999999 | 0.09578619999999999 | 0.16500181999999977 | 757621.1359250448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1044875 | 0.1197958 | 0.15268909999999988 | 604959.3051437421 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0997045 | 0.11917805 | 0.15239996999999988 | 611824.0348189058 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1069605 | 0.12526204999999999 | 0.12749238999999998 | 582617.851301723 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.25967850000000003 | 0.33446895 | 0.35476958 | 236877.41675109742 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.099784 | 0.12070734999999999 | 0.19504483999999983 | 1184153.4366815581 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1184875 | 0.1397947 | 0.14371239 | 1048509.7972591626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.13880599999999998 | 0.1555149 | 0.18720043999999988 | 916635.0499186558 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.133399 | 0.15546734999999998 | 0.18005008999999994 | 940166.7591408814 | - |
| `full_mlp_capacity_search_hd32_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.279668 | 0.30220579999999997 | 0.3053288 | 454554.91367824626 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.17659 | 0.020453 | 0.023208649999999997 | 0.02570935 | 47474.49907282303 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 2 | ok | 45.120368 | 0.019813499999999998 | 0.02046495 | 0.022871439999999993 | 50353.02505868645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.264575 | 0.019805 | 0.02034555 | 0.0240488 | 50049.44885546921 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.709117 | 0.0204435 | 0.0211325 | 0.024421819999999993 | 48717.5589774769 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.734833 | 0.0202535 | 0.0219338 | 0.025425489999999988 | 49078.502045591966 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 1 | ok | 45.177532 | 0.021402 | 0.0218768 | 0.02532405999999999 | 92773.84519756191 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 2 | ok | 45.375864 | 0.021209 | 0.0238029 | 0.02671117 | 91846.59414459593 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 4 | ok | 46.425694 | 0.021421 | 0.0217675 | 0.023979039999999993 | 92873.70754626735 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.418714 | 0.0211005 | 0.02382675 | 0.026028639999999992 | 92199.54932860288 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 2 | 128 | ok | 52.818544 | 0.021394 | 0.0218759 | 0.02545258 | 92875.51894196209 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 1 | ok | 47.253465 | 0.021758 | 0.02466835 | 0.025735239999999996 | 179753.79123214932 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.349317 | 0.0215 | 0.02346945 | 0.026460089999999988 | 181882.3366787558 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 4 | ok | 46.556276 | 0.021603 | 0.0219589 | 0.025543579999999986 | 184536.40303488568 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.228988 | 0.0217345 | 0.0231444 | 0.025524469999999994 | 183526.81647960696 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 4 | 128 | ok | 51.424472 | 0.021275500000000003 | 0.0219233 | 0.04045533999999994 | 182557.03994713147 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.750783 | 0.0219585 | 0.022468150000000003 | 0.024694419999999995 | 362637.1334985146 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.293137 | 0.022094000000000003 | 0.02265615 | 0.025241149999999993 | 363287.60753313184 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.797267 | 0.021659499999999998 | 0.02228935 | 0.024374409999999992 | 371806.3000718515 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 8 | ok | 45.177988 | 0.021706 | 0.024738899999999998 | 0.02903879999999999 | 357623.09386890964 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 8 | 128 | ok | 51.070557 | 0.021879 | 0.0225114 | 0.02562044999999999 | 366615.95146004803 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.500292 | 0.0224785 | 0.02300025 | 0.02581978999999999 | 715461.7545447473 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.016044 | 0.022933 | 0.028562499999999998 | 0.03103932999999999 | 658805.4703912234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 4 | ok | 45.145886 | 0.022586000000000002 | 0.023157550000000002 | 0.02358987 | 716040.0123158883 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 8 | ok | 45.671616 | 0.022533499999999998 | 0.0282297 | 0.029734399999999994 | 684537.0775229684 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 16 | 128 | ok | 51.569919 | 0.022185 | 0.023309049999999998 | 0.026169539999999998 | 716425.3954892065 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.522028 | 0.024920499999999998 | 0.03076615 | 0.031864109999999994 | 1252944.419385556 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.7253 | 0.024644 | 0.02555545 | 0.027131719999999998 | 1294432.4035263574 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 4 | ok | 46.483179 | 0.0318185 | 0.0329036 | 0.037555889999999995 | 999248.0658304625 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 8 | ok | 45.673331 | 0.025277 | 0.030856150000000002 | 0.03141469 | 1211717.306352428 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 32 | 128 | ok | 54.376634 | 0.0240555 | 0.024697 | 0.026451799999999998 | 1325498.5324245936 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 1 | ok | 47.190537 | 0.0279715 | 0.03169169999999999 | 0.0358439 | 2244347.7504621954 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 2 | ok | 46.441294 | 0.040669 | 0.04529004999999999 | 0.04778791 | 1554854.0356479443 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 4 | ok | 46.34102 | 0.0373995 | 0.04108659999999999 | 0.04451156999999999 | 1707672.1440250687 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 8 | ok | 46.688242 | 0.0363005 | 0.03781965 | 0.04062491999999999 | 1758078.5080945783 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 64 | 128 | ok | 67.704794 | 0.10433100000000001 | 0.11873374999999999 | 0.13412809 | 611278.2362789048 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 1 | ok | 47.914286 | 0.0364845 | 0.037649049999999996 | 0.04004964999999999 | 3508554.56871367 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 2 | ok | 49.047087 | 0.0595895 | 0.0625609 | 0.06488762 | 2147947.9713302646 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.458608 | 0.059441999999999995 | 0.0635309 | 0.06523376 | 2147834.077132749 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 8 | ok | 50.195011 | 0.050469 | 0.055743549999999996 | 0.05920793999999999 | 2502751.07090372 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `fp32` | 128 | 128 | ok | 79.157914 | 0.230465 | 0.24618435 | 0.25645440999999997 | 550757.7953957164 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 1 | ok | 45.962393 | 0.0280535 | 0.030153 | 0.03173097 | 35392.11633530208 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.320717 | 0.030348 | 0.0316382 | 0.03818788 | 32576.770417165088 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 4 | ok | 48.221734 | 0.0310005 | 0.033984349999999997 | 0.03528989 | 31880.43561427223 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 8 | ok | 48.532221 | 0.0314465 | 0.0348483 | 0.0391459 | 31388.166786674086 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 1 | 128 | ok | 90.134898 | 0.139317 | 0.17799885 | 0.18493120999999998 | 7029.922443083638 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 1 | ok | 46.641936 | 0.0313075 | 0.03399835 | 0.03723076999999999 | 63530.14124021 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.542314 | 0.036262 | 0.039398249999999996 | 0.04531541999999999 | 55047.83106040839 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 4 | ok | 46.985177 | 0.0374725 | 0.0405997 | 0.04256715 | 53162.30664869072 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 8 | ok | 47.952111 | 0.036587999999999996 | 0.0401929 | 0.042983299999999995 | 53986.38139542919 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 2 | 128 | ok | 108.518349 | 0.1211625 | 0.17287829999999996 | 0.21317849999999988 | 15458.05541681951 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.030843 | 0.034481 | 0.036746949999999994 | 0.03827401 | 115317.17701244331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 2 | ok | 48.102451 | 0.039284 | 0.0420276 | 0.04390669 | 102328.01347045969 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.538223 | 0.0392405 | 0.0406176 | 0.049961289999999985 | 100925.9450831655 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 8 | ok | 48.84336 | 0.0392165 | 0.04149465 | 0.044617389999999986 | 101068.29184479953 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 4 | 128 | ok | 104.157925 | 0.1462715 | 0.16581545 | 0.17977833999999998 | 26995.36948427236 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 1 | ok | 48.915052 | 0.0449185 | 0.04755185 | 0.04938694999999999 | 177594.57910306743 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 2 | ok | 47.184395 | 0.048147999999999996 | 0.051038299999999995 | 0.05599202999999999 | 165731.18319220634 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 4 | ok | 48.746014 | 0.052414 | 0.0565457 | 0.058695199999999996 | 152286.79563073954 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 8 | ok | 47.628311 | 0.049967 | 0.052524100000000004 | 0.05592216 | 159138.8677587837 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 8 | 128 | ok | 86.059475 | 0.116284 | 0.14930854999999998 | 0.16274138999999999 | 66267.96545252156 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 1 | ok | 46.385089 | 0.048485 | 0.05068485 | 0.05457467999999999 | 333068.1278365435 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 2 | ok | 47.025683 | 0.053052 | 0.0617038 | 0.06596867999999999 | 295383.2338999522 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 4 | ok | 48.521303 | 0.0562075 | 0.06047155 | 0.06484755 | 286273.5415257664 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 8 | ok | 48.244327 | 0.057075 | 0.05971355 | 0.06433993 | 282115.40000966244 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 16 | 128 | ok | 86.355218 | 0.122626 | 0.15238084999999998 | 0.17480917999999995 | 126672.05128444667 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.633839 | 0.052098500000000006 | 0.058706249999999995 | 0.06159314 | 606710.7517866683 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.5317 | 0.0598445 | 0.06392535 | 0.06720095 | 538923.7692328419 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 4 | ok | 53.732584 | 0.06455849999999999 | 0.06960085 | 0.0705635 | 493547.6358297076 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 8 | ok | 49.815161 | 0.06341 | 0.0687951 | 0.07142209 | 503122.82045621925 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 32 | 128 | ok | 92.970779 | 0.1220305 | 0.13028874999999998 | 0.13437727 | 260158.66751933668 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 1 | ok | 48.778047 | 0.061527 | 0.0650576 | 0.06780726999999999 | 1038231.5801495314 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 2 | ok | 48.231186 | 0.072615 | 0.0768679 | 0.07875845 | 881274.4329894069 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 4 | ok | 49.807556 | 0.079237 | 0.0863222 | 0.08819719 | 800901.6149930785 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 8 | ok | 49.64857 | 0.08217849999999999 | 0.0886586 | 0.09109796 | 775833.5604262819 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 64 | 128 | ok | 93.393103 | 0.2349905 | 0.26493524999999996 | 0.27548056 | 269565.2115311911 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 1 | ok | 49.661137 | 0.0780295 | 0.0846325 | 0.08934331 | 1620788.6402976173 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.899535 | 0.0921885 | 0.1007754 | 0.10668683999999998 | 1374953.1656577948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.613598 | 0.1079485 | 0.1162443 | 0.12554919999999997 | 1175461.2904985442 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 8 | ok | 49.842656 | 0.105514 | 0.1158957 | 0.11729302 | 1209941.0267181448 | - |
| `full_mlp_capacity_search_hd32_depth3` | `jit` | `bf16` | 128 | 128 | ok | 151.613619 | 0.209009 | 0.2280025 | 0.23965205999999997 | 604839.5861460736 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 1 | ok | 495.454709 | 0.033002500000000004 | 0.036814299999999994 | 0.04345764 | 29961.39773515802 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 2 | ok | 510.088929 | 0.033256 | 0.03684385 | 0.04255969999999999 | 29628.50594110801 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 4 | ok | 518.472023 | 0.032552 | 0.036294099999999996 | 0.040115609999999996 | 30236.052864714828 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 8 | ok | 549.232175 | 0.0355585 | 0.04078045 | 0.04190489 | 27790.638667783478 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 1 | 128 | ok | 781.575564 | 0.032512 | 0.03551435 | 0.036559009999999996 | 30251.01688793269 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 1 | ok | 497.555747 | 0.0363225 | 0.03956175 | 0.04252221999999999 | 54736.54479939878 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 2 | ok | 510.41244 | 0.036737 | 0.0396214 | 0.041200259999999995 | 53869.81927214332 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 4 | ok | 512.4665 | 0.034093 | 0.0372513 | 0.03999429999999999 | 57544.617218960724 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 8 | ok | 534.645286 | 0.0335885 | 0.03703315 | 0.037439929999999996 | 58627.02541716061 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 2 | 128 | ok | 763.789861 | 0.033479 | 0.0353395 | 0.038297199999999997 | 59413.43492238526 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 1 | ok | 497.612375 | 0.0338345 | 0.0379755 | 0.0383697 | 115963.27211245656 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 2 | ok | 516.130213 | 0.035944500000000004 | 0.038743049999999994 | 0.042704169999999986 | 111640.03679655615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 4 | ok | 506.770225 | 0.0358025 | 0.03873485 | 0.04134726 | 110825.91907934692 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 8 | ok | 533.473869 | 0.03375 | 0.037944599999999995 | 0.04435881 | 115584.55443599074 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 4 | 128 | ok | 716.24297 | 0.033524 | 0.03830499999999999 | 0.04131043999999999 | 117359.87231245893 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 1 | ok | 499.596936 | 0.036408499999999996 | 0.03982025 | 0.04067321 | 218247.2129830902 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 2 | ok | 508.072925 | 0.035973500000000005 | 0.0409751 | 0.04643008 | 218430.5112475331 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 4 | ok | 513.72308 | 0.034723500000000004 | 0.04021784999999999 | 0.04204976 | 225698.60772171363 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 8 | ok | 533.393699 | 0.037024 | 0.04114225 | 0.044244069999999996 | 211235.1765714841 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 8 | 128 | ok | 708.352293 | 0.034536 | 0.0387065 | 0.04643670999999999 | 224160.0861671371 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 1 | ok | 497.230916 | 0.037359 | 0.0403547 | 0.04156101 | 426300.1622072117 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 2 | ok | 513.791805 | 0.0365645 | 0.041108599999999995 | 0.046785819999999985 | 430136.20262856234 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 4 | ok | 515.786396 | 0.0346405 | 0.03749735 | 0.040239519999999994 | 457718.2744021055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 8 | ok | 548.705963 | 0.035883 | 0.0418444 | 0.04630177999999999 | 428221.51042291155 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 16 | 128 | ok | 830.556856 | 0.0350125 | 0.03716435 | 0.04355035999999999 | 451737.4953414571 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 1 | ok | 504.964384 | 0.039728 | 0.04431499999999999 | 0.045463479999999994 | 793798.449612403 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 2 | ok | 515.003943 | 0.039219500000000004 | 0.04268755 | 0.044170509999999996 | 809659.6443873635 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 4 | ok | 510.849714 | 0.0468815 | 0.05304785 | 0.05495931 | 674165.9198172336 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 8 | ok | 542.413865 | 0.0371175 | 0.04254164999999999 | 0.04549912999999999 | 844279.0595575557 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 32 | 128 | ok | 714.974589 | 0.040292 | 0.043070950000000004 | 0.04366237 | 791975.3101697055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 1 | ok | 497.91659 | 0.0434555 | 0.051486250000000004 | 0.05414586999999999 | 1443692.821869059 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 2 | ok | 505.989916 | 0.070329 | 0.07631489999999999 | 0.07859664999999999 | 898552.8245165645 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 4 | ok | 514.416878 | 0.0527595 | 0.0576625 | 0.06440802999999999 | 1190972.8722432235 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 8 | ok | 537.758312 | 0.050917500000000004 | 0.05634695 | 0.05750893 | 1248554.4080993724 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 64 | 128 | ok | 677.460772 | 5.9837620000000005 | 6.01715705 | 6.624173159999997 | 16596.085761310427 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 1 | ok | 496.208945 | 0.048950499999999994 | 0.0511547 | 0.05288544 | 2602205.6946018874 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 2 | ok | 514.079289 | 0.100188 | 0.10552415 | 0.11325601999999997 | 1273221.3694291592 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 4 | ok | 521.337983 | 0.108325 | 0.11418435 | 0.1169166 | 1190752.1720994115 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 8 | ok | 533.231827 | 0.069414 | 0.0740388 | 0.07956002999999999 | 1831499.2108813948 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `fp32` | 128 | 128 | ok | 785.719475 | 0.14198699999999997 | 0.2164872 | 0.21965263999999998 | 813600.2433681729 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 1 | ok | 498.365376 | 0.054063 | 0.059984 | 0.06126113 | 18309.78694731908 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 2 | ok | 497.23781 | 0.053912 | 0.061213649999999994 | 0.0621053 | 18312.543103148328 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 4 | ok | 508.36189 | 0.050825 | 0.05913635 | 0.06111042 | 19442.354391347373 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 8 | ok | 517.658272 | 0.057726 | 0.06137395 | 0.06471985999999999 | 17218.976827734125 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 1 | 128 | ok | 532.311819 | 0.1461715 | 0.19622599999999998 | 0.20423882 | 6621.322087326762 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 1 | ok | 541.982203 | 0.057963 | 0.06419074999999999 | 0.06625879 | 34259.30525554872 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 2 | ok | 505.541923 | 0.052862 | 0.0582082 | 0.06044976 | 37445.89068795591 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 4 | ok | 511.757819 | 0.0622435 | 0.0670308 | 0.06887905 | 31948.575572758087 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 8 | ok | 507.671518 | 0.056288000000000005 | 0.061764799999999995 | 0.06499772999999999 | 35150.45978558923 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 2 | 128 | ok | 660.74707 | 0.163295 | 0.1909069 | 0.21381638999999997 | 12089.674921939992 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 1 | ok | 494.894488 | 0.055565 | 0.05904084999999999 | 0.06046811999999999 | 72136.40995121775 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 2 | ok | 505.037802 | 0.0537725 | 0.06292184999999999 | 0.06510131 | 73193.71631945396 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 4 | ok | 505.144431 | 0.056166 | 0.06020205 | 0.06416551 | 71091.29401795198 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 8 | ok | 500.972977 | 0.063748 | 0.06960205 | 0.07343445 | 62064.84783561256 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 4 | 128 | ok | 574.877191 | 5.9955805 | 8.168221549999995 | 11.629343189999997 | 736.5984981832055 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 1 | ok | 498.767719 | 0.0709475 | 0.0762328 | 0.08092329999999999 | 112099.80021013108 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 2 | ok | 504.625169 | 0.074629 | 0.0829622 | 0.09904798999999995 | 105158.7463289739 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 4 | ok | 505.4465 | 0.0662205 | 0.0724187 | 0.0739438 | 119283.73694476775 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 8 | ok | 502.690296 | 0.0659265 | 0.0735174 | 0.09061170999999994 | 118860.41389573325 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 8 | 128 | ok | 630.026861 | 0.1509125 | 6.0049643999999995 | 7.036393899999997 | 4300.893243916043 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 1 | ok | 548.294903 | 0.0666775 | 0.071077 | 0.07317864 | 240580.376099302 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 2 | ok | 504.179734 | 0.0673925 | 0.0744632 | 0.08002632 | 236337.47810185552 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 4 | ok | 509.725806 | 0.0781135 | 0.08392115 | 0.08747299 | 204679.27525115426 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 8 | ok | 504.413992 | 0.0715065 | 0.0790661 | 0.08278504999999999 | 222366.76061662304 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 16 | 128 | ok | 528.746419 | 0.149682 | 0.1645257 | 0.1677692 | 106173.10330045727 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 1 | ok | 528.560182 | 0.06543299999999999 | 0.07328185 | 0.07623055 | 483402.81745289615 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 2 | ok | 513.188372 | 0.07776050000000001 | 0.08740909999999999 | 0.09124194 | 403765.6191051797 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 4 | ok | 512.957798 | 0.0719515 | 0.08240455 | 0.08707540999999999 | 431824.1439356237 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 8 | ok | 513.476039 | 0.07344999999999999 | 0.08473925 | 0.08646202 | 428477.4695834556 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 32 | 128 | ok | 637.369845 | 5.9969135 | 6.0991785 | 7.921194989999999 | 6960.912713134719 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 1 | ok | 499.161574 | 0.0671145 | 0.07140675 | 0.07180997 | 957859.2802944699 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 2 | ok | 506.240732 | 0.1002885 | 0.1080743 | 0.11166176999999998 | 638213.5127351518 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 4 | ok | 501.990947 | 0.0846895 | 0.09414995 | 0.0967694 | 745560.8259975197 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 8 | ok | 504.894766 | 0.08475 | 0.09719755 | 0.10290774999999999 | 741374.341219394 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 64 | 128 | ok | 636.646252 | 0.19760149999999999 | 0.25311005 | 0.25782201 | 314378.3801202046 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 1 | ok | 495.974757 | 0.08100550000000001 | 0.0886789 | 0.10071219999999996 | 1553782.7326182697 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 2 | ok | 506.226303 | 0.1069075 | 0.11187065 | 0.11731342999999998 | 1206431.3346410545 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 4 | ok | 502.406425 | 0.116245 | 0.12617275 | 0.13145197 | 1098330.4347706703 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 8 | ok | 507.782365 | 0.0950145 | 0.10341315 | 0.10979114 | 1332270.0153273502 | - |
| `full_mlp_capacity_search_hd32_depth3` | `compile` | `bf16` | 128 | 128 | ok | 610.079271 | 0.3682615 | 0.41511499999999996 | 0.43644528 | 345256.4222145103 | - |
