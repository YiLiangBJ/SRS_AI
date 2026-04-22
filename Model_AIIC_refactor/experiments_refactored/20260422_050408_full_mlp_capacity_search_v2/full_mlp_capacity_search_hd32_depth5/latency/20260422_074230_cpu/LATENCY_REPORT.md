# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth5

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`882782.364` samples/s, p50=`0.143` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.089` ms, throughput=`11148.620` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 1 | 1 | ok | 0.08946399999999999 | 0.0934016 | 0.09719185 | 11095.013479331876 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 1 | 2 | ok | 0.0892595 | 0.09275275 | 0.09953060999999998 | 11148.620023813452 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 1 | 4 | ok | 0.0935655 | 0.09731285 | 0.11120270999999998 | 10571.506196488359 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 1 | 8 | ok | 0.093262 | 0.09637605 | 0.10557641 | 10641.773430258605 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 1 | 64 | ok | 0.0929865 | 0.097373 | 0.09955725 | 10693.620293069358 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 2 | 1 | ok | 0.114563 | 0.12075704999999999 | 0.13207826 | 17301.079968013764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 2 | 2 | ok | 0.11829500000000001 | 0.1262358 | 0.14054053 | 16706.480995291946 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 2 | 4 | ok | 0.11704300000000001 | 0.12397305 | 0.13607772999999998 | 16923.282866411497 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 2 | 8 | ok | 0.11301749999999999 | 0.11538605 | 0.1197684 | 17631.596950156887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 2 | 64 | ok | 0.0995505 | 0.10359294999999999 | 0.10587381 | 20000.50401270112 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 4 | 1 | ok | 0.116985 | 0.12599559999999999 | 0.12860975 | 33928.16762279071 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 4 | 2 | ok | 0.115691 | 0.12105244999999999 | 0.13143667999999997 | 34259.75126735385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 4 | 4 | ok | 0.113205 | 0.11921 | 0.13008749 | 35024.472474529764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 4 | 8 | ok | 0.1164405 | 0.12254559999999999 | 0.13177411 | 34077.883981503204 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 4 | 64 | ok | 0.1199885 | 0.1234996 | 0.12568585999999998 | 33396.212836068175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 8 | 1 | ok | 0.116863 | 0.12289604999999999 | 0.13419813999999997 | 67824.15367325442 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 8 | 2 | ok | 0.11801049999999999 | 0.12468815 | 0.13768054999999996 | 67141.85193341676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 8 | 4 | ok | 0.11528 | 0.12087105 | 0.12966214999999998 | 68805.20112276328 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 8 | 8 | ok | 0.1146705 | 0.12013335 | 0.12981076 | 69212.57031780855 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 8 | 64 | ok | 0.10328499999999999 | 0.1064995 | 0.11281453999999999 | 77180.83121439793 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 16 | 1 | ok | 0.115664 | 0.1195513 | 0.12615171999999997 | 137617.07772887783 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 16 | 2 | ok | 0.114875 | 0.11973805 | 0.12420331999999999 | 138636.6952616404 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 16 | 4 | ok | 0.11833650000000001 | 0.1261349 | 0.14056573999999997 | 133676.05646276113 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 16 | 8 | ok | 0.1183135 | 0.12395414999999999 | 0.13382821 | 134155.43785651808 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 16 | 64 | ok | 0.1064175 | 0.1116269 | 0.1154216 | 149312.98296917454 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 32 | 1 | ok | 0.1196755 | 0.12528309999999998 | 0.13948012999999998 | 264554.0701395672 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 32 | 2 | ok | 0.1241805 | 0.12999715 | 0.13453875999999998 | 256273.41297883887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 32 | 4 | ok | 0.1333275 | 0.1391448 | 0.1472193 | 238333.00363934494 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 32 | 8 | ok | 0.12425649999999999 | 0.1313163 | 0.14257385 | 254491.01082079872 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 32 | 64 | ok | 0.119758 | 0.12468304999999999 | 0.12797608 | 265999.7167103017 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 64 | 1 | ok | 0.126515 | 0.13758659999999998 | 0.14711438999999998 | 501233.975393171 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 64 | 2 | ok | 0.1468335 | 0.15345425 | 0.16236583 | 432763.9647183768 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 64 | 4 | ok | 0.14385799999999999 | 0.15017565 | 0.15646482 | 441858.1627012889 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 64 | 8 | ok | 0.13707750000000002 | 0.1421284 | 0.1493617 | 463897.67113221326 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 64 | 64 | ok | 0.167871 | 0.17251835 | 0.17665456 | 380199.2695659158 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 128 | 1 | ok | 0.14345200000000002 | 0.1527924 | 0.16269758999999998 | 882782.3644911828 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 128 | 2 | ok | 0.1792695 | 0.18828485 | 0.19379842 | 709416.5514419002 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 128 | 4 | ok | 0.195523 | 0.20439235 | 0.21275656 | 662504.2765168631 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 128 | 8 | ok | 0.16492800000000002 | 0.17255405 | 0.17746494 | 774388.644364919 | - |
| `full_mlp_capacity_search_hd32_depth5` | `fp32` | 128 | 64 | ok | 0.190714 | 0.19653125 | 0.20020478 | 668751.9886228568 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 1 | 1 | ok | 0.1643175 | 0.17201734999999999 | 0.17475367 | 6059.512410426772 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 1 | 2 | ok | 0.169816 | 0.17774455 | 0.18302221999999999 | 5853.692573666087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 1 | 4 | ok | 0.172417 | 0.1800533 | 0.18289533 | 5801.596088703155 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 1 | 8 | ok | 0.183788 | 0.19175155 | 0.19697808 | 5453.717679360231 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 1 | 64 | ok | 0.2904115 | 0.30950805 | 0.31492035 | 3449.337068459476 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 2 | 1 | ok | 0.166208 | 0.17408705 | 0.17876698000000002 | 11945.167857082279 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 2 | 2 | ok | 0.1685855 | 0.17467635 | 0.17980843 | 11802.458475705407 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 2 | 4 | ok | 0.1797335 | 0.1912617 | 0.19818779 | 10991.536626712797 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 2 | 8 | ok | 0.192147 | 0.20636824999999998 | 0.21220276 | 10370.788889317548 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 2 | 64 | ok | 0.285441 | 0.3123606 | 0.31880932 | 6893.026772860636 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 4 | 1 | ok | 0.1744995 | 0.1838575 | 0.19383323 | 22764.902075635477 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 4 | 2 | ok | 0.182962 | 0.19979595 | 0.20456792999999998 | 21494.878522768773 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 4 | 4 | ok | 0.1859775 | 0.19796719999999998 | 0.20427097 | 21414.97292076674 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 4 | 8 | ok | 0.192508 | 0.2088834 | 0.21385716000000002 | 20689.553864943144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 4 | 64 | ok | 0.303747 | 0.3230973 | 0.32609422 | 13318.273030190792 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 8 | 1 | ok | 0.25546650000000004 | 0.26492215 | 0.26535892 | 31225.59524571577 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 8 | 2 | ok | 0.2748035 | 0.28398734999999997 | 0.285748 | 29023.799733459935 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 8 | 4 | ok | 0.3021045 | 0.31564964999999995 | 0.32101265999999995 | 26410.580871014354 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 8 | 8 | ok | 0.3149725 | 0.3388364 | 0.34188128 | 25270.932829734185 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 8 | 64 | ok | 0.549627 | 0.6565365 | 0.66565048 | 14192.242654991087 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 16 | 1 | ok | 0.2589595 | 0.27014125 | 0.27582157999999996 | 61345.007697264846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 16 | 2 | ok | 0.2885665 | 0.30094139999999997 | 0.30833568 | 55203.64382691809 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 16 | 4 | ok | 0.301895 | 0.31500564999999997 | 0.31833244 | 52818.39622485794 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 16 | 8 | ok | 0.33033199999999996 | 0.350852 | 0.35977927 | 48134.913498251175 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 16 | 64 | ok | 0.535961 | 0.6347949 | 0.6438476399999999 | 29626.525428131994 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 32 | 1 | ok | 0.27382300000000004 | 0.28453015000000004 | 0.28771116999999996 | 116319.7309524623 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 32 | 2 | ok | 0.3175885 | 0.3346712 | 0.33984558 | 100219.5246049268 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 32 | 4 | ok | 0.30974100000000004 | 0.3251814 | 0.3295724 | 102983.47636558182 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 32 | 8 | ok | 0.3228525 | 0.35295825 | 0.35744069 | 98055.49217697962 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 32 | 64 | ok | 0.5358355 | 0.71115345 | 0.72328112 | 56369.52166197925 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 64 | 1 | ok | 0.296103 | 0.3049151 | 0.30840364 | 215550.26783804764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 64 | 2 | ok | 0.3580515 | 0.3686737 | 0.37356159 | 183175.5543064 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 64 | 4 | ok | 0.3362925 | 0.35937345 | 0.36274664 | 189647.50987782018 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 64 | 8 | ok | 0.359016 | 0.3785458 | 0.38840854999999996 | 178272.54351019976 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 64 | 64 | ok | 0.5367154999999999 | 0.66718305 | 0.67275366 | 114925.79349257115 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 128 | 1 | ok | 0.342097 | 0.35142965 | 0.35778706 | 373234.4915091194 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 128 | 2 | ok | 0.41113500000000003 | 0.4543827 | 0.46017513 | 310030.96967175324 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 128 | 4 | ok | 0.387982 | 0.4078668 | 0.41749497 | 332163.7018980249 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 128 | 8 | ok | 0.401974 | 0.4484984 | 0.45357689 | 313525.4793829074 | - |
| `full_mlp_capacity_search_hd32_depth5` | `bf16` | 128 | 64 | ok | 0.5783685000000001 | 0.6976951499999999 | 0.70378988 | 223668.10449144777 | - |
