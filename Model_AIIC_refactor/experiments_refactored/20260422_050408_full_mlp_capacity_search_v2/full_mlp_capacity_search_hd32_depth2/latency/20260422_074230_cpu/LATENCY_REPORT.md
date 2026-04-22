# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd32_depth2

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1526334.881` samples/s, p50=`0.083` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.055` ms, throughput=`18189.987` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 1 | 1 | ok | 0.056245 | 0.0591021 | 0.06550235999999998 | 17594.37092734299 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 1 | 2 | ok | 0.055764 | 0.059055 | 0.061405549999999996 | 17760.998764900145 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 1 | 4 | ok | 0.059566499999999994 | 0.0622218 | 0.06475746 | 16680.121965051807 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 1 | 8 | ok | 0.0579095 | 0.062211550000000004 | 0.06608696 | 17079.833533110454 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 1 | 64 | ok | 0.054524 | 0.05692054999999999 | 0.06222882999999999 | 18189.987139679095 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 2 | 1 | ok | 0.0695415 | 0.0731256 | 0.08048279999999998 | 28524.273871822177 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 2 | 2 | ok | 0.0674955 | 0.0704248 | 0.07817371999999997 | 29320.63788806964 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 2 | 4 | ok | 0.0709275 | 0.07465225 | 0.08010909 | 27905.42182812325 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 2 | 8 | ok | 0.068785 | 0.0720587 | 0.07629390999999999 | 28850.395409094275 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 2 | 64 | ok | 0.0679965 | 0.07144905 | 0.07888611999999998 | 29183.24274854784 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 4 | 1 | ok | 0.07304749999999999 | 0.0756449 | 0.07696215000000001 | 54530.666820079605 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 4 | 2 | ok | 0.068403 | 0.07158405 | 0.07597543999999999 | 58027.83130845216 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 4 | 4 | ok | 0.0704015 | 0.0746636 | 0.07792852 | 56329.15831000132 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 4 | 8 | ok | 0.0675865 | 0.07012165 | 0.07339621999999998 | 58710.02911136793 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 4 | 64 | ok | 0.067427 | 0.07052119999999999 | 0.07476073 | 58959.031138033126 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 8 | 1 | ok | 0.0708385 | 0.07432129999999999 | 0.08048415999999999 | 112023.44874829201 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 8 | 2 | ok | 0.0694275 | 0.07251045 | 0.07371989999999999 | 114634.20226058649 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 8 | 4 | ok | 0.07576 | 0.0783292 | 0.08262913 | 105041.32719716913 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 8 | 8 | ok | 0.07546449999999999 | 0.07897230000000001 | 0.08483969999999998 | 105213.40303019862 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 8 | 64 | ok | 0.068858 | 0.0734084 | 0.08277477999999998 | 114646.52327252767 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 16 | 1 | ok | 0.069132 | 0.07233415 | 0.07610621 | 229859.56155437932 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 16 | 2 | ok | 0.07295750000000001 | 0.07647805 | 0.08423933999999997 | 217157.8596487037 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 16 | 4 | ok | 0.0747475 | 0.0783233 | 0.08192914 | 212479.22351592555 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 16 | 8 | ok | 0.0700045 | 0.07291605 | 0.07413887 | 227432.12359343894 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 16 | 64 | ok | 0.0739925 | 0.07728884999999999 | 0.08061979999999999 | 215010.99243698834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 32 | 1 | ok | 0.071163 | 0.07550464999999999 | 0.07927143999999998 | 444987.2052272646 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 32 | 2 | ok | 0.0713055 | 0.07494935 | 0.08058478999999998 | 445019.632319217 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 32 | 4 | ok | 0.074201 | 0.07757395 | 0.08014891 | 429332.0746748011 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 32 | 8 | ok | 0.0706955 | 0.07459444999999999 | 0.07986639 | 448168.47549330746 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 32 | 64 | ok | 0.06965 | 0.07304245 | 0.07735062999999999 | 455043.28741472564 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 64 | 1 | ok | 0.0740815 | 0.07835944999999998 | 0.08489717999999997 | 855422.6175343994 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 64 | 2 | ok | 0.0934585 | 0.09713719999999999 | 0.10362852999999997 | 680636.3013542749 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 64 | 4 | ok | 0.088147 | 0.0923601 | 0.09561109 | 722141.7460439608 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 64 | 8 | ok | 0.074614 | 0.078206 | 0.08073952 | 852794.8218298418 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 64 | 64 | ok | 0.074872 | 0.07909655 | 0.08227416 | 846757.751339531 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 128 | 1 | ok | 0.083258 | 0.08672745 | 0.09281491999999998 | 1526334.8812117956 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 128 | 2 | ok | 0.120832 | 0.12547740000000002 | 0.13312185999999998 | 1053267.870540871 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 128 | 4 | ok | 0.11501149999999999 | 0.12013635 | 0.122617 | 1107133.8863928402 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 128 | 8 | ok | 0.1050105 | 0.11092555 | 0.12029171999999998 | 1207422.32689903 | - |
| `full_mlp_capacity_search_hd32_depth2` | `fp32` | 128 | 64 | ok | 0.15369549999999998 | 0.1632381 | 0.16538439 | 824974.5579135363 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 1 | 1 | ok | 0.0719805 | 0.0746805 | 0.07804226999999998 | 13805.172024868085 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 1 | 2 | ok | 0.0734515 | 0.07663635 | 0.07968236 | 13515.135329753082 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 1 | 4 | ok | 0.069051 | 0.07292035 | 0.07785352 | 14348.474513235176 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 1 | 8 | ok | 0.070693 | 0.07389855000000001 | 0.07963886999999997 | 14027.665361626201 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 1 | 64 | ok | 0.072126 | 0.07646185 | 0.07844977 | 13760.450374036565 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 2 | 1 | ok | 0.11457200000000001 | 0.12055795 | 0.12860316 | 17310.753647895115 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 2 | 2 | ok | 0.1203645 | 0.12640415 | 0.13281637 | 16450.377108444834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 2 | 4 | ok | 0.120309 | 0.13145020000000002 | 0.14020086 | 16428.313738735927 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 2 | 8 | ok | 0.141014 | 0.15133005 | 0.15419114 | 14192.830321449155 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 2 | 64 | ok | 0.23774699999999999 | 0.2472557 | 0.25099108 | 8433.111803919355 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 4 | 1 | ok | 0.119482 | 0.1247051 | 0.1284725 | 33359.89336843684 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 4 | 2 | ok | 0.12046899999999999 | 0.1371609 | 0.13901166 | 32428.78112696825 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 4 | 4 | ok | 0.1314275 | 0.1398014 | 0.14110969 | 30379.088493677198 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 4 | 8 | ok | 0.1383255 | 0.15384745 | 0.16306365999999997 | 28514.903884813477 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 4 | 64 | ok | 0.2544015 | 0.27126205000000003 | 0.28181107 | 15622.507721815005 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 8 | 1 | ok | 0.1202915 | 0.1296089 | 0.13207617000000002 | 65990.41885108702 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 8 | 2 | ok | 0.129913 | 0.14681629999999998 | 0.15478896 | 60155.2878678665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 8 | 4 | ok | 0.13074950000000002 | 0.1562769 | 0.16451889 | 59457.371220164845 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 8 | 8 | ok | 0.132216 | 0.14529065 | 0.14874584999999999 | 60180.90077869573 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 8 | 64 | ok | 0.2454045 | 0.26468845 | 0.26644322 | 32347.832359623142 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 16 | 1 | ok | 0.125064 | 0.13069435000000001 | 0.13579992 | 127004.77029917244 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 16 | 2 | ok | 0.1357425 | 0.15317899999999998 | 0.15885527 | 115327.35091921665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 16 | 4 | ok | 0.132375 | 0.15956545 | 0.16145921000000002 | 117487.48317726099 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 16 | 8 | ok | 0.1469755 | 0.21005675 | 0.23006151999999994 | 103670.55984175726 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 16 | 64 | ok | 0.2593985 | 0.2694166 | 0.28026337999999995 | 61646.49939967868 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 32 | 1 | ok | 0.1242465 | 0.1307374 | 0.13819457 | 255245.9829466968 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 32 | 2 | ok | 0.146414 | 0.15106125 | 0.15688419 | 217424.71567301988 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 32 | 4 | ok | 0.1497305 | 0.1790899 | 0.18297443 | 207000.09638441988 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 32 | 8 | ok | 0.1501095 | 0.20571435 | 0.22672470999999994 | 201267.35537839835 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 32 | 64 | ok | 0.2612455 | 0.28091689999999997 | 0.28822662 | 121468.95443319928 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 64 | 1 | ok | 0.134857 | 0.1492405 | 0.15324174 | 467857.8717964163 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 64 | 2 | ok | 0.162083 | 0.18932435 | 0.19210884 | 387299.24736863445 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 64 | 4 | ok | 0.161196 | 0.1710286 | 0.17372337999999998 | 395490.0294491763 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 64 | 8 | ok | 0.17126249999999998 | 0.23312829999999998 | 0.25829467999999994 | 352660.64270859247 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 64 | 64 | ok | 0.253552 | 0.27055999999999997 | 0.27595452 | 256300.87672520528 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 128 | 1 | ok | 0.161445 | 0.18396154999999997 | 0.19290709 | 778932.789054047 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 128 | 2 | ok | 0.2116445 | 0.24139505 | 0.25367142 | 599704.6454621098 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 128 | 4 | ok | 0.202601 | 0.23504564999999997 | 0.25344940999999993 | 627747.1909048848 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 128 | 8 | ok | 0.21432849999999998 | 0.22943309999999997 | 0.2331089 | 595895.1760795758 | - |
| `full_mlp_capacity_search_hd32_depth2` | `bf16` | 128 | 64 | ok | 0.3704345 | 0.9527612499999998 | 1.02392855 | 297778.93208472145 | - |
