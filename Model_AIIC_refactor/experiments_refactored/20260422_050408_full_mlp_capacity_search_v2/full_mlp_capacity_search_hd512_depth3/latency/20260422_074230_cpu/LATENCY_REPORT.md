# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth3

- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`530287.670` samples/s, p50=`0.239` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.081` ms, throughput=`12291.191` samples/s

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

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 1 | 1 | ok | 0.0855125 | 0.09033825 | 0.09585559999999999 | 11576.400072977627 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 1 | 2 | ok | 0.0929 | 0.09646484999999999 | 0.10411673999999999 | 10706.553288549963 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 1 | 4 | ok | 0.085835 | 0.1036606 | 0.10486516 | 11219.337830193526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 1 | 8 | ok | 0.0806665 | 0.0848091 | 0.09364164999999998 | 12291.19109999769 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 1 | 64 | ok | 0.08781449999999999 | 0.10136015 | 0.10281266 | 10901.444833892505 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 2 | 1 | ok | 0.098329 | 0.10203699999999999 | 0.11049535999999997 | 20179.321522780036 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 2 | 2 | ok | 0.097518 | 0.10147519999999999 | 0.11062702999999997 | 20375.757494254543 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 2 | 4 | ok | 0.09941849999999999 | 0.10496269999999999 | 0.11394933999999998 | 19948.489011674057 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 2 | 8 | ok | 0.0957745 | 0.10156469999999998 | 0.10837341999999998 | 20695.78826221812 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 2 | 64 | ok | 0.0951105 | 0.11720205 | 0.12299963999999998 | 19679.96829163509 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 4 | 1 | ok | 0.1016405 | 0.11164819999999999 | 0.11646116999999999 | 38914.397276303505 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 4 | 2 | ok | 0.10737849999999999 | 0.11272949999999998 | 0.12421996999999996 | 36923.32686559724 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 4 | 4 | ok | 0.105086 | 0.11406264999999999 | 0.13543371999999998 | 37390.18036649107 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 4 | 8 | ok | 0.101131 | 0.1041075 | 0.10982408999999999 | 39382.85491041188 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 4 | 64 | ok | 0.1165735 | 0.12050345 | 0.12395170999999999 | 34161.22492587441 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 8 | 1 | ok | 0.117178 | 0.12376455 | 0.12988075 | 67784.5583048031 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 8 | 2 | ok | 0.12742599999999998 | 0.13593834999999999 | 0.14186242000000002 | 62221.88760673776 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 8 | 4 | ok | 0.11590700000000001 | 0.1199887 | 0.12830903999999999 | 68626.58473796226 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 8 | 8 | ok | 0.111843 | 0.11600514999999999 | 0.12310697999999998 | 71070.12418082797 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 8 | 64 | ok | 0.1032005 | 0.10677895 | 0.11114511999999999 | 77233.74968116943 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 16 | 1 | ok | 0.19539800000000002 | 0.20315575 | 0.21116187 | 81446.804750222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 16 | 2 | ok | 0.2945225 | 0.30200360000000004 | 0.31107282 | 58543.172186665775 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 16 | 4 | ok | 0.2255295 | 0.23048745 | 0.23819934999999998 | 75967.47908187985 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 16 | 8 | ok | 0.161996 | 0.16747465 | 0.17526295999999997 | 98175.63891785411 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 16 | 64 | ok | 0.1483005 | 0.1526101 | 0.1562722 | 107653.09919143103 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 32 | 1 | ok | 0.2251475 | 0.2362425 | 0.24364254999999999 | 141237.6763506868 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 32 | 2 | ok | 0.27619099999999996 | 0.3408172 | 0.35573142999999996 | 107017.19077332562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 32 | 4 | ok | 0.2256755 | 0.2617247 | 0.27008968999999994 | 135156.13912972962 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 32 | 8 | ok | 0.196492 | 0.20643584999999998 | 0.21349534 | 161577.05670425625 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 32 | 64 | ok | 0.1811585 | 0.1858018 | 0.19023756 | 178662.15544724162 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 64 | 1 | ok | 0.26549449999999997 | 0.28036485 | 0.28542892 | 239391.08483660955 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 64 | 2 | ok | 0.2583485 | 0.31642844999999997 | 0.32567031 | 230803.61274009975 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 64 | 4 | ok | 0.2938645 | 0.3190034 | 0.32907314 | 222796.83692545656 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 64 | 8 | ok | 0.2411995 | 0.24997645 | 0.25838809999999995 | 276343.2397203337 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 64 | 64 | ok | 0.18942900000000001 | 0.1968122 | 0.20213402 | 338671.64513985027 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 128 | 1 | ok | 0.360623 | 0.3720259 | 0.37497937 | 353228.29133430554 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 128 | 2 | ok | 0.42153949999999996 | 0.5498067999999999 | 0.56048933 | 297713.5229297094 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 128 | 4 | ok | 0.37957850000000004 | 0.3919913 | 0.4436333999999998 | 363589.61344913434 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 128 | 8 | ok | 0.312632 | 0.3229847 | 0.33162674999999997 | 430401.4899423578 | - |
| `full_mlp_capacity_search_hd512_depth3` | `fp32` | 128 | 64 | ok | 0.23866900000000002 | 0.25250995 | 0.25710204999999997 | 530287.6702896638 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 1 | 1 | ok | 0.2005035 | 0.20800735 | 0.21145279999999997 | 4967.621046022028 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 1 | 2 | ok | 0.213961 | 0.22460819999999998 | 0.23377447 | 4636.653730711868 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 1 | 4 | ok | 0.204182 | 0.215922 | 0.21699171 | 4854.507026267835 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 1 | 8 | ok | 0.23002699999999998 | 0.2464372 | 0.25431771 | 4323.083015298526 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 1 | 64 | ok | 0.3776215 | 0.43278835 | 0.43715459 | 2528.6470423427004 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 2 | 1 | ok | 0.225773 | 0.23257304999999998 | 0.23724409 | 8828.504273923061 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 2 | 2 | ok | 0.236262 | 0.2416327 | 0.24668284999999998 | 8590.950893695142 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 2 | 4 | ok | 0.22156900000000002 | 0.23587639999999999 | 0.24204752 | 9005.080846715333 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 2 | 8 | ok | 0.2327505 | 0.24754674999999998 | 0.25812056 | 8556.456483873091 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 2 | 64 | ok | 0.47570650000000003 | 0.7255988 | 0.7374687099999999 | 4104.235431421222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 4 | 1 | ok | 0.2403395 | 0.2628562 | 0.27667919999999996 | 16473.58933565721 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 4 | 2 | ok | 0.2689635 | 0.29218765 | 0.29362472 | 14634.73708511892 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 4 | 4 | ok | 0.2290455 | 0.2403385 | 0.25169570999999996 | 17585.09206235322 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 4 | 8 | ok | 0.246747 | 0.26072765 | 0.26609778999999995 | 16193.629167187562 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 4 | 64 | ok | 0.491328 | 0.7273309 | 2.0672373899999945 | 7667.208201581945 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 8 | 1 | ok | 0.25429599999999997 | 0.26718695 | 0.27251362 | 31241.723384062865 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 8 | 2 | ok | 0.32733199999999996 | 0.335582 | 0.33922123 | 25628.221387340196 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 8 | 4 | ok | 0.26575899999999997 | 0.2850631 | 0.29809210999999997 | 29549.419372377717 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 8 | 8 | ok | 0.2529 | 0.2690779 | 0.27399341 | 31774.3815553528 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 8 | 64 | ok | 0.3915305 | 0.4343507 | 0.44720142999999996 | 20846.211297510406 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 16 | 1 | ok | 0.3041155 | 0.3142806 | 0.31939498 | 52364.88664671171 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 16 | 2 | ok | 0.376742 | 0.39113729999999997 | 0.4138012199999999 | 44492.42952969222 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 16 | 4 | ok | 0.3012285 | 0.3496972 | 0.3542529 | 50537.21374063885 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 16 | 8 | ok | 0.2976665 | 0.31709339999999997 | 0.32964326 | 54127.41498765117 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 16 | 64 | ok | 0.3486995 | 0.47514025000000004 | 0.47858781 | 41019.05483791295 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 32 | 1 | ok | 0.385919 | 0.39306055 | 0.39813023 | 82779.46541538606 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 32 | 2 | ok | 0.399736 | 0.5489390999999997 | 0.59408922 | 72761.72871316972 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 32 | 4 | ok | 0.36201 | 0.4338492 | 0.43649849 | 88168.01252382534 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 32 | 8 | ok | 0.3530675 | 0.38191365 | 0.38733487 | 92313.2328769194 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 32 | 64 | ok | 0.398955 | 0.4664241 | 0.48219528 | 79196.47650916188 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 64 | 1 | ok | 0.5665309999999999 | 0.57660145 | 0.58400331 | 112721.24406210025 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 64 | 2 | ok | 0.5249215 | 0.6710401 | 0.68113255 | 119307.58352556145 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 64 | 4 | ok | 0.4211905 | 0.51421225 | 0.51607808 | 144323.6515615729 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 64 | 8 | ok | 0.396593 | 0.48661455000000003 | 0.5168501799999999 | 155454.6385065123 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 64 | 64 | ok | 0.4572705 | 0.5446322 | 0.55235552 | 142309.35982445785 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 128 | 1 | ok | 0.912636 | 0.92472235 | 0.9620546799999998 | 139753.85546494677 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 128 | 2 | ok | 0.5901345 | 0.89999905 | 0.90450098 | 193468.4448432044 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 128 | 4 | ok | 0.5131025 | 0.65212815 | 0.7584047499999996 | 237126.8114913728 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 128 | 8 | ok | 0.4495795 | 0.56879155 | 0.57372817 | 274139.3865137471 | - |
| `full_mlp_capacity_search_hd512_depth3` | `bf16` | 128 | 64 | ok | 0.490203 | 0.5893314 | 0.60630283 | 254361.3330711822 | - |
