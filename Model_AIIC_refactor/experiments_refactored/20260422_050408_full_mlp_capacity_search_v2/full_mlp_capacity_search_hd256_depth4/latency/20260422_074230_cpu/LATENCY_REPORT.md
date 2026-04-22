# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth4

- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`357533.670` samples/s, p50=`0.344` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.095` ms, throughput=`10471.307` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `109,200`
- MACs / sample: `108,544`
- FLOPs / sample estimate: `217,816`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 1 | 1 | ok | 0.0986055 | 0.1024413 | 0.11248037999999998 | 10048.489993311725 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 1 | 2 | ok | 0.10962949999999999 | 0.1131459 | 0.11616209999999999 | 9082.561391303085 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 1 | 4 | ok | 0.0964795 | 0.09900725 | 0.10361754 | 10332.339706664876 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 1 | 8 | ok | 0.09475900000000001 | 0.09948485 | 0.10278559999999999 | 10471.3072568463 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 1 | 64 | ok | 0.098692 | 0.1190001 | 0.12113789 | 9649.933232111967 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 2 | 1 | ok | 0.1146605 | 0.12306829999999999 | 0.12961259999999997 | 17276.707197821754 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 2 | 2 | ok | 0.1174375 | 0.12350594999999999 | 0.13378544999999997 | 16838.71908537466 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 2 | 4 | ok | 0.11371 | 0.11850195 | 0.13005671 | 17476.327440665245 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 2 | 8 | ok | 0.116149 | 0.1198121 | 0.13354448999999996 | 17096.66780815419 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 2 | 64 | ok | 0.11410000000000001 | 0.14002985 | 0.14750724999999998 | 16684.633235889512 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 4 | 1 | ok | 0.119686 | 0.12650175 | 0.13040186 | 33116.28369064424 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 4 | 2 | ok | 0.138371 | 0.14362919999999998 | 0.15225792999999999 | 29538.52843186746 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 4 | 4 | ok | 0.1245675 | 0.1305039 | 0.13305349 | 31888.20759733357 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 4 | 8 | ok | 0.11773800000000001 | 0.1217587 | 0.12590863 | 33799.37031773098 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 4 | 64 | ok | 0.15771649999999998 | 0.16437155 | 0.17237797 | 25303.11230811701 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 8 | 1 | ok | 0.13568249999999998 | 0.1413002 | 0.15200978999999995 | 58564.77644946357 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 8 | 2 | ok | 0.152756 | 0.15921335 | 0.16842645999999997 | 51992.37375861709 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 8 | 4 | ok | 0.1361275 | 0.1403546 | 0.14545386999999999 | 58492.282455137516 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 8 | 8 | ok | 0.1241315 | 0.13136905 | 0.13764413999999997 | 63919.40019313247 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 8 | 64 | ok | 0.1747645 | 0.18071565 | 0.18459825 | 45626.26612888507 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 16 | 1 | ok | 0.187495 | 0.19503774999999998 | 0.19902723 | 84769.33469317474 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 16 | 2 | ok | 0.2748235 | 0.3501663 | 0.35192923 | 52387.607789958696 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 16 | 4 | ok | 0.2415215 | 0.2596339 | 0.26280752999999996 | 65080.27529932655 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 16 | 8 | ok | 0.1881535 | 0.19717964999999998 | 0.20377837000000001 | 84055.95184434518 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 16 | 64 | ok | 0.249274 | 0.26243679999999997 | 0.26340044999999995 | 64123.9593082179 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 32 | 1 | ok | 0.21884900000000002 | 0.22865829999999998 | 0.23422785 | 145296.14623766526 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 32 | 2 | ok | 0.32945 | 0.4139548 | 0.42565866999999996 | 91688.82097803548 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 32 | 4 | ok | 0.2719645 | 0.3001455 | 0.30861509 | 118287.89217703765 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 32 | 8 | ok | 0.2166145 | 0.22246955 | 0.22705356000000002 | 152696.21917480862 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 32 | 64 | ok | 0.2909945 | 0.3019384 | 0.30551638999999997 | 113247.03673203026 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 64 | 1 | ok | 0.270237 | 0.2812358 | 0.3081741399999999 | 234961.4571428467 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 64 | 2 | ok | 0.32929200000000003 | 0.38016885 | 0.38248296 | 200965.61465772855 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 64 | 4 | ok | 0.32226350000000004 | 0.36519595 | 0.37244517 | 199524.30910655134 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 64 | 8 | ok | 0.245107 | 0.2750588 | 0.28167926 | 251816.83881888774 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 64 | 64 | ok | 0.315188 | 0.35914395 | 0.35991473 | 194228.66516471255 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 128 | 1 | ok | 0.3833405 | 0.39270435 | 0.39807885 | 332896.32022129284 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 128 | 2 | ok | 0.4658085 | 0.6216742999999999 | 0.62562326 | 286328.21687179175 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 128 | 4 | ok | 0.3480975 | 0.41377225 | 0.41837749999999996 | 343097.715623244 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 128 | 8 | ok | 0.344098 | 0.4162528 | 0.4370221899999999 | 357533.6701747424 | - |
| `full_mlp_capacity_search_hd256_depth4` | `fp32` | 128 | 64 | ok | 0.373757 | 0.40735285 | 0.41135936 | 339723.92971833807 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 1 | 1 | ok | 0.25569200000000003 | 0.2637681 | 0.26881436 | 3891.847125131447 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 1 | 2 | ok | 0.262375 | 0.2805227 | 0.28534474 | 3750.024281407222 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 1 | 4 | ok | 0.27307950000000003 | 0.2965847 | 0.30169307 | 3627.061739772393 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 1 | 8 | ok | 0.3091665 | 0.33488275 | 0.34165528999999994 | 3234.9101147875504 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 1 | 64 | ok | 0.5151159999999999 | 0.7039382499999999 | 0.71978767 | 1850.3087628735695 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 2 | 1 | ok | 0.2937245 | 0.32736405 | 0.33492956 | 6616.904019425113 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 2 | 2 | ok | 0.2826225 | 0.30300855 | 0.30701379 | 6966.391480270413 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 2 | 4 | ok | 0.286512 | 0.3021647 | 0.30874074999999995 | 7065.220962312768 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 2 | 8 | ok | 0.311527 | 0.3392403 | 0.35127400999999997 | 6380.711262989055 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 2 | 64 | ok | 0.505223 | 0.71689105 | 0.73324481 | 3770.3539257622588 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 4 | 1 | ok | 0.291188 | 0.3004389 | 0.30514647 | 13687.965711645893 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 4 | 2 | ok | 0.343237 | 0.3588689 | 0.36703606 | 11970.049977352664 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 4 | 4 | ok | 0.29134899999999997 | 0.31834694999999996 | 0.32190605 | 13491.96556823403 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 4 | 8 | ok | 0.317546 | 0.3428271 | 0.34510333 | 12570.87538107823 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 4 | 64 | ok | 0.552511 | 0.67693635 | 0.691382 | 7171.807424204844 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 8 | 1 | ok | 0.327352 | 0.33720265 | 0.33975514 | 24361.473596304073 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 8 | 2 | ok | 0.36745300000000003 | 0.42016265 | 0.42560296 | 21870.485282557645 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 8 | 4 | ok | 0.3374855 | 0.37773715 | 0.38350382999999993 | 23187.966233915453 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 8 | 8 | ok | 0.319853 | 0.3422117 | 0.35831397 | 25086.45419276181 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 8 | 64 | ok | 0.5568854999999999 | 0.65102785 | 0.6680454499999999 | 14453.859648469846 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 16 | 1 | ok | 0.377254 | 0.3855628 | 0.38826881 | 42350.62260444365 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 16 | 2 | ok | 0.4170375 | 0.5002161 | 0.50443683 | 37120.59618647123 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 16 | 4 | ok | 0.374613 | 0.43910635 | 0.45120217 | 41565.232345752076 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 16 | 8 | ok | 0.37371449999999995 | 0.42288715 | 0.43346726999999996 | 42689.01977579513 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 16 | 64 | ok | 0.5226824999999999 | 0.6149759499999999 | 0.63203733 | 30565.0456838545 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 32 | 1 | ok | 0.48273299999999997 | 0.494628 | 0.49747395 | 66125.66778142143 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 32 | 2 | ok | 0.4945585 | 0.6126182499999999 | 0.61830989 | 64998.029137885234 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 32 | 4 | ok | 0.43187549999999997 | 0.54978895 | 0.55869727 | 73696.91645653066 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 32 | 8 | ok | 0.4114815 | 0.46423559999999997 | 0.47794026999999994 | 76088.1919210509 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 32 | 64 | ok | 0.572893 | 0.6983882499999999 | 0.71447531 | 55293.745430706316 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 64 | 1 | ok | 0.6983085 | 0.7092035999999999 | 0.7110629199999999 | 91546.7923405774 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 64 | 2 | ok | 0.5752390000000001 | 0.7331416 | 0.74000092 | 107818.8694207694 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 64 | 4 | ok | 0.5091515 | 0.6482493500000001 | 0.64880191 | 122854.07995136206 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 64 | 8 | ok | 0.48362150000000004 | 0.6204779 | 0.62132765 | 133349.77424925016 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 64 | 64 | ok | 0.589691 | 0.73016865 | 0.7534449 | 105669.07312570637 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 128 | 1 | ok | 1.125456 | 1.13760515 | 1.14820359 | 113501.56641915701 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 128 | 2 | ok | 0.734958 | 1.12461565 | 1.1335024500000002 | 162313.6183154687 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 128 | 4 | ok | 0.550989 | 0.87493985 | 0.88182962 | 204143.70900539146 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 128 | 8 | ok | 0.536819 | 0.702527 | 0.71895332 | 229895.1401568294 | - |
| `full_mlp_capacity_search_hd256_depth4` | `bf16` | 128 | 64 | ok | 0.649033 | 0.8168983 | 0.8513954699999999 | 190479.81866321262 | - |
