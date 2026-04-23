# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd512_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`256275.978` samples/s, p50=`0.512` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.138` ms, throughput=`7042.828` samples/s

### full_mlp_capacity_search_hd512_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`287534.107` samples/s, p50=`0.442` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.087` ms, throughput=`11402.064` samples/s

### full_mlp_capacity_search_hd512_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`258354.457` samples/s, p50=`0.480` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.120` ms, throughput=`8207.418` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `611,984`
- MACs / sample: `610,304`
- FLOPs / sample estimate: `1,222,360`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2039285 | 0.2077724 | 0.22333193999999998 | 4882.579814555713 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2382865 | 0.2502986 | 0.25558667 | 4300.891764101936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.165908 | 0.17717829999999998 | 0.18420179999999997 | 5967.930964884336 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.1382795 | 0.15451235 | 0.1575783 | 7042.8278587789055 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.178967 | 0.18407844999999998 | 0.18576152999999998 | 5662.379490408496 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.2310005 | 0.24138969999999998 | 0.25014747 | 8607.29788687394 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.2230385 | 0.22962155 | 0.24483614999999997 | 9182.820778886859 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1766635 | 0.18407305 | 0.19175424 | 11247.67159138469 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.15859250000000003 | 0.16488704999999998 | 0.17345583999999997 | 12573.523679717146 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.180122 | 0.18464155 | 0.18983899999999998 | 11078.276330985662 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.28886599999999996 | 0.298527 | 0.30805516 | 13783.970854620668 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.2754765 | 0.31947575 | 0.3587902099999999 | 13716.691073232314 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.222829 | 0.23414495 | 0.24406185999999996 | 17820.366076652168 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.18153750000000002 | 0.18560015 | 0.19215184999999999 | 22289.280761687136 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.172464 | 0.17828395 | 0.18303595 | 23188.556354684737 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3605095 | 0.3768856 | 0.38177592 | 22070.917499627692 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.33297200000000005 | 0.40453895 | 0.41765406 | 23149.266761443625 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.27516799999999997 | 0.2863962 | 0.29708115 | 29490.578276325836 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.208127 | 0.2158775 | 0.22246115 | 39251.92888885051 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.192868 | 0.2006283 | 0.20372525 | 41237.113402061856 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.4909165 | 0.50437705 | 0.50556126 | 32475.254566416585 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.5694115 | 0.9048530999999995 | 0.9861658799999999 | 25324.246107481355 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.5052455 | 0.70043565 | 0.7067827799999999 | 32014.17587707837 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.41583899999999996 | 0.54534415 | 0.5538586 | 38612.45375193352 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.25636950000000003 | 0.2807689 | 0.28478651 | 60714.87822644555 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.617307 | 0.631795 | 0.63355799 | 51668.989080082094 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.662298 | 0.9135824499999999 | 0.9200279100000001 | 46041.05221645222 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.47093850000000004 | 0.6872946999999996 | 0.79484768 | 61336.76972585723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.4087375 | 0.52699175 | 0.53453588 | 72862.01114542861 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.31254 | 0.3402513999999999 | 0.35308133999999997 | 103618.25222560673 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.8734785 | 0.8861769 | 0.89176859 | 73420.27502271383 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.615246 | 0.7972581 | 0.80571571 | 100756.68268697921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.575338 | 0.7642491 | 0.7682236800000001 | 107331.74784217068 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.485753 | 0.6891082 | 0.69831849 | 127887.07571214618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.40497249999999996 | 0.48416625 | 0.48566548 | 156298.49221775154 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.332578 | 1.3851201499999999 | 1.38797607 | 97738.84093127339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.839368 | 0.96128255 | 0.96460856 | 151138.43610070014 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.6357685 | 0.8403888999999999 | 0.86415996 | 189328.6727713922 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.5661975 | 0.70398445 | 0.7107309599999999 | 218724.7900455614 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.5116674999999999 | 0.70318015 | 0.70728077 | 256275.9784757009 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.502177 | 0.5124235 | 0.51561662 | 1989.2728066049267 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.4603165 | 0.5260040500000001 | 0.52873989 | 2116.5969777111754 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.4027095 | 0.4539938 | 0.46490003999999996 | 2510.671609676932 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.3830795 | 0.44014555 | 0.48777180999999986 | 2541.946954447599 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.5765745 | 0.68852495 | 0.80417815 | 1673.7519936059994 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.585708 | 0.5958112 | 0.59759224 | 3406.4994920227955 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5066085 | 0.6064429999999998 | 0.6700332 | 3772.507107497703 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.460152 | 0.53159115 | 0.6098631699999997 | 4370.737110827383 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.405979 | 0.47887945 | 0.49457951999999994 | 4762.407082461317 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5721805 | 0.7146377999999999 | 0.7703007699999999 | 3385.0846198375643 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.6724905 | 0.68788225 | 0.7450547399999998 | 5910.241982446462 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5952335 | 0.8089111 | 0.81926972 | 6195.710474201713 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5028535000000001 | 0.56840925 | 0.6206928599999998 | 7820.6031835720405 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.457416 | 0.5417807 | 0.55610533 | 8698.6199161079 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5502135 | 0.73775565 | 0.7427957399999999 | 6838.110169271605 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.8243324999999999 | 0.8380536 | 0.84061018 | 9684.09091250235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6723595 | 0.8507739 | 0.85572739 | 11451.254728294647 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.60941 | 0.765643 | 0.8422130699999998 | 12784.08555212324 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.5041285 | 0.642293 | 0.66638027 | 15202.811395100049 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.5761324999999999 | 0.7743035 | 0.7807488899999999 | 12978.994665243823 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.0900135 | 1.1016709 | 1.10439789 | 14663.893748798593 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.9703105 | 1.3456474999999999 | 1.36232852 | 15869.749034796907 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.612921 | 0.8522389 | 0.85675927 | 23928.545534466797 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.5606655 | 0.7989773499999999 | 0.8076985 | 26217.686169249653 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.625729 | 0.8272233 | 0.8321972599999999 | 24300.79327509567 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6628785000000001 | 1.6731526 | 1.67513881 | 19238.43431320972 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.080283 | 1.0913286 | 1.25517975 | 29417.641745045687 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.7551375 | 1.0538521999999997 | 1.0985702899999998 | 40096.18372793116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.650081 | 0.9083460499999998 | 0.95573318 | 46501.47328292728 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.6630075 | 0.83603355 | 0.89030764 | 46260.58622918339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 2.825406 | 2.8348934 | 2.91502736 | 22632.686422785795 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.6871935 | 1.69902825 | 1.7474099799999998 | 37867.40255652302 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0757375 | 1.2170871 | 1.22174205 | 58828.796968199116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.786597 | 1.0398612499999997 | 1.1109505400000002 | 77196.87344942843 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.723896 | 0.95288685 | 0.9606885199999999 | 83668.81477633104 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 5.128204 | 5.13700515 | 5.14321848 | 24956.934372539563 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.834862 | 2.8425982000000003 | 2.8926522699999997 | 45136.50651296163 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.6580590000000002 | 1.67068645 | 1.71163702 | 77101.01810328291 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.0953455 | 1.14548345 | 1.2353117299999996 | 115792.75073332545 | - |
| `full_mlp_capacity_search_hd512_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.9052295 | 1.0382463 | 1.1096133399999997 | 137200.8333149863 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 1 | ok | 69.276787 | 0.14760600000000001 | 0.15484635 | 0.17524777999999994 | 6699.786839581912 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 2 | ok | 69.205465 | 0.125407 | 0.13299009999999997 | 0.15623884999999998 | 7884.617143491516 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 4 | ok | 69.002041 | 0.0873045 | 0.0898942 | 0.09449811 | 11402.063910392548 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 8 | ok | 68.706161 | 0.08868000000000001 | 0.10589825 | 0.11136982 | 11010.116535477458 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 1 | 64 | ok | 75.780862 | 0.11057449999999999 | 0.11550745 | 0.11766030999999999 | 9020.547182783888 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 1 | ok | 69.202433 | 0.16001749999999998 | 0.16474704999999998 | 0.16850947 | 12455.172278074675 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 2 | ok | 69.148381 | 0.1506595 | 0.1596465 | 0.17818425999999998 | 13127.742385384307 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 4 | ok | 69.29352 | 0.1099855 | 0.13523225 | 0.13697593 | 17408.128168388128 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 8 | ok | 69.548939 | 0.087231 | 0.1104497 | 0.11126502 | 21687.951583816885 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 2 | 64 | ok | 76.274639 | 0.1162945 | 0.1246408 | 0.13241089999999997 | 17003.766334243035 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 1 | ok | 69.545822 | 0.15845900000000002 | 0.16359100000000001 | 0.16594428 | 25135.13275748744 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 2 | ok | 69.636229 | 0.140449 | 0.1680187 | 0.17194441 | 26975.02719419929 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 4 | ok | 69.704346 | 0.109294 | 0.11244105 | 0.11423026 | 36519.53493833037 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 8 | ok | 69.502719 | 0.0911705 | 0.10507405 | 0.11026168999999998 | 42752.637089537275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 4 | 64 | ok | 75.834696 | 0.12247050000000001 | 0.128731 | 0.13197852 | 32610.164881885168 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 1 | ok | 70.420601 | 0.167674 | 0.17252995 | 0.17972339999999998 | 47492.553464445235 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 2 | ok | 70.356674 | 0.177473 | 0.18185085 | 0.18531281 | 48933.2969432715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 4 | ok | 69.811083 | 0.128291 | 0.13302375 | 0.13703253999999998 | 64698.2336896966 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 8 | ok | 69.67694 | 0.102332 | 0.1071308 | 0.11014849999999998 | 77761.90354918936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 8 | 64 | ok | 76.580712 | 0.1261325 | 0.12988525 | 0.13475974999999998 | 63183.8466704956 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 1 | ok | 71.092965 | 0.218628 | 0.22567345 | 0.22800276 | 72924.63330303921 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 2 | ok | 76.118467 | 0.212056 | 0.2760508 | 0.27988463999999996 | 72824.81826111447 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 4 | ok | 72.22659 | 0.1520215 | 0.18169315 | 0.1843002 | 100385.63140303476 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 8 | ok | 71.912905 | 0.140603 | 0.14671025 | 0.1492344 | 113250.20296559815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 16 | 64 | ok | 76.840997 | 0.137915 | 0.1423294 | 0.14445262 | 115731.46481025682 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 1 | ok | 71.986633 | 0.318892 | 0.32437505 | 0.32615759 | 100098.49065865246 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 2 | ok | 76.938739 | 0.306155 | 0.4069089 | 0.41134713 | 94333.7712778665 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 4 | ok | 73.240972 | 0.2936655 | 0.29922435 | 0.3691644699999999 | 114212.2676396029 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 8 | ok | 72.345907 | 0.210431 | 0.2648686 | 0.26847651 | 139159.54938050517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 32 | 64 | ok | 77.661273 | 7.9964355 | 15.832886299999997 | 19.070177549999993 | 3501.5426341517414 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 1 | ok | 73.111197 | 0.5328280000000001 | 0.5413405499999999 | 0.54576913 | 119797.44049773442 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 2 | ok | 75.258242 | 0.4167345 | 0.4964858 | 0.6154592699999997 | 141698.78062885566 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 4 | ok | 74.733475 | 0.35563500000000003 | 0.4528683 | 0.45584388 | 172116.67854182105 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 8 | ok | 72.888945 | 0.36829449999999997 | 0.42459725 | 0.43054782999999996 | 166822.64584052758 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 64 | 64 | ok | 71.022956 | 5.987858 | 10.9966987 | 11.610612429999998 | 10677.547514686032 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 1 | ok | 76.052782 | 0.983447 | 0.99389595 | 1.00025332 | 129882.97057290241 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 2 | ok | 81.45036 | 0.6986939999999999 | 0.7775899999999998 | 0.8344322599999999 | 181056.18730805747 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 4 | ok | 76.99526 | 0.466608 | 0.54057155 | 0.5410832799999999 | 263477.1115580616 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 8 | ok | 75.264981 | 0.44167049999999997 | 0.59932975 | 0.6092005899999999 | 287534.1070486785 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `fp32` | 128 | 64 | ok | 81.338778 | 0.503136 | 0.54515885 | 0.57713559 | 251361.41663523993 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 1 | ok | 69.223422 | 0.2323955 | 0.2401341 | 0.24164295 | 4288.61154914513 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 2 | ok | 69.357771 | 0.2537085 | 0.27212385 | 0.27891368 | 3941.728015721503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 4 | ok | 69.021751 | 0.22582000000000002 | 0.2442569 | 0.25185561 | 4405.234334678339 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 8 | ok | 69.917162 | 0.2606085 | 0.27337435 | 0.28590765999999995 | 3897.110970312822 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 1 | 64 | ok | 75.220799 | 0.379694 | 0.5213761499999999 | 0.53166267 | 2426.193495530515 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 1 | ok | 68.930926 | 0.311282 | 0.31994639999999996 | 0.32261963 | 6404.577274523175 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 2 | ok | 69.937942 | 0.308197 | 0.3383990999999999 | 0.35713956 | 6597.117218106501 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 4 | ok | 69.460349 | 0.2898165 | 0.3091967 | 0.31320818 | 7031.221930043701 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 8 | ok | 70.125506 | 0.27029899999999996 | 0.30110085 | 0.30396681 | 7282.898436267028 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 2 | 64 | ok | 75.464761 | 0.389882 | 0.5088895999999999 | 1.430117509999997 | 4376.772155045578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 1 | ok | 69.085056 | 0.3878285 | 0.394757 | 0.39774889 | 10290.784114013655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 2 | ok | 70.42758 | 0.4120205 | 0.5014614 | 0.50709492 | 9181.239972364468 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 4 | ok | 69.642692 | 0.334115 | 0.3662586 | 0.37051204 | 12072.64351053157 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 8 | ok | 69.682776 | 0.3055195 | 0.35050834999999997 | 0.36026308999999995 | 13037.321810774485 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 4 | 64 | ok | 69.64138 | 0.419458 | 0.53945695 | 0.55201749 | 9305.4509144676 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 1 | ok | 70.206339 | 0.5363985 | 0.549379 | 0.5551317299999999 | 14872.490885301011 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 2 | ok | 70.354622 | 0.5308555 | 0.7007255 | 0.70556818 | 15406.598230043777 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 4 | ok | 69.766945 | 0.4199 | 0.5982179499999999 | 0.62189314 | 17123.445271435503 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 8 | ok | 69.86781 | 0.350447 | 0.38591285 | 0.39425585999999996 | 22858.83310457356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 8 | 64 | ok | 75.971549 | 0.4316375 | 0.60999765 | 0.61338028 | 17955.969628195613 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 1 | ok | 71.017397 | 0.8316045000000001 | 0.83948895 | 0.83977463 | 19223.31187963669 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 2 | ok | 75.065154 | 0.577557 | 0.8513362 | 0.85903433 | 25595.836683588393 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 4 | ok | 72.825507 | 0.4809295 | 0.7350039 | 0.74519671 | 30095.690376139322 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 8 | ok | 70.642449 | 0.46197350000000004 | 0.568721 | 0.57277849 | 34033.66388812999 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 16 | 64 | ok | 69.77773 | 0.594285 | 1.02286615 | 3.176201459999992 | 20749.388547940413 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 1 | ok | 72.072937 | 1.4197245 | 1.4424167 | 1.45314946 | 22521.118122842283 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 2 | ok | 76.616245 | 0.8608089999999999 | 0.98441615 | 0.98838826 | 36292.79530171618 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 4 | ok | 73.664604 | 0.5832174999999999 | 0.7430612 | 0.8519623599999996 | 50562.63413121275 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 8 | ok | 71.558343 | 0.5247269999999999 | 0.67493645 | 0.67623242 | 58992.03408441746 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 32 | 64 | ok | 71.141975 | 0.5162385 | 0.68441405 | 0.68781968 | 60491.557420174206 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 1 | ok | 73.927768 | 2.5818250000000003 | 2.59060865 | 2.59183642 | 24789.76900823517 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 2 | ok | 76.058492 | 1.4725234999999999 | 1.48221805 | 1.48384294 | 43430.246253975005 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 4 | ok | 75.047447 | 0.9261955 | 0.94224995 | 1.0710683099999996 | 68614.71710634585 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 8 | ok | 72.475251 | 0.645374 | 0.7770722999999999 | 0.8618000899999997 | 93869.65447518883 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 64 | 64 | ok | 78.328197 | 0.5520430000000001 | 0.6956306999999997 | 0.75562485 | 110294.39642425567 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 1 | ok | 75.901023 | 4.8987865 | 4.9195182 | 6.380452719999996 | 25816.764577553102 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 2 | ok | 80.69725 | 2.6243405 | 2.6371045 | 2.6584752899999997 | 48769.58490324087 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 4 | ok | 76.610958 | 1.513997 | 1.52231495 | 1.5246726499999999 | 84578.95893821056 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 8 | ok | 75.141677 | 0.978189 | 0.9958815 | 1.1133573199999995 | 130034.0683163642 | - |
| `full_mlp_capacity_search_hd512_depth5` | `jit` | `bf16` | 128 | 64 | ok | 81.862532 | 0.7928945000000001 | 0.91748805 | 1.02041623 | 155762.73188750798 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1423.256387 | 0.192406 | 0.1964877 | 0.22530650999999993 | 5165.631852856356 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1367.274186 | 0.218839 | 0.2250337 | 0.23086045 | 4776.434219903211 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1329.386497 | 0.143671 | 0.15635735 | 0.15887642 | 6877.1127349961 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1366.711006 | 0.119921 | 0.12876985 | 0.13520931999999997 | 8207.417864265723 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1434.807904 | 0.161061 | 0.16825515 | 0.17200704 | 6254.4922890866355 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1360.630673 | 0.18367050000000001 | 0.18859864999999998 | 0.19202793999999998 | 10867.162629479526 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1330.866184 | 0.180989 | 0.1835528 | 0.18833924999999999 | 11235.915919843424 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1394.337723 | 0.1354225 | 0.13850955 | 0.14209381999999998 | 14726.529814006872 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1402.010803 | 0.121284 | 0.12789695 | 0.14883087999999994 | 16266.304123280366 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1381.265924 | 0.16485 | 0.17230865 | 0.18456920999999998 | 12068.92077900056 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1381.47726 | 0.2458595 | 0.25373924999999997 | 0.2588159 | 16196.307015252225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1407.500055 | 0.233072 | 0.27334655 | 0.27915638 | 16213.18916997875 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1390.324542 | 0.21416849999999998 | 0.22232095 | 0.22672137 | 18598.192832200686 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1298.66477 | 0.143622 | 0.1474096 | 0.15304873 | 27730.639542034034 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1382.639544 | 0.1799505 | 0.1880789 | 0.18850202 | 22241.508081007134 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1389.187436 | 0.2895575 | 0.29614975 | 0.30115282 | 27582.661097911274 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1368.477927 | 0.3129005 | 0.3720242 | 0.37997726 | 24015.78076954367 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1321.559936 | 0.233315 | 0.24264689999999997 | 0.25315213999999997 | 34327.41228380703 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1333.386444 | 0.174933 | 0.1776997 | 0.18222056 | 45721.31373336816 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1404.609993 | 0.2114305 | 0.217837 | 0.22073398 | 37847.72478875529 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1368.481702 | 0.4055015 | 0.41278925 | 0.41837332 | 39350.64161467094 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1308.871339 | 0.553683 | 0.79344475 | 0.8472088599999998 | 27072.482273530815 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1309.328689 | 0.4791485 | 0.65464135 | 0.65944665 | 33345.671231689055 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1362.744487 | 0.38463250000000004 | 0.4935447 | 0.4977048 | 41244.93931039183 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1444.060326 | 0.312751 | 0.3788692 | 0.38123555 | 47743.93780301734 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1327.240577 | 0.513314 | 0.52021095 | 0.5261877500000001 | 62235.41198053463 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1366.471315 | 0.5613625 | 0.8975998999999999 | 0.9014944699999999 | 50988.29659254774 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1373.901162 | 0.5036579999999999 | 0.7051682 | 0.72000077 | 60980.38314432082 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1368.180488 | 0.39129800000000003 | 0.4991156 | 0.50541684 | 76647.7970968018 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1340.755921 | 0.3341655 | 0.41183994999999995 | 0.4568618 | 88655.39520441912 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1381.549246 | 0.730182 | 0.73845325 | 0.7425733099999999 | 87507.77554441527 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1364.45255 | 0.501738 | 0.93220715 | 0.93687868 | 108988.5106696176 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1373.607218 | 0.4732995 | 0.8751325 | 0.88028461 | 112532.87014803982 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1363.376708 | 0.4660835 | 0.63381985 | 0.64010343 | 136320.20559813408 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1399.673579 | 0.40360050000000003 | 0.45505384999999987 | 0.49795847 | 164863.7527307491 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1384.560706 | 1.1574550000000001 | 1.16565035 | 1.166196 | 110463.86573338047 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1338.959573 | 0.7342865000000001 | 0.9608185999999995 | 1.15739068 | 163837.9783704173 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1383.032141 | 0.5619055 | 0.76157165 | 0.7971481499999999 | 217497.48951823526 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1372.003261 | 0.483321 | 0.62445745 | 0.84018026 | 243195.23580533054 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1327.908241 | 0.48049200000000003 | 0.59630075 | 0.59882293 | 258354.45650134457 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1318.707756 | 0.444283 | 0.45295149999999995 | 0.45834077 | 2249.862533399209 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1402.834981 | 0.4413325 | 0.5353641 | 0.5462363699999999 | 2233.774509755742 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1371.766752 | 0.430388 | 0.47451885 | 0.47923531999999996 | 2298.045360749689 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1364.867474 | 0.343511 | 0.3769363 | 0.38867008999999997 | 2879.99670067578 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1401.605966 | 0.5506245 | 0.6706452 | 0.67491148 | 1789.0120738993655 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1308.289699 | 0.522842 | 0.5327115 | 0.53776001 | 3818.32078423116 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1377.893705 | 0.4749755 | 0.56402995 | 0.6090566499999999 | 3987.947624688666 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1403.132053 | 0.4152455 | 0.48943615 | 0.49714776 | 4711.317106646363 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1354.877869 | 0.3736565 | 0.4258244 | 0.43853281 | 5383.751955042225 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1390.166983 | 0.5541739999999999 | 0.6756038 | 0.6894545299999999 | 3556.4044139673233 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1361.773266 | 0.607851 | 0.62017005 | 0.62269071 | 6561.365498896707 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1384.636257 | 0.553901 | 0.7132317499999998 | 0.80293977 | 6731.672634182261 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1386.902835 | 0.47994250000000005 | 0.5694324 | 0.57499985 | 8177.070537495609 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1301.878447 | 0.48541049999999997 | 0.56135575 | 0.5716017499999999 | 8382.692021068722 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1410.497453 | 0.590927 | 0.73528315 | 0.7523031499999999 | 6669.197627166176 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1368.048446 | 0.758121 | 0.76590585 | 0.76995172 | 10542.938952298076 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1393.338113 | 0.6482265 | 0.7899514 | 0.9121402799999996 | 11782.280886984245 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1379.802651 | 0.5696635000000001 | 0.7251261999999999 | 0.73351429 | 13978.176432263364 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1340.080958 | 0.5015615 | 0.6074817499999999 | 0.6197185 | 15648.260895316811 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1424.59042 | 0.597527 | 0.736729 | 0.74523823 | 13103.534465489633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1356.179651 | 1.0670735 | 1.0787446 | 1.08221604 | 14970.798335037633 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1367.233534 | 0.7706305 | 1.09221345 | 1.09932901 | 19545.19551608783 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1379.980845 | 0.6214465 | 0.8508882999999995 | 0.96551409 | 23956.937405014487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1362.66349 | 0.586019 | 0.7485762499999999 | 0.76486107 | 26901.235577112715 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1444.600268 | 0.6600435 | 0.7797935500000001 | 0.7821745600000001 | 24236.67500933415 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1391.252696 | 1.659438 | 1.6710266500000002 | 1.6731600899999999 | 19269.77825651936 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1394.272851 | 1.047446 | 1.1583164999999995 | 1.23939714 | 30194.168887007454 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1305.801044 | 0.708009 | 0.9911806 | 0.9978121999999999 | 42015.68461262471 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1377.702128 | 0.6349115 | 0.8377011999999999 | 0.9094714699999997 | 47971.41335506758 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1411.083704 | 0.6836065 | 0.8244545999999999 | 0.8277490399999999 | 46135.35386191263 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1415.345647 | 2.842031 | 2.8500416 | 2.85487824 | 22520.69366720487 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1310.603499 | 1.655061 | 1.6816842 | 1.8367202199999997 | 38450.251094558495 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1364.853081 | 1.0132984999999999 | 1.1634181499999994 | 1.2603226 | 62162.89016100926 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1365.155971 | 0.702833 | 1.0555424 | 1.06740614 | 83385.79037859832 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1413.538008 | 0.743352 | 0.91322755 | 1.684947749999997 | 78415.03800470584 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1367.221895 | 5.219998 | 5.22989165 | 5.25645528 | 24517.862178185722 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1390.435965 | 2.8502465 | 2.8593798 | 2.86198699 | 44913.232862291065 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1306.958143 | 1.621403 | 1.63469565 | 1.63761972 | 78924.39661867167 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1388.121167 | 1.039188 | 1.1833077 | 1.18397114 | 120632.73679942348 | - |
| `full_mlp_capacity_search_hd512_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1402.765246 | 1.0348255 | 1.1595932 | 1.16213945 | 121546.65074405543 | - |
