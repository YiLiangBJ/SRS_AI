# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`829699.401` samples/s, p50=`0.153` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.072` ms, throughput=`13898.718` samples/s

### full_mlp_capacity_search_hd128_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1381366.914` samples/s, p50=`0.093` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.041` ms, throughput=`24190.431` samples/s

### full_mlp_capacity_search_hd128_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1182789.161` samples/s, p50=`0.107` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.058` ms, throughput=`17146.712` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `21,776`
- MACs / sample: `21,504`
- FLOPs / sample estimate: `43,352`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0715205 | 0.0739833 | 0.07983170999999999 | 13898.718371381538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0733515 | 0.07654665 | 0.08199347999999998 | 13521.772758495728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.074188 | 0.07815884999999999 | 0.08151180999999999 | 13369.491568931226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0722495 | 0.0753617 | 0.07771815 | 13771.244209880371 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.073265 | 0.0753668 | 0.08332596999999997 | 13567.976102995051 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09957250000000001 | 0.10223195 | 0.11177054999999998 | 20241.530032965355 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.088337 | 0.0924539 | 0.09817775999999999 | 22446.200385760396 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.08596200000000001 | 0.08824295 | 0.09314971999999999 | 23154.37088532358 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0859805 | 0.08893995 | 0.09343193 | 23135.562830404757 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.08833350000000001 | 0.09074385 | 0.09671783999999999 | 22544.574569223907 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.087729 | 0.09197055 | 0.10000968999999998 | 45212.13647464243 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0888295 | 0.0925031 | 0.09871484999999999 | 44646.255440425266 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.087906 | 0.09076705 | 0.09503252 | 45310.42740193408 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.086918 | 0.09013195 | 0.09660366999999999 | 45722.37465553906 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.09109 | 0.10888555 | 0.11145775 | 41206.951612737066 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0909395 | 0.0943138 | 0.10014071999999999 | 87271.55307364956 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.093303 | 0.09728695 | 0.10362394999999998 | 85203.5331349085 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.119721 | 0.12512245 | 0.13334608999999997 | 66399.48620077578 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.096371 | 0.1005365 | 0.10782069999999998 | 82305.27189958098 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.09342249999999999 | 0.09776874999999999 | 0.10262959999999999 | 85109.20788009133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0947355 | 0.099981 | 0.10496824999999999 | 167281.9855702559 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1305715 | 0.13620365 | 0.13996357 | 121712.48862939111 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.122039 | 0.1260104 | 0.12754189999999999 | 130755.1683023321 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.113764 | 0.1171559 | 0.12115436 | 139923.49682810923 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.15366600000000002 | 0.1587858 | 0.16174903999999998 | 104038.2309287194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.1036115 | 0.11001214999999999 | 0.11556474 | 306135.6081249921 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1430285 | 0.14810555 | 0.15276938999999998 | 222699.16961047132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.12735000000000002 | 0.13142715 | 0.13538125 | 249992.7345861511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.1182665 | 0.1228638 | 0.12694665 | 268575.88309428963 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.15071099999999998 | 0.1573543 | 0.16320144 | 211527.5088881215 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.11647350000000001 | 0.12158939999999999 | 0.12753354 | 547092.9021876536 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.166217 | 0.17237834999999999 | 0.17975954 | 383593.1917482871 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1523875 | 0.15808904999999998 | 0.16098476 | 418640.54593867 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.13701000000000002 | 0.14188355 | 0.1472434 | 464578.03030819964 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.1701495 | 0.1759387 | 0.18190122999999997 | 376365.56010487425 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15330549999999998 | 0.16060534999999998 | 0.16914501999999998 | 829699.4012033493 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2349215 | 0.24325724999999998 | 0.24643218 | 568168.0982135375 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2049085 | 0.21740795000000002 | 0.22654488999999997 | 613324.4350179887 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1768135 | 0.1830185 | 0.18717171999999999 | 720585.3675233077 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.24771549999999998 | 0.26256685 | 0.27208963999999997 | 518246.48709857767 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1378055 | 0.14507565 | 0.1523472 | 7216.539037362764 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1415095 | 0.14925295 | 0.15076787 | 7015.330461350581 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.152177 | 0.1588618 | 0.15970072 | 6553.17871763467 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.16100399999999998 | 0.16767325 | 0.17116046 | 6236.652005545132 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.2555205 | 0.27694585 | 0.28536519 | 3885.0346754884927 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.18026399999999998 | 0.18964105 | 0.19279367 | 11022.485429376511 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.17940699999999998 | 0.1842847 | 0.18740426 | 11115.12119983732 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.186744 | 0.19560809999999998 | 0.20149369999999997 | 10654.34739991306 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.2061075 | 0.21921715 | 0.23167205 | 9648.979765124534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.3488605 | 0.47489644999999997 | 0.4963931199999999 | 5232.0270583696965 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1752395 | 0.18265284999999998 | 0.18555901 | 22720.37657660953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.193908 | 0.2011919 | 0.20524626999999998 | 20546.97485348209 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.186114 | 0.1953665 | 0.19898583 | 21362.83797611891 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.211369 | 0.2251291 | 0.22787 | 18972.57438482376 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.3710555 | 0.40704914999999997 | 0.41285611 | 10673.495996344967 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.18182 | 0.19200885 | 0.19509751 | 43645.14662696122 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.209858 | 0.21610059999999998 | 0.24082315999999993 | 37865.276859544814 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.2129 | 0.23102209999999998 | 0.23293327 | 37234.54561566949 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.207774 | 0.22173585 | 0.22611472999999999 | 38418.314164035255 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.376795 | 0.4181588 | 0.43735785 | 21004.23891796663 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.20177499999999998 | 0.2065553 | 0.21318722999999998 | 78976.39484406555 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.235993 | 0.24594834999999998 | 0.24904677 | 67469.02518137993 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2217945 | 0.2331892 | 0.23588874999999998 | 71790.6333863238 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.2316165 | 0.2436084 | 0.25248528 | 69050.33605503864 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.38429599999999997 | 0.41323545 | 0.43685804 | 42116.46835513344 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.22803800000000002 | 0.2375448 | 0.24018066999999999 | 139634.75910604082 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.28871800000000003 | 0.2974076 | 0.30045872 | 113506.99951553794 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.250448 | 0.2620404 | 0.27117053 | 127188.39558516361 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.2482435 | 0.25852559999999997 | 0.26001857 | 128986.31189196558 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.38550549999999995 | 0.433415 | 0.45145434999999995 | 82323.97282321008 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2689695 | 0.2787253 | 0.28133205 | 237020.11204533247 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.334503 | 0.39063255 | 0.39477126999999995 | 192260.23961513827 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.287397 | 0.30039075 | 0.30650214000000003 | 224377.83881789903 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2804675 | 0.3058049 | 0.30908775 | 227484.10330422793 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.401379 | 0.54321195 | 0.6582047899999997 | 150279.51520204093 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3673285 | 0.37645435 | 0.38089597999999997 | 347572.887663791 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.427747 | 0.5362209 | 0.55109367 | 291714.9268750444 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3640445 | 0.42464445 | 0.42790045000000004 | 349132.373321137 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.3365415 | 0.385465 | 0.39090159 | 381628.2036122421 | - |
| `full_mlp_capacity_search_hd128_depth3` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.4452275 | 0.5379837 | 0.6439076899999998 | 278620.8840893152 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 1 | ok | 58.179314 | 0.0405595 | 0.0468533 | 0.04831658 | 24190.430942851075 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 2 | ok | 58.178849 | 0.043552 | 0.04500615 | 0.046822329999999995 | 23000.16974125269 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 4 | ok | 58.86282 | 0.046994 | 0.04932304999999999 | 0.05346506999999999 | 21141.702685080807 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 8 | ok | 58.53695 | 0.04253 | 0.050947799999999994 | 0.05523631999999999 | 22678.699077249094 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 1 | 64 | ok | 58.227127 | 0.0428035 | 0.050162099999999994 | 0.051387709999999996 | 22866.059313643218 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 1 | ok | 58.518144 | 0.0447775 | 0.04762555 | 0.04944307 | 44345.111394919826 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 2 | ok | 59.469285 | 0.042379 | 0.045060699999999995 | 0.04873684999999999 | 46826.10330493955 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 4 | ok | 58.964866 | 0.043302 | 0.0472329 | 0.05061653999999999 | 45681.958550018084 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 8 | ok | 58.711841 | 0.0486375 | 0.0525705 | 0.05563941999999999 | 40623.769861468885 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 2 | 64 | ok | 67.753679 | 0.0438165 | 0.045668749999999994 | 0.04977407 | 45269.04525320109 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 1 | ok | 58.423607 | 0.043233999999999995 | 0.04515255 | 0.04700157 | 92029.46599442209 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 2 | ok | 58.585985 | 0.0423005 | 0.0444606 | 0.045870179999999997 | 93902.97381327768 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 4 | ok | 58.869901 | 0.0437485 | 0.047372899999999996 | 0.049663259999999994 | 90484.47195976699 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 8 | ok | 58.41096 | 0.0421595 | 0.045281499999999995 | 0.04837858999999999 | 93946.68619505117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 4 | 64 | ok | 58.442931 | 0.0487555 | 0.05353055 | 0.06178921999999999 | 80463.34009761813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 1 | ok | 58.29193 | 0.044308 | 0.04621965 | 0.048940309999999994 | 179408.64220399928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 2 | ok | 58.361778 | 0.044145500000000004 | 0.04760249999999999 | 0.05112846999999999 | 179037.24511810194 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 4 | ok | 59.315057 | 0.0489395 | 0.05158405 | 0.05247133 | 162855.90631551133 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 8 | ok | 58.656934 | 0.043459 | 0.046248199999999996 | 0.04933028999999999 | 182312.5803884534 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 8 | 64 | ok | 59.040235 | 0.05007 | 0.0524166 | 0.05504302 | 159317.67426465932 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 1 | ok | 59.495944 | 0.050956 | 0.0532253 | 0.0554999 | 315266.58745593653 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 2 | ok | 59.176971 | 0.057482000000000005 | 0.06139265 | 0.06411034 | 276124.5777451184 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 4 | ok | 58.914773 | 0.050814 | 0.054176249999999995 | 0.05681028999999999 | 312345.5353719606 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 8 | ok | 60.121828 | 0.0509205 | 0.053670949999999995 | 0.056806369999999995 | 312037.67230817775 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 16 | 64 | ok | 65.484562 | 0.087177 | 0.0916281 | 0.09666446 | 183145.6548235529 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 1 | ok | 58.511004 | 0.054624 | 0.057837599999999996 | 0.06200859999999998 | 580452.3247296998 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 2 | ok | 60.135919 | 0.06799949999999999 | 0.07016055 | 0.07483569999999999 | 475349.7014061141 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 4 | ok | 59.045771 | 0.058523 | 0.06170994999999999 | 0.06500481999999999 | 542366.8142017394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 8 | ok | 59.354263 | 0.055774000000000004 | 0.0602399 | 0.06330245999999999 | 562690.5894570774 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 32 | 64 | ok | 65.612451 | 0.0927705 | 0.09807814999999999 | 0.10243398 | 343020.0038546873 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 1 | ok | 59.471214 | 0.0640815 | 0.06805564999999998 | 0.07258639 | 990444.0717898625 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 2 | ok | 59.772215 | 0.11619850000000001 | 0.12154855 | 0.12588950000000002 | 549357.4492097578 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 4 | ok | 59.889319 | 0.0763215 | 0.0882931 | 0.09022168 | 822758.3178294813 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 8 | ok | 60.610628 | 0.0706345 | 0.0771217 | 0.08025399999999999 | 907167.5592344981 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 64 | 64 | ok | 66.245536 | 0.104449 | 0.11207545 | 0.11588161 | 607944.1968021755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 1 | ok | 59.677405 | 0.09273300000000001 | 0.098922 | 0.10593066999999999 | 1381366.9143630217 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 2 | ok | 61.224494 | 0.15813549999999998 | 0.16315355 | 0.16477547 | 813087.1457909718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 4 | ok | 60.78083 | 0.1351215 | 0.1701404 | 0.17232232 | 866974.0721921183 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 8 | ok | 60.833377 | 0.097076 | 0.10221585 | 0.10414798 | 1311415.4824482405 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `fp32` | 128 | 64 | ok | 69.499173 | 0.170919 | 0.1795112 | 0.18567060999999999 | 748328.9172183351 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 1 | ok | 58.186169 | 0.0828 | 0.08891205 | 0.09135736 | 11955.027100850935 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 2 | ok | 58.461078 | 0.091258 | 0.0969608 | 0.09957711 | 10890.024258618037 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 4 | ok | 58.343334 | 0.093829 | 0.10099564999999999 | 0.10527937999999998 | 10501.115323458504 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 8 | ok | 58.611583 | 0.104189 | 0.11718214999999998 | 0.12191877999999999 | 9587.885821319394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 1 | 64 | ok | 57.859068 | 0.2156805 | 0.23299915 | 0.23487574 | 4634.167681240445 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 1 | ok | 58.26433 | 0.10864499999999999 | 0.11728019999999999 | 0.11902489 | 18242.82217918538 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 2 | ok | 58.571961 | 0.119205 | 0.1240303 | 0.12995206999999998 | 16716.705337978347 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 4 | ok | 58.512347 | 0.11758199999999999 | 0.13230414999999998 | 0.13275704 | 16763.46449850252 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 8 | ok | 59.257747 | 0.144113 | 0.16066425 | 0.16503448999999998 | 13789.527681097867 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 2 | 64 | ok | 58.662827 | 0.32142950000000003 | 0.3593699 | 0.36040536 | 6160.567427687721 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 1 | ok | 58.326896 | 0.1110265 | 0.11567279999999999 | 0.12209671999999999 | 35853.236211696865 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 2 | ok | 58.743658 | 0.1320595 | 0.13888045 | 0.14147997 | 30165.239146999436 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 4 | ok | 58.893813 | 0.12485299999999999 | 0.1339598 | 0.13752384 | 31787.264368956046 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 8 | ok | 58.645695 | 0.138897 | 0.15061634999999998 | 0.15403592 | 28750.83790723226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 4 | 64 | ok | 58.782223 | 0.3227815 | 0.36888109999999996 | 0.37246354 | 12308.436873442868 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 1 | ok | 58.833128 | 0.1177645 | 0.1293056 | 0.13430806 | 67029.31761808638 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 2 | ok | 58.827194 | 0.142318 | 0.1460714 | 0.15013528 | 56634.52987464372 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 4 | ok | 59.49707 | 0.133565 | 0.1410377 | 0.14399983 | 59928.62201474933 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 8 | ok | 58.646429 | 0.142665 | 0.1544024 | 0.15670651 | 55651.27574984529 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 8 | 64 | ok | 58.493635 | 0.308158 | 0.33396804999999996 | 0.34660988 | 26263.181901358512 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 1 | ok | 58.717346 | 0.127692 | 0.13731274999999998 | 0.14374622 | 123893.82175581704 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 2 | ok | 58.670238 | 0.16786600000000002 | 0.17487049999999998 | 0.18056997999999996 | 94668.70803957435 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 4 | ok | 59.295925 | 0.152886 | 0.1628751 | 0.16899255 | 104971.04832880184 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 8 | ok | 59.844984 | 0.160395 | 0.17665685 | 0.18039525 | 98815.15686165038 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 16 | 64 | ok | 65.628504 | 0.332592 | 0.3782515 | 0.3892563 | 47449.683317847936 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 1 | ok | 58.80409 | 0.154087 | 0.1614344 | 0.16916203 | 206574.67810178013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 2 | ok | 59.253149 | 0.217254 | 0.2234831 | 0.2278286 | 148795.5327117214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 4 | ok | 59.884037 | 0.18394749999999999 | 0.1991619 | 0.20756919 | 174160.18056056718 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 8 | ok | 59.358103 | 0.1717075 | 0.19002884999999997 | 0.19495889 | 185321.0443860117 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 32 | 64 | ok | 65.659755 | 0.34034 | 0.37661705 | 0.39155062999999996 | 94105.93917286172 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 1 | ok | 59.195534 | 0.20543050000000002 | 0.21444839999999998 | 0.21866191 | 309701.5261220172 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 2 | ok | 60.309529 | 0.3211715 | 0.32932324999999996 | 0.33336074000000004 | 217070.18204590818 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 4 | ok | 59.340475 | 0.213764 | 0.24910275 | 0.25390904 | 288389.38045760005 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 8 | ok | 59.90094 | 0.21811750000000002 | 0.23545824999999998 | 0.24596325 | 298048.8511381042 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 64 | 64 | ok | 66.412726 | 0.32521449999999996 | 0.4498276 | 0.45533178999999996 | 191230.38947295828 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 1 | ok | 60.904298 | 0.312432 | 0.32080225 | 0.32890544 | 409097.30123624095 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 2 | ok | 61.063701 | 0.38491450000000005 | 0.42106855 | 0.43175009 | 340159.51993286953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 4 | ok | 60.799391 | 0.3297675 | 0.3398652 | 0.34477697 | 410735.70723169524 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 8 | ok | 60.630496 | 0.275993 | 0.30321529999999997 | 0.33250346 | 466457.03795845166 | - |
| `full_mlp_capacity_search_hd128_depth3` | `jit` | `bf16` | 128 | 64 | ok | 69.984896 | 0.4168265 | 0.45339294999999996 | 2.212817879999993 | 267872.95438713743 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 1 | ok | 1368.257395 | 0.0578395 | 0.0606407 | 0.06884605999999999 | 17146.71172365262 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 2 | ok | 1366.197207 | 0.0596305 | 0.0617937 | 0.06778355999999999 | 16629.920086582017 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 4 | ok | 1375.920875 | 0.06402150000000001 | 0.06608025 | 0.07658757999999999 | 15495.654863419748 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 8 | ok | 1384.516571 | 0.0613135 | 0.0639316 | 0.07807553999999997 | 16133.627739772328 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 1 | 64 | ok | 1503.241679 | 0.060555 | 0.064375 | 0.0672572 | 16403.564428936148 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 1 | ok | 1363.318601 | 0.0586615 | 0.06030215 | 0.06103806 | 33987.60607956703 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 2 | ok | 1364.437231 | 0.061025499999999996 | 0.06398175 | 0.07252506999999998 | 32560.074966316603 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 4 | ok | 1406.953289 | 0.058143 | 0.0607184 | 0.0669469 | 34145.39038937013 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 8 | ok | 1358.519645 | 0.059286 | 0.06134195 | 0.06536231999999999 | 33543.35948820884 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 2 | 64 | ok | 1499.190711 | 0.059528 | 0.0612813 | 0.06522462 | 33419.991370958225 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 1 | ok | 1362.275529 | 0.061688 | 0.0648688 | 0.06873924999999999 | 64374.09071596863 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 2 | ok | 1380.080613 | 0.0635665 | 0.0661431 | 0.07136544999999998 | 62555.79194694268 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 4 | ok | 1325.466605 | 0.060947 | 0.06301245 | 0.06377806999999999 | 65307.252664291 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 8 | ok | 1391.413213 | 0.07028899999999999 | 0.0724733 | 0.08065475999999996 | 56805.72329023294 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 4 | 64 | ok | 1528.34204 | 0.060226 | 0.06202665 | 0.06895693999999998 | 66103.11557204314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 1 | ok | 1353.331178 | 0.062958 | 0.06525405000000001 | 0.06848024999999999 | 126453.46401877329 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 2 | ok | 1323.881805 | 0.0671645 | 0.0693898 | 0.07261769 | 118773.37979747953 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 4 | ok | 1386.151843 | 0.09293950000000001 | 0.0951295 | 0.09973795999999999 | 85736.98764379969 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 8 | ok | 1395.789129 | 0.0659035 | 0.0682711 | 0.0720097 | 120994.00202483464 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 8 | 64 | ok | 1542.602648 | 0.063858 | 0.06600015 | 0.06925713 | 124659.71792622325 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 1 | ok | 1370.203181 | 0.07145950000000001 | 0.07579425 | 0.07883714 | 222740.34100432508 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 2 | ok | 1396.681711 | 0.107465 | 0.11294919999999999 | 0.11886652 | 147973.64885261233 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 4 | ok | 1448.484964 | 0.09201899999999999 | 0.09502334999999999 | 0.10498523 | 172633.34253325188 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 8 | ok | 1399.922643 | 0.084588 | 0.0870608 | 0.10935060999999995 | 187017.96120499415 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 16 | 64 | ok | 1567.454171 | 0.10675 | 0.1104343 | 0.11833714999999997 | 149342.9749157939 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 1 | ok | 1333.857866 | 0.0747275 | 0.0766837 | 0.07972807999999999 | 426746.55362150463 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 2 | ok | 1365.84838 | 0.11333399999999999 | 0.1193197 | 0.12839384999999998 | 280128.2777415848 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 4 | ok | 1393.680232 | 0.0998465 | 0.10223435 | 0.10739963999999999 | 319321.25074940705 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 8 | ok | 1344.541645 | 0.0910695 | 0.09325385 | 0.09627979999999998 | 350936.7598798568 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 32 | 64 | ok | 1489.99983 | 0.1008995 | 0.10376525 | 0.10644580999999999 | 317187.43557130214 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 1 | ok | 1367.027573 | 0.086831 | 0.08929795 | 0.09486724999999999 | 733937.2107541985 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 2 | ok | 1373.574631 | 0.1360245 | 0.139263 | 0.14042548 | 469549.2195431667 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 4 | ok | 1400.647157 | 0.1235495 | 0.12674725 | 0.13857461999999998 | 515090.37743569334 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 8 | ok | 1420.131848 | 0.109318 | 0.11375215 | 0.12352211999999999 | 580786.5992204029 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 64 | 64 | ok | 1541.698314 | 0.1165835 | 0.1211807 | 0.12752404999999997 | 546848.698927322 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 1 | ok | 1324.038198 | 0.10714599999999999 | 0.11434729999999999 | 0.12193646999999999 | 1182789.160994053 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 2 | ok | 1390.460659 | 0.207613 | 0.2105651 | 0.21677633000000002 | 647069.6238826928 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 4 | ok | 1325.769035 | 0.1694405 | 0.17261680000000001 | 0.17670196 | 772489.150751982 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 8 | ok | 1382.857362 | 0.1428505 | 0.14814 | 0.15420167999999998 | 894331.5172348163 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `fp32` | 128 | 64 | ok | 1519.33739 | 0.15457549999999998 | 0.15857084999999999 | 0.16036494 | 829898.0094247886 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 1 | ok | 1400.218777 | 0.117293 | 0.12401754999999998 | 0.12938924 | 8467.434922682158 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 2 | ok | 1316.299323 | 0.1190495 | 0.1266633 | 0.13353871 | 8340.808087514428 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 4 | ok | 1415.15151 | 0.118699 | 0.13123955 | 0.13213695 | 8306.920096400147 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 8 | ok | 1397.989507 | 0.1323105 | 0.1440162 | 0.14894412999999998 | 7565.1754999673185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 1 | 64 | ok | 1574.831905 | 0.2315695 | 0.2479287 | 0.25087831 | 4328.647168766075 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 1 | ok | 1376.376819 | 0.151423 | 0.16166139999999998 | 0.1642804 | 13109.649501721888 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 2 | ok | 1334.395325 | 0.16139399999999998 | 0.16463085 | 0.17355088999999999 | 12362.811426946602 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 4 | ok | 1378.54009 | 0.252687 | 0.26362195 | 0.26908781 | 8403.002628123102 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 8 | ok | 1385.532292 | 0.1774825 | 0.18588339999999998 | 0.19389575 | 11317.24594367271 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 2 | 64 | ok | 1524.253908 | 0.360153 | 0.39194675 | 0.40172803999999995 | 5627.069038116134 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 1 | ok | 1311.574801 | 0.1529665 | 0.16303335 | 0.16607854 | 25945.310916930637 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 2 | ok | 1392.944158 | 0.170456 | 0.1796789 | 0.18202860999999998 | 23364.464145360613 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 4 | ok | 1385.839323 | 0.166623 | 0.17921355 | 0.18242785 | 23725.1864206523 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 8 | ok | 1328.868908 | 0.2334105 | 0.29980455 | 0.30350241 | 15886.976238167383 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 4 | 64 | ok | 1555.594589 | 0.3500065 | 0.38901454999999996 | 0.39225217 | 11347.231876825343 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 1 | ok | 1381.860745 | 0.152473 | 0.16439094999999998 | 0.16700173 | 52098.39981067441 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 2 | ok | 1375.227362 | 0.182135 | 0.18922475 | 0.19248591 | 43742.24532007185 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 4 | ok | 1388.270642 | 0.243156 | 0.28234005 | 0.28568832 | 33524.31015350782 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 8 | ok | 1316.056235 | 0.192628 | 0.20611174999999998 | 0.20814725 | 41182.76488934552 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 8 | 64 | ok | 1581.730327 | 0.3574535 | 0.39036005 | 0.40021421 | 22606.934473687394 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 1 | ok | 1361.698108 | 0.16947050000000002 | 0.17806724999999998 | 0.18208495 | 93997.19794352929 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 2 | ok | 1380.694701 | 0.212813 | 0.2192372 | 0.22459832999999998 | 76015.28629399728 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 4 | ok | 1411.620247 | 0.1922375 | 0.20428834999999998 | 0.20785327999999997 | 82440.95295558545 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 8 | ok | 1336.978337 | 0.2040795 | 0.2183872 | 0.22737597999999998 | 77813.2606246226 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 16 | 64 | ok | 1480.879244 | 0.356126 | 0.40142304999999995 | 0.40776267 | 44898.946502344006 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 1 | ok | 1395.189283 | 0.1941235 | 0.20180800000000002 | 0.20705627999999998 | 164188.44729037755 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 2 | ok | 1329.389722 | 0.2515675 | 0.25709485 | 0.26336645 | 130519.93267781874 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 4 | ok | 1411.938306 | 0.22491250000000002 | 0.2427767 | 0.24404445 | 142134.73039928757 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 8 | ok | 1388.966757 | 0.2160445 | 0.2283651 | 0.23297421 | 148312.92655608762 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 32 | 64 | ok | 1466.204609 | 0.3812175 | 0.4347908 | 0.44455811 | 81871.89443274189 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 1 | ok | 1331.946293 | 0.250904 | 0.2580197 | 0.26022565999999997 | 254264.552891278 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 2 | ok | 1370.604878 | 0.3070155 | 0.37021390000000004 | 0.37345971 | 203819.63083678912 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 4 | ok | 1387.690151 | 0.269034 | 0.28144205 | 0.28433005 | 242420.97341414634 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 8 | ok | 1388.410859 | 0.25006700000000004 | 0.26919035 | 0.27781282 | 257166.52794407654 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 64 | 64 | ok | 1598.795064 | 0.370212 | 0.40605615 | 0.4321883299999999 | 172522.63294769314 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 1 | ok | 1317.716051 | 0.357355 | 0.3638933 | 0.36696667 | 357703.3855339944 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 2 | ok | 1382.555747 | 0.38418300000000005 | 0.46636835 | 0.46933655 | 313539.56430836086 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 4 | ok | 1383.711493 | 0.3196605 | 0.37986525 | 0.38358271 | 402783.1560867175 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 8 | ok | 1407.062113 | 0.31973 | 0.348663 | 0.35831963 | 404364.22728681396 | - |
| `full_mlp_capacity_search_hd128_depth3` | `compile` | `bf16` | 128 | 64 | ok | 1560.985262 | 0.41463300000000003 | 0.46474115 | 0.4665455 | 306603.0708022183 | - |
