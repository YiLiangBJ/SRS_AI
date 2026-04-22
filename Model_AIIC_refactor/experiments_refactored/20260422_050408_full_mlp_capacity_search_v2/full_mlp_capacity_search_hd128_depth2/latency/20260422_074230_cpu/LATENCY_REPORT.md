# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd128_depth2

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1406871.380` samples/s, p50=`0.090` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17565.993` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 1 | 1 | ok | 0.056181999999999996 | 0.0611182 | 0.06421039999999999 | 17565.99280005087 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 1 | 2 | ok | 0.0578765 | 0.06240385 | 0.06582827999999999 | 17054.223904905648 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 1 | 4 | ok | 0.05708 | 0.0598301 | 0.06379734999999999 | 17413.939440587645 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 1 | 8 | ok | 0.059687500000000004 | 0.0638287 | 0.06778245 | 16617.11443143214 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 1 | 64 | ok | 0.0567265 | 0.061446100000000003 | 0.061787089999999996 | 17449.78997432787 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 2 | 1 | ok | 0.070238 | 0.0735726 | 0.07851292999999998 | 28289.87160358874 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 2 | 2 | ok | 0.069756 | 0.07325419999999999 | 0.08509006999999996 | 28405.862970117036 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 2 | 4 | ok | 0.071534 | 0.07440665 | 0.07897815999999999 | 27777.723765537125 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 2 | 8 | ok | 0.06763250000000001 | 0.070156 | 0.08066198999999998 | 29295.332989796727 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 2 | 64 | ok | 0.07045850000000001 | 0.0737578 | 0.08100744999999998 | 28174.077482093962 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 4 | 1 | ok | 0.0683545 | 0.07081475 | 0.07181441 | 58281.9869028719 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 4 | 2 | ok | 0.07126350000000001 | 0.0740904 | 0.07859969999999998 | 55674.683913941415 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 4 | 4 | ok | 0.0688485 | 0.07290115 | 0.07692129999999998 | 57650.810195661084 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 4 | 8 | ok | 0.067605 | 0.07061815 | 0.07457425999999999 | 58815.53741815083 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 4 | 64 | ok | 0.07157649999999999 | 0.07499114999999999 | 0.07957536999999998 | 55506.06264969292 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 8 | 1 | ok | 0.07411899999999999 | 0.07727375 | 0.08603911999999997 | 106951.0423582277 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 8 | 2 | ok | 0.0686855 | 0.07219425 | 0.09700217999999991 | 114240.27929463483 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 8 | 4 | ok | 0.07203999999999999 | 0.07603535 | 0.08189680999999999 | 110204.49546177886 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 8 | 8 | ok | 0.0871555 | 0.09223959999999999 | 0.10050018999999999 | 93643.1297781407 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 8 | 64 | ok | 0.0714255 | 0.07450085 | 0.07751975 | 111466.53454166772 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 16 | 1 | ok | 0.07419600000000001 | 0.0767453 | 0.08060047999999999 | 214679.74746147904 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 16 | 2 | ok | 0.069422 | 0.0730414 | 0.07708351 | 228774.5149765806 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 16 | 4 | ok | 0.0732305 | 0.0762201 | 0.08025679 | 217524.12410512613 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 16 | 8 | ok | 0.0702045 | 0.07356544999999999 | 0.07821298999999998 | 226371.35768485485 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 16 | 64 | ok | 0.069052 | 0.0726176 | 0.07673988999999999 | 229729.92375838154 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 32 | 1 | ok | 0.0764165 | 0.07938855 | 0.08341668999999999 | 416833.4000266774 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 32 | 2 | ok | 0.0735105 | 0.07625975 | 0.08026507999999999 | 431597.58272983856 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 32 | 4 | ok | 0.0721055 | 0.0759894 | 0.08044482 | 438786.2951682499 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 32 | 8 | ok | 0.070157 | 0.10018244999999999 | 0.11551409999999997 | 419307.86946526717 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 32 | 64 | ok | 0.0708635 | 0.07452315 | 0.07911061999999999 | 447766.2481078378 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 64 | 1 | ok | 0.0738615 | 0.0765247 | 0.08832744999999996 | 857641.437803709 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 64 | 2 | ok | 0.0964235 | 0.10120835 | 0.1070912 | 658703.6259782006 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 64 | 4 | ok | 0.088566 | 0.09582575 | 0.10093170999999998 | 713976.2183446372 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 64 | 8 | ok | 0.073532 | 0.07733925 | 0.08267100999999999 | 863388.9364802062 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 64 | 64 | ok | 0.08132249999999999 | 0.08498835 | 0.08905935999999999 | 782045.4104443142 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 128 | 1 | ok | 0.0899885 | 0.09604285 | 0.10487694999999998 | 1406871.379641824 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 128 | 2 | ok | 0.1169325 | 0.12142984999999999 | 0.12582338999999998 | 1088960.7792058855 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 128 | 4 | ok | 0.115554 | 0.1208001 | 0.1229233 | 1103341.5561012116 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 128 | 8 | ok | 0.10867450000000001 | 0.11124665 | 0.11698503999999998 | 1174016.1606993321 | - |
| `full_mlp_capacity_search_hd128_depth2` | `fp32` | 128 | 64 | ok | 0.1525825 | 0.15734875 | 0.16105619999999998 | 836214.8993896286 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 1 | 1 | ok | 0.07117799999999999 | 0.07413344999999999 | 0.07739937 | 13973.580430950811 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 1 | 2 | ok | 0.070558 | 0.0739749 | 0.07595756 | 14094.436672145644 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 1 | 4 | ok | 0.0691745 | 0.0721856 | 0.07381354 | 14379.199624990475 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 1 | 8 | ok | 0.070686 | 0.07409964999999999 | 0.07884776999999998 | 14048.511194977602 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 1 | 64 | ok | 0.0702635 | 0.07295495 | 0.07583302 | 14170.299222872449 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 2 | 1 | ok | 0.3037745 | 0.36672004999999996 | 0.38093286 | 6904.8175049828615 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 2 | 2 | ok | 0.115618 | 0.12329385 | 0.13123987999999998 | 17171.718385879078 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 2 | 4 | ok | 0.1197 | 0.12841545000000001 | 0.13363029999999998 | 16491.442343196482 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 2 | 8 | ok | 0.1386645 | 0.14899185 | 0.15237923 | 14437.341432874377 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 2 | 64 | ok | 0.22983 | 0.24667655 | 0.25183222 | 8691.751467015541 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 4 | 1 | ok | 0.1184025 | 0.12400244999999999 | 0.12939391 | 33537.914612469394 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 4 | 2 | ok | 0.1242325 | 0.1436199 | 0.15462849999999997 | 31321.697322777924 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 4 | 4 | ok | 0.128448 | 0.1345739 | 0.14240524 | 31065.977768254324 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 4 | 8 | ok | 0.1343105 | 0.1433074 | 0.14982785999999998 | 29713.25373053615 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 4 | 64 | ok | 0.231064 | 0.24721495 | 0.24957222999999998 | 17496.43663197444 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 8 | 1 | ok | 0.119004 | 0.12523275 | 0.13163868999999997 | 66787.83092326664 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 8 | 2 | ok | 0.126346 | 0.148722 | 0.15358448000000002 | 61570.62034708899 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 8 | 4 | ok | 0.1274545 | 0.1528009 | 0.15476551 | 60866.36576364313 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 8 | 8 | ok | 0.1360315 | 0.1470076 | 0.15093852 | 58671.750769443344 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 8 | 64 | ok | 0.2291505 | 0.24327525 | 0.25303918999999997 | 34738.78647170912 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 16 | 1 | ok | 0.1177665 | 0.12364285 | 0.12761255999999999 | 135036.09345983065 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 16 | 2 | ok | 0.12919350000000002 | 0.14968135 | 0.15693179999999998 | 120522.11384940702 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 16 | 4 | ok | 0.1328125 | 0.1556901 | 0.16384955999999998 | 117655.60615727086 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 16 | 8 | ok | 0.1429085 | 0.20118609999999995 | 0.21526216999999997 | 106220.93529657947 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 16 | 64 | ok | 0.23354049999999998 | 0.24906984999999998 | 0.25474733 | 68462.99297099009 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 32 | 1 | ok | 0.12811250000000002 | 0.13653754999999998 | 0.14135913999999997 | 247654.74823031333 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 32 | 2 | ok | 0.1397445 | 0.14574665 | 0.15622962999999995 | 227269.14390980484 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 32 | 4 | ok | 0.14062550000000001 | 0.16801354999999998 | 0.16979946 | 220136.9444414252 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 32 | 8 | ok | 0.15661550000000002 | 0.20944784999999996 | 0.22263406 | 195396.36396676113 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 32 | 64 | ok | 0.2415745 | 0.25944544999999997 | 0.26108874 | 131925.49046186949 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 64 | 1 | ok | 0.13594699999999998 | 0.14868854999999997 | 0.15422518999999998 | 464581.8748608069 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 64 | 2 | ok | 0.1643665 | 0.1882646 | 0.19129379 | 386212.92280508863 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 64 | 4 | ok | 0.1612205 | 0.175768 | 0.18423456000000002 | 394629.04931025626 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 64 | 8 | ok | 0.172296 | 0.249601 | 0.26037797999999995 | 351060.72999568196 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 64 | 64 | ok | 0.2511865 | 0.26767275 | 0.27093457 | 254657.85126071557 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 128 | 1 | ok | 0.15996549999999998 | 0.17160455 | 0.18159989999999998 | 794177.4877485713 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 128 | 2 | ok | 0.20796 | 0.23911775 | 0.25005015999999997 | 615963.858320613 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 128 | 4 | ok | 0.20762049999999999 | 0.24741649999999998 | 0.25523635 | 600041.102815543 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 128 | 8 | ok | 0.207037 | 0.2189669 | 0.22054703 | 619878.6761832249 | - |
| `full_mlp_capacity_search_hd128_depth2` | `bf16` | 128 | 64 | ok | 0.281528 | 0.7611907499999999 | 0.8080492099999999 | 356710.5446646748 | - |
