# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`334041.937` samples/s, p50=`0.391` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.111` ms, throughput=`8925.307` samples/s

### full_mlp_capacity_search_hd256_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`436732.685` samples/s, p50=`0.276` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.064` ms, throughput=`15496.462` samples/s

### full_mlp_capacity_search_hd256_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`64`, batch=`128`, precision=`fp32`, throughput=`356939.708` samples/s, p50=`0.360` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.095` ms, throughput=`10477.568` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `174,992`
- MACs / sample: `174,080`
- FLOPs / sample estimate: `349,144`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.121454 | 0.13098155 | 0.13853630999999997 | 8127.027388569922 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.134858 | 0.14240909999999998 | 0.15151247999999998 | 7352.255686749207 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.1164325 | 0.12394029999999999 | 0.1301754 | 8528.844295542194 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.11125850000000001 | 0.11622865 | 0.11996934 | 8925.307316181512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.122877 | 0.1496968 | 0.15360200999999998 | 7854.289730296257 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1465685 | 0.15491024999999997 | 0.16402411 | 13538.266587422679 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.144762 | 0.157972 | 0.16656209 | 13645.96419244412 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.1370905 | 0.14666964999999998 | 0.15342166999999998 | 14468.400579083263 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.14450200000000002 | 0.1565473 | 0.16327761999999998 | 13683.649352653916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.135927 | 0.1772317 | 0.18138528999999998 | 14028.324590180033 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.16531 | 0.17450635 | 0.18141422999999998 | 24002.10642485985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.168546 | 0.1774811 | 0.1852865 | 23554.64571810695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.154857 | 0.1605353 | 0.17332355999999996 | 25663.030044222534 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1416545 | 0.1494907 | 0.1610506 | 27954.98129811751 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.1732525 | 0.18149235 | 0.18303262 | 22965.421538924384 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1779925 | 0.1871138 | 0.19058427 | 44575.9087049729 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.2050145 | 0.2137834 | 0.22209259 | 38806.50591071593 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1714595 | 0.1793326 | 0.18706654 | 47441.00829221384 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.1624675 | 0.17012665 | 0.1784487 | 48901.244059874196 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.177558 | 0.18296735 | 0.18375677 | 44937.00898594134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.2381585 | 0.24936334999999998 | 0.25520857 | 66855.29330545357 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.382412 | 0.42245835 | 0.42717601 | 42125.77619375498 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.3326675 | 0.3607084 | 0.37111953 | 49953.41219894795 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.2504075 | 0.25606819999999997 | 0.26500144 | 67595.99899281962 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.220051 | 0.2376857 | 0.24025617 | 71178.55041501099 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.280287 | 0.2929669 | 0.29610473 | 113614.15723290447 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.452434 | 0.6152384000000001 | 0.6263918199999999 | 72319.25733908253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.331926 | 0.41250404999999996 | 0.42575987 | 92164.47049293994 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2768595 | 0.29097095 | 0.30092285999999996 | 116687.81633337711 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.245803 | 0.25139195 | 0.25649241 | 129789.4223405397 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.3649905 | 0.37559205 | 0.37760702 | 174833.49431673676 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.412292 | 0.5133845 | 0.52033111 | 163547.5257354809 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.409285 | 0.42418904999999996 | 0.44422238 | 169568.33413101104 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3097245 | 0.36891874999999996 | 0.38148426999999996 | 193453.63771731718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.3233745 | 0.3673325 | 0.37325443 | 191163.71989783732 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.528655 | 0.5429396000000001 | 0.54874772 | 241157.83493921845 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.5197780000000001 | 0.6904985499999999 | 0.6930710999999999 | 248973.87750516346 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.441297 | 0.6233791499999998 | 0.65059406 | 269370.291386262 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.453094 | 0.5353921499999997 | 0.57906125 | 292503.5009012764 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.3907775 | 0.40286259999999996 | 0.41790755999999996 | 334041.9371906165 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.3208255 | 0.3313957 | 0.3549314099999999 | 3095.703998689898 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.3505865 | 0.3765903 | 0.38007431999999997 | 2808.854159966379 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.346797 | 0.3801299 | 0.3855569 | 2815.96476304445 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.374154 | 0.40986739999999994 | 0.41929363999999997 | 2702.787292640132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.48688 | 0.8495318999999999 | 0.88386755 | 1757.0783638325238 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3633805 | 0.37283355 | 0.3943902799999999 | 5472.335320192359 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.35419999999999996 | 0.38797755 | 0.39064692 | 5535.637715982592 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.35282650000000004 | 0.3893683 | 0.39456025 | 5535.426591801867 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.38464149999999997 | 0.42604 | 0.43422315 | 5192.718666032116 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.5330379999999999 | 0.6702084 | 0.6951103999999999 | 3568.3932706240553 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.375395 | 0.38314695 | 0.38658984 | 10620.074840729409 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.42640199999999995 | 0.43584 | 0.45332653999999994 | 9730.782337384519 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.382255 | 0.4033586 | 0.41923027999999996 | 10631.03879547779 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3885075 | 0.42405 | 0.42991187999999997 | 10242.329941602844 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.5209315 | 0.9059562 | 0.9453950899999999 | 6779.746051052166 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.4282395 | 0.4401862 | 0.44804506 | 18604.005647059876 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.439381 | 0.56880065 | 0.57690035 | 17126.407228713964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.4199545 | 0.4881988 | 0.5138323499999999 | 18361.46592986388 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.40580700000000003 | 0.46372204999999994 | 0.48603314 | 19657.7439347874 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.549417 | 0.804661 | 0.8883975799999998 | 13876.214797560768 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.501044 | 0.50985 | 0.52864128 | 31862.26072139185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.530472 | 0.656895 | 0.6618415999999999 | 30874.344851226375 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.5032725 | 0.59063305 | 0.59588789 | 32971.7500106643 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.4729035 | 0.56462565 | 0.57041253 | 34228.441301817154 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.552828 | 0.6026374999999999 | 0.61760503 | 28490.603781066377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.6792535 | 0.6862442 | 0.7050167199999999 | 47088.39365567477 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.574925 | 0.7249801 | 0.87141993 | 52408.39675540926 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.5143425 | 0.7090107999999996 | 0.7996226099999999 | 58304.229501772505 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.482491 | 0.57594885 | 0.6016782099999999 | 64557.068105245606 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.5460320000000001 | 0.7967465499999999 | 0.8249940299999999 | 54701.47184566526 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.011349 | 1.02490055 | 1.02929924 | 63172.0631500319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.715782 | 1.0662386 | 1.07097845 | 83518.18038105221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.5735680000000001 | 0.86137265 | 0.8684057199999999 | 101857.14541721308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.51686 | 0.6866356499999997 | 0.7356858900000001 | 115715.21458717568 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.6467125 | 0.77732265 | 0.9534212099999999 | 94136.41066044253 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.6879995 | 1.6966148 | 1.69745339 | 75812.66467858221 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.069571 | 1.2372896 | 1.23839842 | 118228.80103101427 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.7360635 | 1.02210005 | 1.02832651 | 163340.68034609134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.6496999999999999 | 0.8066105 | 0.8341572899999999 | 193229.9822645068 | - |
| `full_mlp_capacity_search_hd256_depth5` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.6666814999999999 | 0.9266649499999999 | 0.9604151799999999 | 181727.06531319 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 1 | ok | 67.554385 | 0.06515599999999999 | 0.0677858 | 0.07153347 | 15240.26893588175 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 2 | ok | 67.810452 | 0.06965450000000001 | 0.07210074999999999 | 0.07579568999999999 | 14308.644911230596 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 4 | ok | 67.493991 | 0.064436 | 0.06649205 | 0.07002768 | 15496.46169290166 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 8 | ok | 67.934362 | 0.0716985 | 0.08017575 | 0.08312417 | 13686.117650794327 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 1 | 64 | ok | 71.263557 | 0.089012 | 0.09202489999999999 | 0.10082881999999997 | 11452.694074490613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 1 | ok | 68.474667 | 0.0701635 | 0.07561579999999998 | 0.08208272999999998 | 28358.81845818776 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 2 | ok | 68.767642 | 0.068185 | 0.0711627 | 0.07710534999999999 | 29162.009250772575 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 4 | ok | 68.307999 | 0.06911800000000001 | 0.0721362 | 0.07748105 | 28930.07889232514 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 8 | ok | 67.93033 | 0.0656365 | 0.06774175 | 0.07198906 | 30276.643749265797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 2 | 64 | ok | 69.574385 | 0.0950415 | 0.0989032 | 0.10340408999999999 | 21881.732735641097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 1 | ok | 68.100189 | 0.07179050000000001 | 0.07731729999999999 | 0.08020816 | 55229.395293350935 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 2 | ok | 68.809195 | 0.0772495 | 0.08723199999999999 | 0.09293973999999999 | 49976.47357506454 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 4 | ok | 68.323423 | 0.073604 | 0.0761751 | 0.08331295999999998 | 54244.79046593563 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 8 | ok | 69.115503 | 0.0753245 | 0.08000589999999999 | 0.08323478 | 52441.12107579813 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 4 | 64 | ok | 75.258766 | 0.1025475 | 0.10742094999999999 | 0.11034211 | 38822.761862925756 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 1 | ok | 68.715548 | 0.0831775 | 0.08891099999999999 | 0.09590573999999999 | 95772.82685470066 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 2 | ok | 69.227777 | 0.08499799999999999 | 0.1008761 | 0.10573397999999999 | 89195.92107052944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 4 | ok | 68.884318 | 0.0828135 | 0.08640774999999999 | 0.08827635 | 96244.0981314449 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 8 | ok | 69.208166 | 0.082872 | 0.0865478 | 0.09050272999999999 | 98231.1275288069 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 8 | 64 | ok | 69.163822 | 0.1133355 | 0.11990685000000001 | 0.12410902999999998 | 70324.1556793964 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 1 | ok | 68.883861 | 0.096253 | 0.0999027 | 0.10403075 | 165270.20231964995 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 2 | ok | 69.990426 | 0.115261 | 0.12181455 | 0.12821919999999998 | 138856.9469696641 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 4 | ok | 69.736522 | 0.10096050000000001 | 0.10366070000000001 | 0.10867711 | 161619.52460828982 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 8 | ok | 70.212577 | 0.09662950000000001 | 0.10053664999999999 | 0.10409025 | 169502.44253019683 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 16 | 64 | ok | 75.35563 | 0.11800250000000001 | 0.1231877 | 0.12520963 | 135251.78980384095 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 1 | ok | 69.715224 | 0.13163550000000002 | 0.13649995 | 0.14212644 | 241841.91640371399 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 2 | ok | 71.176739 | 0.219691 | 0.2265053 | 0.22856224 | 159687.36407860692 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 4 | ok | 69.870172 | 0.148556 | 0.17231849999999999 | 0.17434606 | 203757.77733884958 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 8 | ok | 70.128051 | 0.145652 | 0.15134894999999998 | 0.15431300999999997 | 227305.72534819687 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 32 | 64 | ok | 76.73745 | 0.2226575 | 0.2558143 | 0.30813633999999995 | 137685.96533394395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 1 | ok | 70.325465 | 0.2049225 | 0.21492409999999998 | 0.22463793999999998 | 308445.3889727689 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 2 | ok | 70.83706 | 0.25728399999999996 | 0.30266479999999985 | 0.32955711 | 254906.28888067702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 4 | ok | 70.414601 | 0.2224645 | 0.2676558 | 0.27011492 | 280127.9344276531 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 8 | ok | 70.52726 | 0.1929595 | 0.19827565 | 0.20174785 | 330904.6312171293 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 64 | 64 | ok | 70.633132 | 15.784759000000001 | 43.7124738 | 65.29018288999997 | 3556.3146153470548 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 1 | ok | 71.490105 | 0.34284499999999996 | 0.3485734 | 0.35301467000000003 | 372355.46891567815 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 2 | ok | 73.184087 | 0.3437565 | 0.4473242 | 0.44993514 | 355544.33615650353 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 4 | ok | 72.164134 | 0.33454700000000004 | 0.42418290000000003 | 0.43306539 | 374508.7352406985 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 8 | ok | 72.329674 | 0.27551 | 0.33429495 | 0.3399356 | 436732.68540854944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `fp32` | 128 | 64 | ok | 72.331916 | 5.3732215 | 10.1205824 | 15.023606789999983 | 24241.062610918725 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 1 | ok | 68.007326 | 0.1752385 | 0.1813293 | 0.18394759 | 5687.342012748063 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 2 | ok | 68.43879 | 0.1995165 | 0.20423144999999998 | 0.20968875 | 5039.005429326789 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 4 | ok | 67.675645 | 0.1950945 | 0.2042909 | 0.21643758999999996 | 5148.761094807845 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 8 | ok | 67.850118 | 0.2225915 | 0.24056804999999998 | 0.24832495999999998 | 4471.473608110752 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 1 | 64 | ok | 69.904034 | 0.429851 | 0.5299277 | 0.53802357 | 2292.6260160230713 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 1 | ok | 68.433246 | 0.206975 | 0.23057445 | 0.23231216 | 9542.226930569039 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 2 | ok | 69.238995 | 0.228369 | 0.23500465 | 0.24402913999999998 | 8899.087638838004 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 4 | ok | 68.846646 | 0.21637 | 0.2229223 | 0.23102193999999998 | 9313.5911279476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 8 | ok | 68.610962 | 0.2295035 | 0.25176845 | 0.25836986 | 8622.530119359959 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 2 | 64 | ok | 69.074005 | 0.44211849999999997 | 0.59549045 | 0.59889265 | 4430.683136461097 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 1 | ok | 68.883554 | 0.23611700000000002 | 0.24478994999999998 | 0.25176527 | 16875.01829884797 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 2 | ok | 68.495399 | 0.26814150000000003 | 0.31481155 | 0.31944552 | 14037.183938766435 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 4 | ok | 69.058501 | 0.233244 | 0.25405695 | 0.26082621 | 17026.858507146342 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 8 | ok | 68.880588 | 0.2551115 | 0.2689448 | 0.27122805 | 15778.947050350698 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 4 | 64 | ok | 75.322511 | 0.39258550000000003 | 0.4928743 | 0.5434181599999998 | 9710.099634846847 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 1 | ok | 68.435825 | 0.265245 | 0.2735661 | 0.27672364 | 30043.170533898687 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 2 | ok | 69.186006 | 0.34452950000000004 | 0.40897279999999997 | 0.41562268 | 23263.160784266285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 4 | ok | 69.121448 | 0.293366 | 0.33650985 | 0.34453077 | 27523.258357454604 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 8 | ok | 68.947272 | 0.2643645 | 0.30843875 | 0.32665934999999996 | 29487.43249313072 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 8 | 64 | ok | 75.379271 | 0.40185950000000004 | 0.54531345 | 0.55366749 | 19098.853085223953 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 1 | ok | 68.865078 | 0.35708300000000004 | 0.36893295000000004 | 0.37303708 | 44600.126140306755 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 2 | ok | 70.625792 | 0.376767 | 0.4731321 | 0.47998563 | 40512.4911151043 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 4 | ok | 69.44219 | 0.317275 | 0.3672906 | 0.37207542 | 47953.58668252172 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 8 | ok | 69.012283 | 0.3227765 | 0.33976535 | 0.35575315999999996 | 50895.53557721995 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 16 | 64 | ok | 75.740889 | 0.475273 | 0.5740326499999999 | 0.5794699000000001 | 35219.15316162558 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 1 | ok | 68.88882 | 0.5245735 | 0.53279085 | 0.5374288 | 60905.10466999025 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 2 | ok | 71.127752 | 0.4885615 | 0.6902051499999999 | 0.69347775 | 63126.839654197494 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 4 | ok | 69.629807 | 0.394395 | 0.51633435 | 0.52150202 | 75687.06351033952 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 8 | ok | 69.536229 | 0.39213299999999995 | 0.5082691500000001 | 0.5170645700000001 | 82959.62614244479 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 32 | 64 | ok | 77.198856 | 0.4313825 | 0.6013447 | 3.6188098499999883 | 55631.007040486395 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 1 | ok | 69.851857 | 0.8787785 | 0.892582 | 0.898641 | 72697.66164790787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 2 | ok | 71.846616 | 0.5811085 | 0.8896683 | 0.89585686 | 100304.2478598208 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 4 | ok | 70.480209 | 0.469426 | 0.7421686000000001 | 0.74538872 | 119781.41389232324 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 8 | ok | 70.060207 | 0.4624835 | 0.57768585 | 0.5828076600000001 | 133848.82419454693 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 64 | 64 | ok | 70.759499 | 0.5210695 | 0.6600286 | 0.66298141 | 120119.11461852252 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 1 | ok | 71.018522 | 1.563884 | 1.57496775 | 1.58180502 | 81742.8547325377 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 2 | ok | 72.604741 | 0.9464014999999999 | 0.96186685 | 1.0929303199999996 | 134266.8994770787 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 4 | ok | 72.645391 | 0.6494345 | 0.9332558999999999 | 0.9420869900000001 | 181832.9970748753 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 8 | ok | 71.838844 | 0.5622705 | 0.72685165 | 0.8417984399999996 | 218224.73902026092 | - |
| `full_mlp_capacity_search_hd256_depth5` | `jit` | `bf16` | 128 | 64 | ok | 72.427943 | 0.6082989999999999 | 0.7962993999999999 | 0.8200748099999999 | 202768.84019550082 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 1 | ok | 1375.461248 | 0.1024645 | 0.1046668 | 0.10945028 | 9730.962248148102 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 2 | ok | 1364.642704 | 0.111322 | 0.11429625 | 0.12463345999999997 | 8930.532956345769 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 4 | ok | 1383.473461 | 0.102269 | 0.10749714999999999 | 0.11209614 | 9720.434472315523 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 8 | ok | 1350.188582 | 0.0951675 | 0.097664 | 0.10373156999999998 | 10477.567527922718 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 1 | 64 | ok | 1345.638422 | 0.095532 | 0.0985894 | 0.10421544999999999 | 10418.963179800881 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 1 | ok | 1353.132633 | 0.09922500000000001 | 0.1023857 | 0.10778495999999999 | 20072.796001980783 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 2 | ok | 1375.453258 | 0.09936500000000001 | 0.10160605 | 0.10871178999999997 | 20021.3146916207 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 4 | ok | 1360.103176 | 0.09518299999999999 | 0.10004489999999999 | 0.10728481 | 20832.99045703206 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 8 | ok | 1401.473581 | 0.10260749999999999 | 0.10966055 | 0.11856202999999997 | 19323.80965815601 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 2 | 64 | ok | 1361.996533 | 0.0998575 | 0.10491985 | 0.11132861999999999 | 19909.340825616473 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 1 | ok | 1335.668964 | 0.1134965 | 0.1202567 | 0.12486517 | 35085.56492145219 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 2 | ok | 1380.20325 | 0.13507950000000002 | 0.13771535 | 0.14340669 | 29526.139417410883 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 4 | ok | 1392.402594 | 0.114679 | 0.11714509999999999 | 0.12321112999999999 | 34758.86989876653 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 8 | ok | 1371.863951 | 0.1058515 | 0.109162 | 0.11623271999999998 | 37609.60143036836 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 4 | 64 | ok | 1373.330025 | 0.15007500000000001 | 0.15469345 | 0.15530343 | 26630.513082106398 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 1 | ok | 1417.67505 | 0.135943 | 0.14121135 | 0.14784174 | 58557.38609197667 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 2 | ok | 1382.494802 | 0.16635850000000002 | 0.16979914999999998 | 0.17460857999999999 | 48608.66801920285 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 4 | ok | 1364.588726 | 0.13822099999999998 | 0.14028549999999998 | 0.14517639 | 57761.716169901476 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 8 | ok | 1346.997397 | 0.115201 | 0.1182967 | 0.12436872999999998 | 69268.02743706567 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 8 | 64 | ok | 1437.546119 | 0.1789115 | 0.18803705 | 0.19150631999999998 | 44637.979845728914 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 1 | ok | 1428.986983 | 0.184714 | 0.1892476 | 0.19416151 | 86400.54363222054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 2 | ok | 1397.521325 | 0.3496595 | 0.4847433 | 0.48912603 | 41095.42738371201 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 4 | ok | 1330.57062 | 0.25535149999999995 | 0.3188622 | 0.32212297 | 58681.93766291114 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 8 | ok | 1369.826239 | 0.2162115 | 0.21920945 | 0.22354738999999998 | 78564.79405515916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 16 | 64 | ok | 1446.424207 | 0.290406 | 0.32362035 | 0.33572918999999996 | 53255.20794328054 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 1 | ok | 1297.695496 | 0.219919 | 0.22536774999999998 | 0.2289655 | 145098.4080074733 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 2 | ok | 1389.372477 | 0.40799399999999997 | 0.5683082 | 0.57157566 | 76978.72365812132 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 4 | ok | 1365.012357 | 0.3230535 | 0.36809325 | 0.374101 | 99727.6065087721 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 8 | ok | 1431.519925 | 0.2476495 | 0.2518759 | 0.26019816 | 135314.6098507683 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 32 | 64 | ok | 1344.155891 | 0.299327 | 0.33926915 | 0.34587615 | 103929.36804253778 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 1 | ok | 1375.276147 | 0.294672 | 0.3021028 | 0.30378618999999996 | 216797.95250593702 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 2 | ok | 1362.204496 | 0.3529515 | 0.47819904999999996 | 0.482352 | 161801.90693659944 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 4 | ok | 1362.455154 | 0.36658999999999997 | 0.46840994999999996 | 0.4745889 | 161481.5039842281 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 8 | ok | 1325.608825 | 0.311713 | 0.3182395 | 0.32401532 | 218573.96471246288 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 64 | 64 | ok | 1447.105067 | 0.3604235 | 0.4026068 | 0.41236565999999997 | 186002.32526156868 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 1 | ok | 1367.061445 | 0.42539400000000005 | 0.43149805 | 0.43580038 | 300302.3011883807 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 2 | ok | 1302.570085 | 0.469472 | 0.6270308 | 0.63767164 | 265098.1032810613 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 4 | ok | 1372.834881 | 0.43835599999999997 | 0.59614975 | 0.60504307 | 294718.17597447295 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 8 | ok | 1364.381258 | 0.3458785 | 0.44043495 | 0.4439638 | 333227.27334024047 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `fp32` | 128 | 64 | ok | 1384.799813 | 0.360178 | 0.42547865 | 0.43139079999999996 | 356939.708254214 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 1 | ok | 1352.609062 | 0.27100349999999995 | 0.27924794999999997 | 0.28338742 | 3678.9268129015545 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 2 | ok | 1312.492177 | 0.303371 | 0.31503945 | 0.31815439999999995 | 3369.7490884828712 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 4 | ok | 1370.361536 | 0.29309799999999997 | 0.31168534999999997 | 0.3166413 | 3402.0761373752903 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 8 | ok | 1369.342504 | 0.3025095 | 0.3263643 | 0.34120188 | 3283.497718363106 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 1 | 64 | ok | 1390.53249 | 0.5468755 | 0.6537725999999999 | 0.66360425 | 1787.5201279235205 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 1 | ok | 1384.859954 | 0.3087855 | 0.31851175 | 0.32151549 | 6448.152533369512 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 2 | ok | 1331.407621 | 0.320917 | 0.34468624999999997 | 0.34899579999999997 | 6155.9512997767915 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 4 | ok | 1363.712101 | 0.309497 | 0.32588405 | 0.33169244000000003 | 6530.067237490318 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 8 | ok | 1364.362294 | 0.32987299999999997 | 0.36118265 | 0.36790652 | 6015.897369753416 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 2 | 64 | ok | 1344.726004 | 0.54305 | 0.6636479999999999 | 0.70306836 | 3554.7207491360696 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 1 | ok | 1362.200899 | 0.319902 | 0.3298795 | 0.33205887 | 12439.673028170322 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 2 | ok | 1364.312488 | 0.3912 | 0.42560595 | 0.43232143 | 10120.118213100846 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 4 | ok | 1341.25783 | 0.3242025 | 0.34938505 | 0.35673815 | 12235.75435475086 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 8 | ok | 1359.362487 | 0.3364505 | 0.38145305 | 0.3893354 | 11703.194580063338 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 4 | 64 | ok | 1394.628208 | 0.570071 | 0.6713226 | 0.6847836599999999 | 7193.107679311407 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 1 | ok | 1329.297938 | 0.3644585 | 0.3768735 | 0.38159535 | 21881.519667164583 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 2 | ok | 1365.41627 | 0.444681 | 0.5391232499999999 | 0.54355383 | 17802.208737840312 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 4 | ok | 1369.037081 | 0.378222 | 0.44719525 | 0.45740643 | 20258.443035334672 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 8 | ok | 1359.852375 | 0.363389 | 0.40837285 | 0.41496295 | 21981.70814138723 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 8 | 64 | ok | 1379.444993 | 0.575577 | 0.6729117499999999 | 1.684457289999996 | 12799.116963322467 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 1 | ok | 1303.390372 | 0.45844 | 0.46758755 | 0.47164881000000003 | 34822.33643948574 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 2 | ok | 1368.87957 | 0.47882749999999996 | 0.58471175 | 0.5865735400000001 | 33011.77389674858 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 4 | ok | 1376.402513 | 0.442716 | 0.5245970499999999 | 0.54812472 | 35712.48095855156 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 8 | ok | 1321.864968 | 0.424784 | 0.4873967 | 0.49904498999999997 | 38577.04704278002 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 16 | 64 | ok | 1448.344316 | 0.568048 | 0.65974625 | 0.7625842299999999 | 27495.346154894414 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 1 | ok | 1368.875628 | 0.628893 | 0.6399922 | 0.64429832 | 50737.28891385164 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 2 | ok | 1314.69564 | 0.540177 | 0.7913554999999995 | 0.86554701 | 54640.60619381324 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 4 | ok | 1375.525724 | 0.49825600000000003 | 0.61094495 | 0.61345604 | 63367.10456768308 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 8 | ok | 1393.293466 | 0.478789 | 0.59465415 | 0.6166625 | 66820.0576774033 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 32 | 64 | ok | 1405.378325 | 0.5510865 | 0.64188595 | 0.704991 | 55840.325705451774 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 1 | ok | 1370.027936 | 0.990094 | 1.00343095 | 1.00978105 | 64523.355387886695 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 2 | ok | 1306.537883 | 0.6561185 | 0.9782964999999997 | 1.0438039799999999 | 89807.65361592134 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 4 | ok | 1411.202192 | 0.5412254999999999 | 0.87031265 | 0.88217846 | 107090.36971444185 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 8 | ok | 1379.236476 | 0.5495335 | 0.66827255 | 0.68683684 | 117438.49755058157 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 64 | 64 | ok | 1452.966894 | 0.6407134999999999 | 0.7540727500000001 | 1.306181639999998 | 95937.41330967916 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 1 | ok | 1400.686231 | 1.717608 | 1.7251796 | 1.72968086 | 74514.74687008664 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 2 | ok | 1388.163133 | 1.038504 | 1.1812558999999998 | 1.2279734199999999 | 121708.64870215316 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 4 | ok | 1364.72746 | 0.662192 | 0.8224635 | 1.0738341799999997 | 178573.70558547578 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 8 | ok | 1316.98101 | 0.7228215 | 0.78635335 | 0.8819594299999997 | 173001.99522660463 | - |
| `full_mlp_capacity_search_hd256_depth5` | `compile` | `bf16` | 128 | 64 | ok | 1405.31949 | 0.703945 | 0.90000545 | 0.9206096100000001 | 170920.49956109485 | - |
