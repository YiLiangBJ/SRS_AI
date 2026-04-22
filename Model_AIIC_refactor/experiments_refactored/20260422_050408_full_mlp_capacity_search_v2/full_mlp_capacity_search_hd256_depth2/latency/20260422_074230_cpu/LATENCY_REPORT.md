# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth2

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1516874.518` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.055` ms, throughput=`17972.694` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 1 | 1 | ok | 0.07932449999999999 | 0.08360975 | 0.09094064999999998 | 13051.52480964351 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 1 | 2 | ok | 0.0551015 | 0.05878095 | 0.06546745999999998 | 17972.69444478393 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 1 | 4 | ok | 0.055668499999999996 | 0.06042185 | 0.06406492999999999 | 17727.71156871456 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 1 | 8 | ok | 0.0592265 | 0.06200715 | 0.06544306999999999 | 16753.342878271218 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 1 | 64 | ok | 0.058601 | 0.06382945 | 0.06633298 | 16717.672284807617 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 2 | 1 | ok | 0.0689145 | 0.0721007 | 0.07380021 | 28945.956163086154 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 2 | 2 | ok | 0.0682425 | 0.07158629999999999 | 0.07614911999999999 | 29053.4755365378 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 2 | 4 | ok | 0.06732199999999999 | 0.0711281 | 0.07419398999999999 | 29512.37586725806 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 2 | 8 | ok | 0.067386 | 0.07093050000000001 | 0.07519620999999999 | 29414.77539465804 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 2 | 64 | ok | 0.0702235 | 0.0734095 | 0.07748979999999998 | 28264.548257617786 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 4 | 1 | ok | 0.0688275 | 0.07261524999999999 | 0.07511216999999999 | 57746.78956723402 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 4 | 2 | ok | 0.082866 | 0.0871436 | 0.09162964999999998 | 47950.534228889475 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 4 | 4 | ok | 0.068294 | 0.0715351 | 0.07630282 | 57971.40096876009 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 4 | 8 | ok | 0.06794 | 0.07109069999999999 | 0.07596299999999999 | 58384.97240872167 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 4 | 64 | ok | 0.0718905 | 0.07509175 | 0.07930467 | 55261.06015580856 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 8 | 1 | ok | 0.07105 | 0.07419315 | 0.08005102999999998 | 111698.08454539717 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 8 | 2 | ok | 0.068906 | 0.071518 | 0.07514172 | 115469.90767603532 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 8 | 4 | ok | 0.0678995 | 0.07127815 | 0.07295390999999998 | 116976.23452331308 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 8 | 8 | ok | 0.06966 | 0.07264570000000001 | 0.08677004999999996 | 113695.37541218125 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 8 | 64 | ok | 0.07278799999999999 | 0.0769566 | 0.08067511 | 108799.92862724682 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 16 | 1 | ok | 0.07016349999999999 | 0.07359115 | 0.07858336999999999 | 226360.340762857 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 16 | 2 | ok | 0.073418 | 0.07764739999999999 | 0.08111189999999999 | 216479.85357302707 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 16 | 4 | ok | 0.07169700000000001 | 0.07467635 | 0.07910133 | 221617.75978241564 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 16 | 8 | ok | 0.073746 | 0.07689885 | 0.07965251 | 216005.9898460984 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 16 | 64 | ok | 0.06969449999999999 | 0.07344625 | 0.07698986999999999 | 227372.8488396879 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 32 | 1 | ok | 0.07141800000000001 | 0.07557415 | 0.08015452999999999 | 443559.66612155025 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 32 | 2 | ok | 0.07297200000000001 | 0.0768861 | 0.08735397999999997 | 433568.7340578808 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 32 | 4 | ok | 0.074653 | 0.0781718 | 0.08226813999999999 | 425691.5492245763 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 32 | 8 | ok | 0.0713385 | 0.07470665 | 0.07793006999999999 | 445332.51169484924 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 32 | 64 | ok | 0.0761665 | 0.0801909 | 0.08422606999999999 | 417000.15819943504 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 64 | 1 | ok | 0.074907 | 0.07766975 | 0.08904537999999997 | 846131.5131634265 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 64 | 2 | ok | 0.094359 | 0.09829790000000001 | 0.10638307999999998 | 673332.8016809754 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 64 | 4 | ok | 0.08928649999999999 | 0.0937149 | 0.10419007999999996 | 709820.499704759 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 64 | 8 | ok | 0.0762925 | 0.07999830000000001 | 0.08941294999999998 | 831724.0339655301 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 64 | 64 | ok | 0.07678299999999999 | 0.0807129 | 0.0849313 | 827039.6477638141 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 128 | 1 | ok | 0.083986 | 0.08788385 | 0.0923914 | 1516874.5179775702 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 128 | 2 | ok | 0.1194655 | 0.1257709 | 0.13370953 | 1062711.255374164 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 128 | 4 | ok | 0.117673 | 0.1232716 | 0.12421628 | 1085116.541516559 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 128 | 8 | ok | 0.120889 | 0.1259084 | 0.12856429 | 1099337.7005668632 | - |
| `full_mlp_capacity_search_hd256_depth2` | `fp32` | 128 | 64 | ok | 0.1530285 | 0.15962130000000002 | 0.16206575 | 836250.3006907819 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 1 | 1 | ok | 0.0705805 | 0.0736782 | 0.07736259 | 14074.627050532414 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 1 | 2 | ok | 0.0698375 | 0.07309955 | 0.07733134 | 14231.698889130512 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 1 | 4 | ok | 0.0703385 | 0.0737641 | 0.07556597 | 14125.622551499902 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 1 | 8 | ok | 0.071259 | 0.0748528 | 0.07791593 | 13932.326787740445 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 1 | 64 | ok | 0.0707865 | 0.07418745 | 0.07857043 | 14012.133386542077 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 2 | 1 | ok | 0.113326 | 0.11874595 | 0.12121989 | 17545.137058224416 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 2 | 2 | ok | 0.115909 | 0.12184585 | 0.12459054 | 17172.033900342445 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 2 | 4 | ok | 0.12281049999999999 | 0.1331032 | 0.13821133 | 16202.922812840363 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 2 | 8 | ok | 0.13135 | 0.14322975 | 0.14445401 | 15162.372329034399 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 2 | 64 | ok | 0.237915 | 0.256173 | 0.26088268000000003 | 8410.514522477479 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 4 | 1 | ok | 0.1201265 | 0.13008945 | 0.13594946 | 32958.34970951335 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 4 | 2 | ok | 0.1162675 | 0.13358759999999997 | 0.13571012 | 33527.59220255494 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 4 | 4 | ok | 0.130184 | 0.1412688 | 0.14349582 | 30615.339234031766 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 4 | 8 | ok | 0.130496 | 0.14136805 | 0.1435446 | 30528.65556990742 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 4 | 64 | ok | 0.241742 | 0.2603274 | 0.26372425 | 16626.492903189333 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 8 | 1 | ok | 0.121016 | 0.12782380000000002 | 0.13507033 | 65394.685537393096 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 8 | 2 | ok | 0.12211 | 0.13754795 | 0.14182915 | 63959.84345188717 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 8 | 4 | ok | 0.1346545 | 0.15995315 | 0.16343750999999998 | 57790.79097966659 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 8 | 8 | ok | 0.142992 | 0.15615185 | 0.16175537999999998 | 55561.14253711068 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 8 | 64 | ok | 0.25677099999999997 | 0.2796014 | 0.28426851000000003 | 30965.852019612223 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 16 | 1 | ok | 0.117145 | 0.12150895 | 0.12610408999999997 | 135967.70766942852 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 16 | 2 | ok | 0.129191 | 0.14885925 | 0.15158958 | 120818.65512526328 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 16 | 4 | ok | 0.131262 | 0.162882 | 0.16540957 | 116888.2136646413 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 16 | 8 | ok | 0.14127699999999999 | 0.20274484999999998 | 0.22038473 | 106708.66719807583 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 16 | 64 | ok | 0.2439335 | 0.25588875 | 0.26068353 | 65340.83576809623 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 32 | 1 | ok | 0.1222635 | 0.128181 | 0.13356041 | 259760.66950714955 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 32 | 2 | ok | 0.147091 | 0.15576805000000002 | 0.16339014999999998 | 216147.019960772 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 32 | 4 | ok | 0.142443 | 0.1698059 | 0.17693718 | 218004.70917422406 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 32 | 8 | ok | 0.1549925 | 0.22805119999999998 | 0.24240543 | 192856.61508435244 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 32 | 64 | ok | 0.244344 | 0.26499835 | 0.3553297899999998 | 128912.17221763823 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 64 | 1 | ok | 0.143023 | 0.1585017 | 0.16522491 | 441695.80823776487 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 64 | 2 | ok | 0.1688845 | 0.18773415 | 0.19086975999999997 | 371033.80802084657 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 64 | 4 | ok | 0.15908250000000002 | 0.1722892 | 0.17537225 | 402409.4770449915 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 64 | 8 | ok | 0.16667300000000002 | 0.2326122 | 0.2512291 | 366391.75453676004 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 64 | 64 | ok | 0.255683 | 0.2694962 | 0.28508125999999995 | 253219.6081869698 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 128 | 1 | ok | 0.159281 | 0.17858825 | 0.19163321 | 789901.4067280099 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 128 | 2 | ok | 0.2118865 | 0.23997015 | 0.24650825999999998 | 603080.7816341547 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 128 | 4 | ok | 0.209011 | 0.2434784 | 0.26657933999999994 | 594131.5032102596 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 128 | 8 | ok | 0.2033435 | 0.22418199999999996 | 0.24076865 | 632585.5908072663 | - |
| `full_mlp_capacity_search_hd256_depth2` | `bf16` | 128 | 64 | ok | 0.287964 | 0.9093909499999996 | 1.0536701899999996 | 330821.7244630311 | - |
