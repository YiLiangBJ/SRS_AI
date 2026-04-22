# Latency Report

- Device: `cpu`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth3

- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`561388.686` samples/s, p50=`0.227` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.074` ms, throughput=`12268.918` samples/s

## Run References

### full_mlp_capacity_search_hd256_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd256_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,408`
- MACs / sample: `43,008`
- FLOPs / sample estimate: `86,488`

## Results

| Run | Precision | Batch | Threads | Status | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 1 | 1 | ok | 0.078584 | 0.0809251 | 0.08543419 | 12654.995217677308 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 1 | 2 | ok | 0.0843875 | 0.08669 | 0.09013101 | 11811.816966919117 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 1 | 4 | ok | 0.08245949999999999 | 0.1119815 | 0.11310938 | 10653.484754863315 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 1 | 8 | ok | 0.0762135 | 0.07878885 | 0.08174268999999999 | 13078.484245527095 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 1 | 64 | ok | 0.0743825 | 0.0931336 | 0.10117594999999997 | 12268.917996760514 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 2 | 1 | ok | 0.0922505 | 0.0952995 | 0.10129735 | 21560.185581453414 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 2 | 2 | ok | 0.08830299999999999 | 0.09216725 | 0.09827072999999999 | 22457.643200600425 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 2 | 4 | ok | 0.086462 | 0.09149909999999999 | 0.09718661999999999 | 22924.680275215374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 2 | 8 | ok | 0.08827099999999999 | 0.0922162 | 0.10124556999999998 | 22474.748496326953 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 2 | 64 | ok | 0.0893545 | 0.10992115 | 0.12119306999999997 | 20638.59528767209 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 4 | 1 | ok | 0.091877 | 0.09629455 | 0.1230279499999999 | 42785.28802200191 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 4 | 2 | ok | 0.096774 | 0.09980375000000001 | 0.10448135999999998 | 41194.12711807328 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 4 | 4 | ok | 0.0930555 | 0.09613725 | 0.10240537999999999 | 42766.55108293461 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 4 | 8 | ok | 0.095763 | 0.10046404999999999 | 0.10766084999999999 | 41428.55604082204 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 4 | 64 | ok | 0.094329 | 0.1164748 | 0.11786796000000001 | 39513.15837565357 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 8 | 1 | ok | 0.10272300000000001 | 0.10887615 | 0.11408860999999998 | 77232.51195269663 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 8 | 2 | ok | 0.10775299999999999 | 0.11033535 | 0.11378456 | 73949.26414935976 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 8 | 4 | ok | 0.10086 | 0.1052847 | 0.10773813 | 78980.78456747062 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 8 | 8 | ok | 0.0979365 | 0.10279835 | 0.10775425999999999 | 81108.46066685754 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 8 | 64 | ok | 0.135992 | 0.13905135 | 0.1442713 | 58698.90645404749 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 16 | 1 | ok | 0.1387425 | 0.145882 | 0.15078139 | 114665.8479525123 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 16 | 2 | ok | 0.1729425 | 0.17575539999999998 | 0.17970911 | 95118.18970110179 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 16 | 4 | ok | 0.151022 | 0.15764955 | 0.16319002 | 105299.88815835628 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 16 | 8 | ok | 0.1377355 | 0.14311 | 0.14604946 | 115656.67547199488 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 16 | 64 | ok | 0.16009600000000002 | 0.16508055 | 0.16787744 | 99837.32755441601 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 32 | 1 | ok | 0.14978950000000002 | 0.1559069 | 0.16217320999999998 | 212212.56802407556 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 32 | 2 | ok | 0.213756 | 0.22136255 | 0.22861478 | 148964.00191791152 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 32 | 4 | ok | 0.17436200000000002 | 0.17906080000000002 | 0.18518226999999998 | 186948.5383253267 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 32 | 8 | ok | 0.15413700000000002 | 0.1584864 | 0.17006540999999997 | 206753.92716162524 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 32 | 64 | ok | 0.18361 | 0.19063125 | 0.19365865 | 175363.18263132893 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 64 | 1 | ok | 0.18060700000000002 | 0.19684985 | 0.20207709999999998 | 344936.7170475487 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 64 | 2 | ok | 0.2170365 | 0.2348678 | 0.23995586 | 288970.9365674479 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 64 | 4 | ok | 0.2094505 | 0.22011999999999998 | 0.22828009 | 313227.1519390523 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 64 | 8 | ok | 0.1767985 | 0.18124505 | 0.18466412 | 369008.58348559745 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 64 | 64 | ok | 0.22716999999999998 | 0.23441115 | 0.24214108999999998 | 285077.0875170623 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 128 | 1 | ok | 0.22710249999999998 | 0.23486465 | 0.23949578 | 561388.6861408309 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 128 | 2 | ok | 0.3071485 | 0.3561181 | 0.36015001 | 424633.915802791 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 128 | 4 | ok | 0.2486935 | 0.29324645 | 0.29701332 | 490715.35871599417 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 128 | 8 | ok | 0.2372505 | 0.244844 | 0.25118149999999995 | 539539.3396257671 | - |
| `full_mlp_capacity_search_hd256_depth3` | `fp32` | 128 | 64 | ok | 0.2755575 | 0.29649485 | 0.30019999 | 463836.178859579 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 1 | 1 | ok | 0.173829 | 0.17846145 | 0.18491486 | 5737.669175175631 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 1 | 2 | ok | 0.19417 | 0.20150585 | 0.20639341 | 5123.942533549783 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 1 | 4 | ok | 0.1954225 | 0.20795025 | 0.21576648 | 5118.702191961364 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 1 | 8 | ok | 0.2111045 | 0.2244382 | 0.23227293 | 4711.342632769875 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 1 | 64 | ok | 0.43981000000000003 | 0.6632658499999999 | 0.68339376 | 2165.189495002894 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 2 | 1 | ok | 0.19782899999999998 | 0.20743815 | 0.21063621 | 10045.131772540848 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 2 | 2 | ok | 0.196697 | 0.205822 | 0.20843426 | 10086.430624017203 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 2 | 4 | ok | 0.208397 | 0.2226857 | 0.22379608 | 9566.419266309214 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 2 | 8 | ok | 0.215532 | 0.22863475 | 0.22941858 | 9282.284481598335 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 2 | 64 | ok | 0.4029355 | 0.5732404 | 0.6084869099999999 | 4508.2898433639775 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 4 | 1 | ok | 0.194266 | 0.20386674999999999 | 0.21666372999999997 | 20462.516302230957 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 4 | 2 | ok | 0.22708 | 0.23397035 | 0.23737788 | 17846.632465380208 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 4 | 4 | ok | 0.20580749999999998 | 0.21674735 | 0.21876109 | 19309.39001154219 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 4 | 8 | ok | 0.223689 | 0.24044835 | 0.25245573 | 17663.540985332373 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 4 | 64 | ok | 0.39655450000000003 | 0.5116243 | 0.5269107599999999 | 9478.169429955877 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 8 | 1 | ok | 0.206661 | 0.21187405 | 0.21519454999999998 | 38599.58021026543 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 8 | 2 | ok | 0.2574195 | 0.26273850000000004 | 0.26757823999999997 | 31420.216568555727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 8 | 4 | ok | 0.234364 | 0.24405795 | 0.24644818999999998 | 34054.80712549162 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 8 | 8 | ok | 0.22980650000000002 | 0.244517 | 0.25091746 | 35020.86324151534 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 8 | 64 | ok | 0.5231155000000001 | 0.776646 | 0.77946443 | 15608.79368218467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 16 | 1 | ok | 0.23403849999999998 | 0.24179435 | 0.2469911 | 68174.47005853972 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 16 | 2 | ok | 0.2838385 | 0.29926929999999996 | 0.32190842999999997 | 56155.16401835403 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 16 | 4 | ok | 0.261944 | 0.27385085 | 0.27618973 | 61911.83754333828 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 16 | 8 | ok | 0.2663455 | 0.280283 | 0.28214594 | 60750.208885796375 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 16 | 64 | ok | 0.40969350000000004 | 0.6195129499999998 | 0.7418948899999999 | 34413.34305108785 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 32 | 1 | ok | 0.27664750000000005 | 0.2889674 | 0.29269675 | 115027.83278145136 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 32 | 2 | ok | 0.3468205 | 0.39098964999999997 | 0.3967616 | 94937.55571276952 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 32 | 4 | ok | 0.29175399999999996 | 0.3326968 | 0.34409829999999997 | 106168.66484941303 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 32 | 8 | ok | 0.28313299999999997 | 0.29881235 | 0.30058907 | 114933.99125381061 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 32 | 64 | ok | 0.45131 | 0.5181175 | 0.52658606 | 71571.95177517676 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 64 | 1 | ok | 0.36557249999999997 | 0.3745326 | 0.38001608 | 174396.32982923873 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 64 | 2 | ok | 0.4433205 | 0.57497175 | 0.58015574 | 147366.32205205763 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 64 | 4 | ok | 0.360207 | 0.4299321 | 0.44386188 | 173686.4487390228 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 64 | 8 | ok | 0.31493899999999997 | 0.3697289 | 0.3721639 | 197308.94035788512 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 64 | 64 | ok | 0.4679085 | 0.5350906 | 0.54517228 | 136421.2161934371 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 128 | 1 | ok | 0.559833 | 0.56867805 | 0.58189509 | 228009.50728392403 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 128 | 2 | ok | 0.5617835 | 0.7065575 | 0.71155889 | 235880.20190018462 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 128 | 4 | ok | 0.487446 | 0.6328302 | 0.6407356999999999 | 270933.3934805467 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 128 | 8 | ok | 0.410565 | 0.49560055 | 0.49943469999999995 | 317301.0845698118 | - |
| `full_mlp_capacity_search_hd256_depth3` | `bf16` | 128 | 64 | ok | 0.4738685 | 0.61622015 | 0.6265544 | 265165.4105974516 | - |
