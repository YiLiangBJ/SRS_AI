# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 128]`

## Hardware Summary

- Hostname: `sh14l07002s1404`
- CPU model: `Intel(R) Xeon(R) 6760P`
- CPU capability: `AVX512`
- CPU flag summary: `['avx2', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'amx_bf16', 'amx_int8', 'amx_tile', 'fma']`
- Logical CPU count: `256`
- Physical CPU count: `128`
- mkldnn available: `True`
- mkldnn enabled: `True`
- oneDNN version: `None`
- torch.compile available: `True`
- Python: `3.11.9`
- PyTorch: `2.1.2+cu121`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd256_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1075879.784` samples/s, p50=`0.113` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.037` ms, throughput=`25988.255` samples/s

### full_mlp_capacity_search_hd256_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1493106.630` samples/s, p50=`0.085` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.021` ms, throughput=`46973.895` samples/s

### full_mlp_capacity_search_hd256_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`1297855.112` samples/s, p50=`0.097` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.035` ms, throughput=`28036.084` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0372335 | 0.042838 | 0.04522069 | 25988.255387625228 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.040774000000000005 | 0.04596935 | 0.04837581999999999 | 24059.027380135518 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.039421 | 0.04553595 | 0.050180949999999995 | 24491.567308460024 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0416865 | 0.04843785 | 0.06639415999999994 | 22855.983079258607 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.062607 | 0.17631985 | 0.18467798 | 10080.44192657406 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0386305 | 0.04427575 | 0.04655971 | 49771.10269868874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0393095 | 0.045353700000000004 | 0.04618513 | 48779.53600905349 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.038893 | 0.04627095 | 0.06921319999999992 | 48299.028948023 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.039224999999999996 | 0.04499565 | 0.04851074 | 49154.80765969537 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0564205 | 0.07326305 | 0.7439986299999974 | 23734.280489353394 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.040716 | 0.04625625 | 0.048086899999999995 | 95691.76534082535 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.044204 | 0.049033049999999995 | 0.051592799999999994 | 90551.45382622903 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.045679 | 0.05212195 | 0.05910935 | 84570.74427329205 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.041401 | 0.0456226 | 0.047787199999999995 | 95193.3090526933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0485925 | 0.06061179999999998 | 0.07241092999999998 | 79377.89952583612 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.046459 | 0.052468299999999995 | 0.055252199999999994 | 167929.49648019776 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0564165 | 0.0649228 | 0.06688999999999999 | 137979.37484304846 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.051726999999999995 | 0.059119399999999996 | 0.06517103999999999 | 150724.9303933431 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.047338000000000005 | 0.054347099999999995 | 0.0828561899999999 | 160152.91400228938 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.127551 | 0.13642585 | 0.16337965999999998 | 62602.38424310509 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.052144499999999996 | 0.06473314999999999 | 0.1445507299999997 | 278840.4697904235 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.06414349999999999 | 0.07066005 | 0.07354 | 245363.92530754063 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0611695 | 0.0702434 | 0.07395858999999999 | 259127.35623695256 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0546985 | 0.06234075 | 0.06656543999999999 | 284398.2328915799 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.14826699999999998 | 0.16524919999999998 | 0.17132238 | 107319.26798063908 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0604575 | 0.07905115 | 0.10236938999999992 | 476335.3615296081 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.086032 | 0.09293879999999999 | 0.0945398 | 368189.5292420726 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0763315 | 0.08596804999999999 | 0.08924546 | 409025.4533983241 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.06890550000000001 | 0.08394319999999998 | 0.14120432999999993 | 444649.8479714379 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.14747549999999998 | 0.16939 | 0.17516695 | 213059.8780818113 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.082122 | 0.10504024999999997 | 0.18099796999999998 | 722931.6867961724 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.102607 | 0.1168139 | 0.11967405 | 613713.779619601 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.104681 | 0.11954944999999999 | 0.1550333499999999 | 593477.0932678926 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.08857999999999999 | 0.12365424999999995 | 0.13353533 | 684793.5807449741 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.2373535 | 0.2536318 | 0.26033879 | 267586.6518746996 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.1132525 | 0.14282345 | 0.14609386 | 1075879.7838826485 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.143486 | 0.17434334999999998 | 0.23343716999999997 | 837707.7384299473 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.148509 | 0.1705286 | 0.23427538999999975 | 836075.8362587278 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.128407 | 0.14694664999999998 | 0.17662348999999988 | 978019.4708395092 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.4744625 | 0.5057685 | 0.51355838 | 268270.0159507483 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.086147 | 0.10208349999999998 | 0.1317293099999999 | 11214.091358062347 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.09623000000000001 | 0.1083316 | 0.11042554 | 10405.32486256647 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.09198200000000001 | 0.10565309999999999 | 0.14235921999999993 | 10623.707227626739 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0944915 | 0.10323514999999998 | 0.10671710000000001 | 10700.554609745424 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.3171505 | 0.3391271 | 0.34951853 | 3148.6304716900336 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.084419 | 0.10753795 | 0.17530806999999995 | 22438.680695329836 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.09868299999999999 | 0.11543535 | 0.1163515 | 19877.02482282614 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.108592 | 0.1311865 | 0.1694014999999999 | 17545.309445745428 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.096547 | 0.11097425 | 0.11533465999999999 | 20475.020475020476 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.3685575 | 0.40351889999999996 | 0.41080152 | 5463.9305997762085 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.084224 | 0.09672835 | 0.09886655 | 45949.81131858727 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1022775 | 0.1174165 | 0.11991756 | 38656.65047081868 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0940915 | 0.12052945 | 0.1481658499999999 | 39583.847098682134 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.1041695 | 0.1278423 | 0.16832515999999986 | 36433.48929902878 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.2523175 | 0.2638525 | 0.26877498 | 15837.927322127209 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0860065 | 0.10209339999999999 | 0.19176336999999982 | 86911.72441335129 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.10925750000000001 | 0.1244604 | 0.13223884 | 71610.48764235717 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1016965 | 0.12110334999999998 | 0.15429540999999988 | 74922.82480777144 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1095235 | 0.13100284999999998 | 0.16331129999999988 | 70065.16585913642 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.36748400000000003 | 0.4011421 | 0.41549942 | 21598.764291497355 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0969785 | 0.1125692 | 0.1632892999999998 | 157154.89913700352 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.11111399999999999 | 0.12992084999999998 | 0.1629802899999999 | 139615.51980079658 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1018225 | 0.12263625 | 0.12476944000000001 | 154528.24747389776 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.112238 | 0.12257425 | 0.12812741999999996 | 141441.39031229023 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.248203 | 0.26083515 | 0.26580368000000004 | 64015.117810222066 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.0989535 | 0.11631675 | 0.11819645999999999 | 311156.49674345553 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.113691 | 0.12785815 | 0.1786594399999999 | 274751.3628955497 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.122934 | 0.14237524999999998 | 0.14815753 | 256957.81539604504 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1231275 | 0.1439018 | 0.18945903999999986 | 252278.70742481464 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.449266 | 0.480456 | 0.50639947 | 70601.7555303017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.099289 | 0.12515584999999996 | 0.22101260999999986 | 600671.8514658646 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.14273750000000002 | 0.160774 | 0.2441520299999997 | 436362.64463035314 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.133754 | 0.1500665 | 0.15277291999999998 | 476075.7787669905 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.130053 | 0.14266225 | 0.14983769 | 488369.9280112203 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.310845 | 0.3279629 | 0.3641483799999999 | 203566.67906090873 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.123838 | 0.1422153 | 0.17790378999999987 | 1007724.3647124529 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.19462649999999998 | 0.22292375 | 0.30015827999999983 | 643474.828420956 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1835505 | 0.22034504999999996 | 0.2499254099999999 | 678459.4221900331 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.197845 | 0.20605555 | 0.20776451 | 652235.746941906 | - |
| `full_mlp_capacity_search_hd256_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.4745855 | 0.50159405 | 0.5117567399999999 | 269619.0514506874 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 1 | ok | 47.221593 | 0.021717 | 0.022993299999999998 | 0.02610858 | 45564.81230031222 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.104867 | 0.0257715 | 0.02664585 | 0.031330939999999995 | 38710.686394664735 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 4 | ok | 45.370327 | 0.024092000000000002 | 0.02720635 | 0.029670029999999997 | 40442.178603984525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 8 | ok | 45.879814 | 0.024207 | 0.025390549999999998 | 0.029121709999999988 | 41099.02069253494 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 1 | 128 | ok | 55.039832 | 0.0214035 | 0.021760349999999998 | 0.025036759999999998 | 46973.89472774401 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.314158 | 0.022952 | 0.02371395 | 0.027329839999999998 | 86439.68757239325 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 2 | ok | 45.798803 | 0.022796999999999998 | 0.02781165 | 0.03040696999999999 | 81594.28709439481 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.817628 | 0.022491999999999998 | 0.027338650000000003 | 0.02823843 | 84863.56484679581 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 8 | ok | 46.826646 | 0.022704000000000002 | 0.02320395 | 0.02665622999999999 | 87730.6108770166 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 2 | 128 | ok | 51.26952 | 0.0231115 | 0.02428 | 0.030349819999999996 | 85843.83632670791 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 1 | ok | 45.814447 | 0.023285 | 0.025441549999999997 | 0.03136489999999999 | 168599.80391842802 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.803639 | 0.023855 | 0.02922665 | 0.029973419999999997 | 156337.81788363933 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 4 | ok | 47.321865 | 0.0283015 | 0.0291044 | 0.030271529999999998 | 141380.42432506755 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 8 | ok | 46.816322 | 0.0235415 | 0.023907 | 0.026807089999999992 | 170157.0123831766 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 4 | 128 | ok | 53.352152 | 0.0236865 | 0.024645649999999998 | 0.026848309999999993 | 169960.6456125084 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.455742 | 0.025528000000000002 | 0.026260449999999998 | 0.02949095999999999 | 313758.7117067297 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 2 | ok | 47.606195 | 0.030986 | 0.0324105 | 0.03574615 | 256650.45491293137 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 4 | ok | 47.072977 | 0.0309085 | 0.0315003 | 0.036334929999999994 | 257857.06596048214 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 8 | ok | 48.227874 | 0.030899 | 0.03295035 | 0.034819369999999995 | 256944.23924592006 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 8 | 128 | ok | 55.274126 | 0.10391 | 0.12107180000000001 | 0.12766582999999998 | 77095.08792502039 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 1 | ok | 48.064703 | 0.029642 | 0.030270949999999998 | 0.031097299999999998 | 545149.6333527997 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 2 | ok | 47.02873 | 0.036953 | 0.03936025 | 0.04187398 | 430696.8405694889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 4 | ok | 47.613139 | 0.0372785 | 0.0398008 | 0.04130169 | 430725.82685898576 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.314106 | 0.0354155 | 0.039583799999999995 | 0.04233529999999999 | 450238.26045945694 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 16 | 128 | ok | 60.014734 | 0.106562 | 0.1245448 | 0.13152288999999998 | 148499.1010235672 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 1 | ok | 46.596018 | 0.037476999999999996 | 0.0379863 | 0.04102918999999999 | 852409.5486917645 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 2 | ok | 47.694381 | 0.0599585 | 0.06575704999999998 | 0.06874239 | 532212.1397589078 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 4 | ok | 45.815109 | 0.058581999999999995 | 0.06189255 | 0.06586103999999998 | 549720.8792235743 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.556614 | 0.052624500000000005 | 0.05595385 | 0.05709861 | 604676.720928844 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 32 | 128 | ok | 59.672134 | 0.13651449999999998 | 0.1784159 | 0.31341479999999977 | 217583.67078067525 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 1 | ok | 46.837196 | 0.0534635 | 0.0589587 | 0.06075 | 1181407.3071518706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.577697 | 0.088199 | 0.09255899999999999 | 0.0936012 | 744568.02722513 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.068874 | 0.08407300000000001 | 0.08878454999999999 | 0.09294083 | 757265.0709356223 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 8 | ok | 48.053176 | 0.072904 | 0.08178239999999999 | 0.08713314 | 870609.9493305009 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 64 | 128 | ok | 56.795592 | 33.985655 | 47.858607549999974 | 53.83508509999999 | 1879.191855280889 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 1 | ok | 48.823215 | 0.08499000000000001 | 0.08997295 | 0.09103838 | 1493106.6299766817 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.214926 | 0.122753 | 0.12812500000000002 | 0.13080265 | 1038145.5217240872 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 4 | ok | 48.031791 | 0.11281050000000001 | 0.1168766 | 0.12030381999999999 | 1131908.3550400343 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 8 | ok | 49.265077 | 0.1286735 | 0.13681315 | 0.14801975 | 989602.3715820835 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `fp32` | 128 | 128 | ok | 77.387845 | 26.838806499999997 | 43.27217259999999 | 49.96520785999999 | 4414.151048478918 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.843669 | 0.048804 | 0.0551779 | 0.0568644 | 19902.96112274992 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 2 | ok | 46.002655 | 0.054861 | 0.060956699999999996 | 0.06695798999999998 | 17929.179740026895 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 4 | ok | 47.192389 | 0.058111499999999996 | 0.06340345 | 0.06605431999999999 | 17334.976226813604 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 8 | ok | 46.743079 | 0.059053 | 0.0627325 | 0.06850708999999999 | 16925.69181226429 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 1 | 128 | ok | 54.636417 | 0.1970545 | 0.22714449999999997 | 0.23475264 | 5034.873548143863 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 1 | ok | 48.192634 | 0.0584185 | 0.06248165 | 0.06724999 | 34343.0416533015 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 2 | ok | 47.378572 | 0.068384 | 0.07399385 | 0.07744237999999999 | 29603.29804422851 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 4 | ok | 47.203769 | 0.0662945 | 0.0704249 | 0.07326574999999999 | 30059.358214666503 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 8 | ok | 45.934962 | 0.064856 | 0.07187545 | 0.07802825 | 31045.420381380565 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 2 | 128 | ok | 54.954269 | 0.2115315 | 0.24886684999999997 | 0.25617741 | 9346.486207203374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 1 | ok | 47.251099 | 0.05514 | 0.06342485 | 0.06954128 | 70434.01794024871 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 2 | ok | 47.629667 | 0.069306 | 0.07395394999999999 | 0.080785 | 57902.303758959664 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.502327 | 0.0685 | 0.07232904999999999 | 0.07693388999999999 | 58704.87649668083 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 8 | ok | 46.852547 | 0.068878 | 0.07582264999999999 | 0.07997549999999999 | 57475.376830123765 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 4 | 128 | ok | 54.293313 | 0.21070650000000002 | 0.234138 | 0.24092572999999998 | 18886.11577755555 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 1 | ok | 45.710469 | 0.0567535 | 0.06481795 | 0.06768487999999999 | 138233.38426081598 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 2 | ok | 46.09139 | 0.06663150000000001 | 0.0725305 | 0.07376904000000001 | 120159.70426293589 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 4 | ok | 47.443616 | 0.074587 | 0.07914265 | 0.08260768 | 108587.89049562771 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 8 | ok | 46.117935 | 0.071498 | 0.0771449 | 0.08050592 | 112514.09942308396 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 8 | 128 | ok | 66.867427 | 0.210397 | 0.24125244999999998 | 0.25195555 | 37422.81777726051 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 1 | ok | 47.71242 | 0.059417 | 0.06569245 | 0.07283379999999999 | 264455.55378149956 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 2 | ok | 46.353886 | 0.07470299999999999 | 0.08162855 | 0.08346798 | 214731.20484605385 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.444091 | 0.0772835 | 0.0847958 | 0.08808904000000001 | 206516.36577662418 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 8 | ok | 45.922835 | 0.0762555 | 0.08125589999999999 | 0.08327737 | 210418.33795815308 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 16 | 128 | ok | 78.065831 | 0.21090150000000002 | 0.22782634999999998 | 0.23734432 | 75561.09544320313 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 1 | ok | 45.928706 | 0.0620965 | 0.0690513 | 0.07764141999999999 | 504499.18681037327 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 2 | ok | 47.651092 | 0.08483399999999999 | 0.08996485 | 0.09921474999999999 | 377396.23018905666 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.523793 | 0.08284849999999999 | 0.09178085 | 0.09523245 | 384057.6700997422 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 8 | ok | 46.168942 | 0.083626 | 0.0909243 | 0.09574006999999998 | 381033.75887659815 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 32 | 128 | ok | 58.496019 | 0.21228950000000002 | 0.24410685 | 0.25912992999999995 | 149028.35376201587 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 1 | ok | 50.062377 | 0.071701 | 0.0766524 | 0.07971104999999999 | 904622.280620616 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 2 | ok | 49.178164 | 0.098524 | 0.10783195 | 0.11552681999999997 | 643006.8285315793 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 4 | ok | 48.168474 | 0.10718050000000001 | 0.11741044999999999 | 0.11972605 | 590297.210956359 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 8 | ok | 48.142179 | 0.101228 | 0.10776764999999999 | 0.11110576999999999 | 633482.8550834723 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 64 | 128 | ok | 69.15604 | 0.272577 | 0.3114754 | 0.3562742699999999 | 230434.17615353368 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 1 | ok | 50.269563 | 0.09386 | 0.10457255 | 0.10677439 | 1354966.5026874915 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.615425 | 0.1270595 | 0.13665645 | 0.14090849 | 1003736.7236214068 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 4 | ok | 50.951025 | 0.142454 | 0.151935 | 0.15645949 | 888155.2972843125 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.818775 | 0.1406695 | 0.14844905 | 0.15046784 | 910329.0341622306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `jit` | `bf16` | 128 | 128 | ok | 73.779269 | 0.30344899999999997 | 0.32866965 | 0.34008097 | 416874.479680398 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 1 | ok | 490.295555 | 0.035321500000000006 | 0.03922555 | 0.04054449 | 28036.08356098633 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 2 | ok | 501.562335 | 0.0436155 | 0.04874 | 0.05402846999999999 | 22747.497320344817 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 4 | ok | 507.116141 | 0.037403 | 0.0431837 | 0.044755439999999994 | 26485.46371809296 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 8 | ok | 504.6677 | 0.0393795 | 0.042226599999999996 | 0.04715749999999999 | 25129.82065349596 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 1 | 128 | ok | 646.678505 | 0.0355025 | 0.03951395 | 0.04173241 | 27625.39304027948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 1 | ok | 503.786011 | 0.0369425 | 0.04297935 | 0.04654811 | 52327.747520711324 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 2 | ok | 504.463311 | 0.0355675 | 0.04090335 | 0.04409302999999999 | 54802.23790418706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 4 | ok | 508.342181 | 0.0362255 | 0.042138800000000004 | 0.04349853 | 53877.01708817351 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 8 | ok | 512.818572 | 0.038245 | 0.04231405 | 0.04625364999999999 | 51887.35048659957 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 2 | 128 | ok | 543.753793 | 0.036581 | 0.04220525 | 0.044598389999999995 | 53481.05513843303 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 1 | ok | 496.207045 | 0.038548 | 0.0451748 | 0.04718649 | 100084.27095614508 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 2 | ok | 500.630495 | 0.04195 | 0.04677324999999999 | 0.05093978 | 94330.63453387932 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 4 | ok | 505.725401 | 0.0456325 | 0.04939775 | 0.053646239999999984 | 87006.0773745046 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 8 | ok | 501.059886 | 0.039897 | 0.045664949999999996 | 0.04955161999999999 | 97739.24245245352 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 4 | 128 | ok | 538.48335 | 0.0395655 | 0.0454531 | 0.04672739 | 98437.79223719571 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 1 | ok | 492.544496 | 0.0417025 | 0.04451115 | 0.04704783 | 191287.98022847436 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 2 | ok | 508.904448 | 0.0539445 | 0.060356749999999994 | 0.061604809999999996 | 146640.4131447 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 4 | ok | 499.353301 | 0.053288 | 0.05885779999999999 | 0.06319164 | 148037.39424578648 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 8 | ok | 509.742873 | 0.051152 | 0.0546618 | 0.05898864 | 155342.58087717294 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 8 | 128 | ok | 544.728166 | 0.12174399999999999 | 0.14374415 | 0.15089452 | 64557.4770550622 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 1 | ok | 495.104256 | 0.047964 | 0.051267 | 0.057869369999999996 | 330297.35018950806 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.691163 | 0.0662875 | 0.06988105 | 0.07193339 | 240722.5528145281 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 4 | ok | 501.619091 | 0.061949 | 0.06624129999999999 | 0.06706103000000001 | 257661.73265920434 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 8 | ok | 506.745872 | 0.0642075 | 0.06967155 | 0.07053001 | 247181.66554713968 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 16 | 128 | ok | 549.472084 | 0.0990925 | 0.11501579999999999 | 0.12294118 | 158312.01398211706 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 1 | ok | 494.091209 | 0.057070499999999996 | 0.0601382 | 0.0649841 | 555949.1598392056 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 2 | ok | 502.133539 | 0.0994705 | 0.10743475 | 0.1086356 | 321152.16550898 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 4 | ok | 504.077705 | 0.0917745 | 0.09692405 | 0.09861711 | 347431.29926060105 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 8 | ok | 510.975963 | 0.0675615 | 0.072623 | 0.07548518 | 471802.8501020421 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 32 | 128 | ok | 616.760579 | 0.140919 | 0.15642894999999998 | 0.16000841999999998 | 225715.13254909552 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 1 | ok | 493.879799 | 0.075954 | 0.0781975 | 0.0796704 | 844322.7212521306 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 2 | ok | 501.613931 | 0.15400550000000002 | 0.17657764999999997 | 0.18009482999999998 | 412013.11798266025 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 4 | ok | 504.898675 | 0.12439 | 0.13315215 | 0.13571397 | 520355.66309572593 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 8 | ok | 509.64231 | 0.100606 | 0.10917745 | 0.11059864 | 633470.0638874356 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 64 | 128 | ok | 557.608969 | 0.2589955 | 0.29511099999999996 | 0.36431462999999986 | 241051.14570479625 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 1 | ok | 498.103673 | 0.10924600000000001 | 0.11474899999999999 | 0.11808331 | 1162612.4191711883 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 2 | ok | 499.088113 | 0.165355 | 0.2796107 | 0.28346002 | 653518.6875710958 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 4 | ok | 509.018987 | 0.1812375 | 0.2306076 | 0.23730865 | 690727.586537374 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 8 | ok | 509.597011 | 0.1424185 | 0.1482712 | 0.15067349 | 910475.504367864 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `fp32` | 128 | 128 | ok | 547.010118 | 0.23890050000000002 | 11.995780400000001 | 12.00130236 | 87821.58704283599 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 1 | ok | 497.294149 | 0.08238999999999999 | 0.09041039999999999 | 0.09669290999999999 | 11950.943766029202 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 2 | ok | 508.067244 | 0.09574450000000001 | 0.10915825 | 0.10985497999999999 | 10277.498629495558 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 4 | ok | 503.027431 | 0.10660449999999999 | 0.11429125 | 0.12003797999999999 | 9289.501748284229 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 8 | ok | 509.053281 | 0.100012 | 0.105614 | 0.10830369999999999 | 9991.88658809047 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 1 | 128 | ok | 547.464353 | 0.242991 | 0.288251 | 0.4222625799999995 | 3908.323548845329 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 1 | ok | 495.973472 | 0.07683999999999999 | 0.08469014999999999 | 0.09063809999999997 | 25655.228112178 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 2 | ok | 501.757547 | 0.1009 | 0.1088725 | 0.12028562 | 19618.35626817273 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 4 | ok | 502.648763 | 0.11713399999999999 | 0.12714565 | 0.12893338 | 16897.078038985954 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 8 | ok | 513.741712 | 0.117322 | 0.13292415 | 0.2510675299999997 | 16268.193534239017 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 2 | 128 | ok | 587.533926 | 0.303106 | 0.3230871 | 0.32843140000000004 | 6563.717815794248 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 1 | ok | 498.664845 | 0.08618400000000001 | 0.09744345 | 0.10020754 | 45617.014599041264 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 2 | ok | 508.395984 | 0.10338800000000001 | 0.11128875 | 0.11431213 | 38354.07329846948 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 4 | ok | 509.036049 | 0.108465 | 0.11588854999999999 | 0.13013737999999994 | 36774.30505757018 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 8 | ok | 508.355527 | 0.116326 | 0.12665659999999998 | 0.13246339999999998 | 33942.14222436435 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 4 | 128 | ok | 549.022295 | 0.213065 | 0.24550065 | 0.25980521999999995 | 18302.58697000398 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 1 | ok | 497.50443 | 0.077841 | 0.08574315 | 0.08962176 | 101665.07052505942 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 2 | ok | 509.347972 | 0.10371649999999999 | 0.1145808 | 0.11916624999999999 | 75918.11092964573 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 4 | ok | 509.007487 | 0.12832100000000002 | 0.13723724999999998 | 0.14088454 | 62168.10971801011 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 8 | ok | 503.801652 | 0.11717549999999999 | 0.1286252 | 0.13066513 | 67750.09226715691 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 8 | 128 | ok | 670.757144 | 0.3043925 | 0.3583903 | 0.7434499599999996 | 24641.044662016655 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 1 | ok | 496.749599 | 0.077723 | 0.08757519999999999 | 0.09271426999999999 | 202783.35361589334 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 2 | ok | 505.124232 | 0.112021 | 0.1272267 | 0.13927775999999997 | 140553.54554224768 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 4 | ok | 503.746336 | 0.1112435 | 0.12061295 | 0.12986602 | 142606.6898580796 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 8 | ok | 501.164966 | 0.1261715 | 0.13734425 | 0.14280539999999997 | 125670.11625899542 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 16 | 128 | ok | 667.620829 | 0.278621 | 0.3286382 | 0.4314406999999997 | 55593.21223556567 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 1 | ok | 504.799514 | 0.09260550000000001 | 0.10321825 | 0.10826366 | 339214.39218823175 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 2 | ok | 499.906111 | 0.1317415 | 0.1417044 | 0.14485873999999999 | 241574.44939521077 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 4 | ok | 503.929002 | 0.12174650000000001 | 0.13297699999999998 | 0.13460769 | 260243.38286469737 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 8 | ok | 520.181731 | 0.136879 | 0.14653385000000002 | 0.14922894 | 234173.21997983474 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 32 | 128 | ok | 546.163432 | 0.2209305 | 0.24475614999999998 | 0.25281191 | 143411.1452330046 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 1 | ok | 496.290955 | 0.094835 | 0.1045199 | 0.10668151 | 668772.743498066 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 2 | ok | 509.695642 | 0.143519 | 0.15219465 | 0.15898436999999999 | 447036.5735987743 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 4 | ok | 518.641861 | 0.15576099999999998 | 0.17180549999999997 | 0.17777282 | 414630.8722745147 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 8 | ok | 502.903111 | 0.14143650000000002 | 0.15880575 | 0.16225888 | 453255.92855213484 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 64 | 128 | ok | 544.578369 | 0.38177649999999996 | 0.4283382 | 0.44498263 | 165844.86218577 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 1 | ok | 493.44499 | 0.097057 | 0.10534009999999999 | 0.11251540999999998 | 1297855.1119166825 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 2 | ok | 508.492684 | 0.172259 | 0.18846565 | 0.19074457 | 766114.2862957118 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 4 | ok | 509.074912 | 0.173321 | 0.1833303 | 0.18676334 | 762052.8178808072 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 8 | ok | 504.007604 | 0.170487 | 0.19612725 | 0.19854531 | 763866.4436813206 | - |
| `full_mlp_capacity_search_hd256_depth3` | `compile` | `bf16` | 128 | 128 | ok | 553.527247 | 0.2963605 | 0.3264666 | 0.33844788 | 428604.98987286777 | - |
