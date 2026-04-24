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

### full_mlp_capacity_search_hd32_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2967605.342` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`bf16`, p50=`0.026` ms, throughput=`37664.500` samples/s

### full_mlp_capacity_search_hd32_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4167084.460` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.016` ms, throughput=`59597.455` samples/s

### full_mlp_capacity_search_hd32_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3088038.531` samples/s, p50=`0.041` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.026` ms, throughput=`36981.127` samples/s

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

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028247500000000002 | 0.03328505 | 0.03949395999999998 | 33499.468363437074 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.028624 | 0.03448495 | 0.03942166 | 33427.66620737054 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.031379000000000004 | 0.03695845 | 0.05568937999999993 | 30611.22035915533 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.028024 | 0.03306625 | 0.03659613999999999 | 34329.93123714773 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.027625 | 0.0297476 | 0.033345959999999994 | 35678.734403933224 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.029015 | 0.03353945 | 0.03496199 | 66740.39255364092 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0335695 | 0.037135699999999994 | 0.04183816999999999 | 60454.77710625955 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.029429499999999997 | 0.03427285 | 0.04096763999999999 | 65033.37512811575 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029822500000000002 | 0.034908299999999996 | 0.038374939999999996 | 63930.811518542185 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.031176000000000002 | 0.03480175 | 0.03660313999999999 | 63826.55516592989 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.029428500000000003 | 0.0343357 | 0.0355997 | 131149.86298118066 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.030405500000000002 | 0.0336935 | 0.04046277 | 128767.99291260967 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029880499999999997 | 0.03506545 | 0.03728458 | 128431.28252120882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.0296275 | 0.0348287 | 0.03756878999999999 | 129134.3992086644 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0291955 | 0.03167015 | 0.04157684 | 134137.17269691505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0297885 | 0.0349441 | 0.03849228 | 255431.91941122944 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0295375 | 0.035310799999999996 | 0.036650129999999996 | 260011.58351604565 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0300295 | 0.0343653 | 0.037578719999999996 | 256518.29007473018 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.031156 | 0.0351444 | 0.038513799999999994 | 252664.66472030655 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0293765 | 0.0306635 | 0.032992 | 270363.98427292705 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.030509500000000002 | 0.036453 | 0.03764888 | 506053.34687869455 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.030949 | 0.03468185 | 0.03838788 | 503253.85067829187 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.034191 | 0.03800115 | 0.042031439999999996 | 460399.592316161 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0303575 | 0.0353249 | 0.03762303 | 505199.76861850603 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0302895 | 0.036275949999999994 | 0.039676119999999995 | 509322.5119276966 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0327185 | 0.038171649999999994 | 0.04083891 | 949793.3309074029 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0322445 | 0.037739949999999994 | 0.039640470000000004 | 946569.6906018645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.033154 | 0.037811149999999995 | 0.039984439999999996 | 946563.5306819694 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0333515 | 0.03903854999999999 | 0.04373782 | 938035.1563851173 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.040190500000000004 | 0.04702274999999999 | 0.08629008999999986 | 792001.9681248908 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0403835 | 0.050749550000000004 | 0.07469351999999992 | 1497296.677639044 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.050497 | 0.059420499999999994 | 0.05993515 | 1228788.5182000857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046968499999999996 | 0.0532095 | 0.05529588999999999 | 1329194.5537914652 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034755499999999995 | 0.045471 | 0.12135197999999976 | 1623697.8704187619 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.0356365 | 0.03664365 | 0.039240189999999994 | 1806061.707484602 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0410185 | 0.0491439 | 0.052180399999999995 | 2967605.341875091 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0715205 | 0.0833704 | 0.08492415 | 1733849.2618543138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.074577 | 0.08873094999999999 | 0.09134764999999999 | 1681646.8998708024 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.061880000000000004 | 0.070505 | 0.07438847 | 2046997.135163541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.27085000000000004 | 0.28895085000000004 | 0.3679958899999997 | 471061.23619422555 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.025749 | 0.030288199999999994 | 0.03372832 | 37664.499702450456 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.026007 | 0.03254235 | 0.037549869999999985 | 35803.30808245359 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.02696 | 0.030457699999999997 | 0.03897508999999998 | 36065.070041972525 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.026853500000000002 | 0.03830014999999997 | 0.09274891999999993 | 33300.210723733464 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.025951500000000002 | 0.02770895 | 0.03487452999999997 | 38116.96113289707 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.042423 | 0.06212124999999998 | 0.0674213 | 43030.685612216934 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0480665 | 0.05920699999999998 | 0.08243467999999994 | 38988.87803265241 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.047455 | 0.0541938 | 0.0553651 | 40920.515172917825 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.04752 | 0.05883794999999998 | 0.0943710099999999 | 39450.96875798882 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.18341 | 0.29461569999999987 | 0.38589291999999986 | 10727.76630204777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.043776999999999996 | 0.053485200000000004 | 0.056823839999999994 | 87002.2546634296 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.048962 | 0.0595294 | 0.05985617 | 78601.76885420628 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.0483205 | 0.059821049999999994 | 0.06778610999999998 | 78787.6471999461 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.049618499999999996 | 0.06333849999999999 | 0.10733209999999985 | 74448.52256906961 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.157446 | 0.22379654999999998 | 0.23330531 | 24203.90922178623 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.048072000000000004 | 0.0538068 | 0.05934545999999999 | 167573.0272779563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.048017000000000004 | 0.057644549999999996 | 0.05838814 | 157956.11110474897 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0508375 | 0.06089669999999999 | 0.10423726999999988 | 148620.65173128198 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.049951499999999996 | 0.0585404 | 0.060902029999999996 | 157417.90557837897 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.15617150000000002 | 0.18094955 | 0.22111508999999985 | 50007.96376823008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0494915 | 0.05604175 | 0.06413243999999999 | 316103.2061162809 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.052336 | 0.06054875 | 0.06424724 | 296116.1041632814 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.054102 | 0.06356574999999998 | 0.07468764 | 286706.5977280653 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.054361 | 0.0657543 | 0.06810713 | 282932.55347690504 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1316885 | 0.15862879999999996 | 0.2195392399999999 | 118348.37738457187 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.048770999999999995 | 0.05668885 | 0.05811194 | 631819.9534269717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.058746 | 0.07020715 | 0.07310345 | 526070.7512553363 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.058009 | 0.06837945 | 0.06967138 | 534274.5469936205 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0597105 | 0.07555139999999998 | 0.11555048999999987 | 496319.480849823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.12252350000000001 | 0.137511 | 0.146244 | 260370.7712328703 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.060196 | 0.06889375 | 0.07200772999999999 | 1036985.7158458222 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.073559 | 0.0858399 | 0.08735153 | 852270.8356915048 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.0767015 | 0.0918799 | 0.09515977 | 802806.6119152558 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.075888 | 0.0888887 | 0.12229738999999988 | 810614.3874427725 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.17057699999999998 | 0.18952409999999997 | 0.19799493999999998 | 374847.58638567623 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.076375 | 0.0921742 | 0.09301634 | 1602908.4774323008 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.08946299999999999 | 0.1028604 | 0.11164427999999998 | 1401626.105286211 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.104484 | 0.11850805 | 0.15737861999999986 | 1195731.984149078 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.0983235 | 0.1043846 | 0.10786413 | 1320315.299545502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.19097 | 0.21952335 | 0.22350822999999997 | 663157.3210185019 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 1 | ok | 43.846825 | 0.017752 | 0.021800849999999997 | 0.03132127999999997 | 53673.119964790436 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 2 | ok | 42.349928 | 0.0189845 | 0.0206047 | 0.02407947999999999 | 52990.91311821849 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.505078 | 0.018281 | 0.02122035 | 0.023943399999999997 | 53719.483304521775 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.392841 | 0.0189695 | 0.020002 | 0.02083212 | 53255.67945193515 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 1 | 128 | ok | 43.449875 | 0.0189005 | 0.0198167 | 0.02155925 | 53587.69626493757 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 1 | ok | 41.38204 | 0.019893 | 0.02144935 | 0.022885549999999998 | 101294.03124920864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.089415 | 0.0192055 | 0.0195429 | 0.022051109999999995 | 103616.6350290541 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.950749 | 0.019558 | 0.02032045 | 0.02455970999999999 | 101543.66682304391 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.190256 | 0.019898 | 0.0214933 | 0.02412978 | 101021.52970841144 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 2 | 128 | ok | 46.661796 | 0.019015999999999998 | 0.01952 | 0.020754589999999996 | 104800.05198082578 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.645209 | 0.01931 | 0.019873349999999998 | 0.020682179999999998 | 207206.21784418507 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 2 | ok | 41.312654 | 0.0191555 | 0.01964405 | 0.022576689999999993 | 208565.1448015659 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.234417 | 0.0192605 | 0.0196948 | 0.020644839999999998 | 208301.0032817823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 8 | ok | 41.184173 | 0.0195505 | 0.0198898 | 0.021093419999999995 | 205753.69636515516 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.121747 | 0.019540000000000002 | 0.0202099 | 0.022240879999999994 | 205702.27270155994 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 1 | ok | 41.459285 | 0.019806 | 0.0202844 | 0.02471998 | 400744.1819458735 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.304984 | 0.019355999999999998 | 0.02000265 | 0.024960499999999997 | 409146.0508200309 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.529299 | 0.0194275 | 0.02238775 | 0.02685480999999999 | 392031.5663817251 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 8 | ok | 41.152275 | 0.0197915 | 0.0201668 | 0.024863199999999988 | 401023.81379662285 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 8 | 128 | ok | 47.098213 | 0.019568500000000003 | 0.02000305 | 0.022876429999999996 | 406179.204234012 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 1 | ok | 42.353218 | 0.020195499999999998 | 0.02069045 | 0.022381149999999992 | 788879.1703357766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.982488 | 0.020101 | 0.0204758 | 0.021376519999999996 | 796331.3016930999 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.193705 | 0.0197665 | 0.02004195 | 0.02276361999999999 | 805409.1277016442 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 8 | ok | 41.473203 | 0.020385 | 0.02098965 | 0.027007629999999998 | 778982.2791268777 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 16 | 128 | ok | 48.032858 | 0.0200185 | 0.0203588 | 0.02125278 | 798039.2176422529 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 1 | ok | 43.189298 | 0.0216995 | 0.0221127 | 0.023229869999999996 | 1472301.409544562 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 2 | ok | 43.402913 | 0.0218635 | 0.02526755 | 0.026966659999999996 | 1394972.5190413748 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 4 | ok | 43.452378 | 0.022035 | 0.022712049999999998 | 0.024354699999999993 | 1460416.3281847574 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 8 | ok | 41.830747 | 0.021618 | 0.02471375 | 0.02535744 | 1427713.5257117823 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 32 | 128 | ok | 43.657286 | 0.021561999999999998 | 0.022038299999999997 | 0.02475102999999999 | 1480889.1258311493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 1 | ok | 41.972458 | 0.024953999999999997 | 0.0263492 | 0.02938581 | 2543831.407573463 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.948331 | 0.0365075 | 0.0395011 | 0.04105336 | 1737408.134544886 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.013829 | 0.032344 | 0.0342793 | 0.03673004999999999 | 1957972.128266754 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 8 | ok | 42.077622 | 0.025049 | 0.02645185 | 0.028251179999999994 | 2535575.310152364 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 64 | 128 | ok | 48.974265 | 0.024653500000000002 | 0.0255565 | 0.028892699999999986 | 2592285.358772293 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 1 | ok | 44.287317 | 0.030557 | 0.0312256 | 0.03585251999999999 | 4167084.460290941 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 2 | ok | 43.010977 | 0.0533335 | 0.0561596 | 0.05868747999999999 | 2391097.6445820155 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.948292 | 0.053305500000000006 | 0.056649649999999996 | 0.05831139999999999 | 2403734.20108138 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 8 | ok | 45.48402 | 0.0458695 | 0.0507945 | 0.05437896999999999 | 2758601.6648161043 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `fp32` | 128 | 128 | ok | 89.946533 | 0.16880699999999998 | 0.1817862 | 0.18737449999999997 | 752549.1426348721 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 1 | ok | 40.92889 | 0.016055 | 0.017141299999999998 | 0.019408529999999997 | 61607.68962538827 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.420626 | 0.015847 | 0.02151775 | 0.026668069999999978 | 59597.454950283805 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.116667 | 0.015891000000000002 | 0.01695195 | 0.02109829999999999 | 61939.524725638876 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 8 | ok | 42.602454 | 0.015886 | 0.0170243 | 0.019999319999999998 | 62346.39407160608 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 1 | 128 | ok | 47.081314 | 0.016002 | 0.01682225 | 0.01842255 | 62140.51710852717 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 1 | ok | 43.384008 | 0.029670500000000002 | 0.0330286 | 0.03409784 | 66706.55718786501 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 2 | ok | 42.586042 | 0.034416 | 0.03738349999999999 | 0.04308629 | 57286.56458200859 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 4 | ok | 44.40495 | 0.035571 | 0.03839094999999999 | 0.04422548999999999 | 55915.08958156502 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.460695 | 0.0341085 | 0.03657414999999999 | 0.04080144999999999 | 58121.89556425317 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 2 | 128 | ok | 89.420303 | 0.1379565 | 0.21292479999999994 | 0.22185377 | 13589.072428940703 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 1 | ok | 44.761235 | 0.031252 | 0.0345118 | 0.03734712 | 126089.17404656095 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.550682 | 0.035071 | 0.03857164999999999 | 0.04549958999999999 | 113813.4211062437 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 4 | ok | 44.11311 | 0.036236500000000005 | 0.038807549999999996 | 0.041037899999999995 | 109745.69179568645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.788835 | 0.036890000000000006 | 0.04123354999999998 | 0.04601748999999999 | 107366.81281079337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 4 | 128 | ok | 73.636981 | 0.1002835 | 0.11890424999999999 | 0.12432637999999999 | 39415.32877902348 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 1 | ok | 44.627468 | 0.032004 | 0.035174899999999995 | 0.03740568 | 246004.57814519928 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.321699 | 0.035789 | 0.041380249999999986 | 0.04466493999999999 | 217693.1169246058 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.791676 | 0.0376395 | 0.040585949999999996 | 0.04497573 | 210284.0464328203 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 8 | ok | 43.815735 | 0.0375765 | 0.04172114999999999 | 0.046301539999999995 | 209984.22493510175 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 8 | 128 | ok | 101.314583 | 0.1043065 | 0.15529405 | 0.17246447 | 71190.40139817947 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 1 | ok | 43.039856 | 0.035266000000000006 | 0.03789615 | 0.041249619999999994 | 450272.97799290816 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 2 | ok | 44.677858 | 0.039573 | 0.042065399999999996 | 0.046600630000000004 | 405524.4597146932 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 4 | ok | 44.556901 | 0.041185 | 0.04494275 | 0.046847429999999995 | 384842.404630039 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 8 | ok | 43.224673 | 0.0416715 | 0.048645049999999995 | 0.051763819999999995 | 377248.69695942267 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 16 | 128 | ok | 81.692202 | 0.109663 | 0.1234395 | 0.13014804 | 144917.00874628493 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 1 | ok | 45.059989 | 0.0382375 | 0.0434258 | 0.0449245 | 823827.1404702509 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.658099 | 0.0439595 | 0.049581099999999996 | 0.05512590999999999 | 723555.6246953153 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 4 | ok | 43.974074 | 0.049017500000000006 | 0.05208354999999999 | 0.05549772 | 649965.267481019 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 8 | ok | 51.034903 | 0.055705500000000005 | 0.0586675 | 0.06479051999999999 | 573027.3176448004 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 32 | 128 | ok | 76.997294 | 0.11827499999999999 | 0.1616496 | 0.18796192 | 261753.59530832857 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 1 | ok | 43.298628 | 0.046767500000000004 | 0.048733399999999996 | 0.053619169999999994 | 1355002.074846927 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.045756 | 0.05427 | 0.05768155 | 0.059450289999999996 | 1174095.2037110215 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 4 | ok | 44.356558 | 0.0599065 | 0.06306465 | 0.06432262 | 1071507.3798396557 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 8 | ok | 47.299612 | 0.062101 | 0.06599189999999999 | 0.06909706 | 1028224.4395453834 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 64 | 128 | ok | 98.913978 | 0.126595 | 0.1460549 | 0.15063664000000002 | 498351.8569680335 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.921379 | 0.06251899999999999 | 0.06517674999999999 | 0.06716623 | 2035769.7468074993 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 2 | ok | 45.86039 | 0.0729835 | 0.0811934 | 0.08165849 | 1720430.1075268819 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.491496 | 0.0836505 | 0.0878419 | 0.0905348 | 1555860.6230881452 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 8 | ok | 45.073567 | 0.08296100000000001 | 0.0901104 | 0.09244605 | 1540543.4941163273 | - |
| `full_mlp_capacity_search_hd32_depth2` | `jit` | `bf16` | 128 | 128 | ok | 101.766456 | 0.19449850000000002 | 0.228237 | 0.23317276 | 642837.8075012141 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 1 | ok | 498.302799 | 0.029915499999999998 | 0.032708049999999995 | 0.03932052 | 33016.44152755189 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 2 | ok | 508.416534 | 0.0290025 | 0.03260645 | 0.03485901999999999 | 33961.90018189994 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 4 | ok | 511.425052 | 0.02683 | 0.031169249999999996 | 0.0362484 | 36210.97091027864 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 8 | ok | 548.903551 | 0.027195 | 0.03040605 | 0.032115149999999995 | 36132.75463113516 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 1 | 128 | ok | 852.415844 | 0.0264805 | 0.0309466 | 0.03150972 | 36981.127051620475 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 1 | ok | 498.873714 | 0.029700999999999998 | 0.0328662 | 0.03995992 | 66840.45184145444 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 2 | ok | 509.166256 | 0.027697 | 0.029289299999999997 | 0.03254806999999999 | 71686.9516844283 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 4 | ok | 508.063547 | 0.02688 | 0.02898025 | 0.02956948 | 74045.82696230696 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 8 | ok | 537.636673 | 0.028591 | 0.03145345 | 0.03365347 | 69229.44854590467 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 2 | 128 | ok | 746.759894 | 0.0273565 | 0.029481149999999998 | 0.0318038 | 72447.07379024253 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 1 | ok | 502.10889 | 0.028414500000000002 | 0.03093205 | 0.032799079999999994 | 139704.41340212378 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 2 | ok | 513.863442 | 0.027875999999999998 | 0.03068915 | 0.035558860000000005 | 141714.61935098955 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 4 | ok | 515.312744 | 0.028123000000000002 | 0.0315218 | 0.03436726 | 140085.0596482184 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 8 | ok | 542.135837 | 0.028053500000000002 | 0.03204745 | 0.03324581 | 140630.95484199407 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 4 | 128 | ok | 721.917328 | 0.0295985 | 0.03209615 | 0.03350052 | 134325.55140638852 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 1 | ok | 498.159274 | 0.0281735 | 0.0321559 | 0.036318149999999993 | 278095.8499061079 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 2 | ok | 509.434709 | 0.030172 | 0.0338197 | 0.03489295 | 260812.469965814 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 4 | ok | 505.653217 | 0.02802 | 0.0303678 | 0.032731039999999996 | 283676.87785228237 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 8 | ok | 547.688592 | 0.0282285 | 0.03036015 | 0.03385934999999999 | 281236.2018488468 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 8 | 128 | ok | 842.324723 | 0.0272955 | 0.03013835 | 0.03075355 | 289661.3352084077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 1 | ok | 499.31587 | 0.030681 | 0.0349325 | 0.03674844 | 510150.398714421 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 2 | ok | 497.924929 | 0.031035 | 0.0345967 | 0.03558276 | 508413.6097239186 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 4 | ok | 521.553315 | 0.02849 | 0.0321686 | 0.03648441999999998 | 547708.8653526219 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 8 | ok | 543.207469 | 0.030402 | 0.03409625 | 0.03813696 | 521356.7266096563 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 16 | 128 | ok | 742.629754 | 0.030865999999999998 | 0.03515045 | 0.036755039999999996 | 505720.3290722181 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 1 | ok | 500.638943 | 0.0308125 | 0.035142 | 0.03954359999999999 | 1006752.7943682249 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 2 | ok | 508.234267 | 0.030455 | 0.03368055 | 0.03384937 | 1030415.9467171914 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 4 | ok | 508.584618 | 0.029844500000000003 | 0.033385899999999996 | 0.03440472 | 1058588.9274244665 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 8 | ok | 547.683191 | 0.031932 | 0.03366015 | 0.036089609999999994 | 1008427.9364791244 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 32 | 128 | ok | 851.737417 | 0.0319975 | 0.0338152 | 0.03621223999999999 | 1007665.1830896148 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 1 | ok | 495.824674 | 0.0342585 | 0.038343749999999996 | 0.043906020000000004 | 1816856.3388982126 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 2 | ok | 517.120053 | 0.062173000000000006 | 0.06664235 | 0.06810578 | 1023811.9462217181 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 4 | ok | 518.56859 | 0.0491475 | 0.056758649999999994 | 0.05774322 | 1291668.9371784704 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 8 | ok | 540.681839 | 0.0343705 | 0.03816894999999999 | 0.04070022 | 1821373.3610485645 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 64 | 128 | ok | 698.081401 | 0.034625500000000003 | 0.03887385 | 0.03979219 | 1817422.8376217887 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 1 | ok | 493.741066 | 0.0406885 | 0.044396899999999996 | 0.04740936 | 3088038.53100077 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 2 | ok | 508.673281 | 0.0733655 | 0.0787969 | 0.08108269 | 1738287.270984047 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 4 | ok | 531.177186 | 0.095122 | 0.10409125 | 0.2598441399999994 | 1256353.4184983908 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.848134 | 0.057874 | 0.06576739999999999 | 0.06733317 | 2178925.9052841337 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `fp32` | 128 | 128 | ok | 797.649847 | 5.9969435 | 8.750185149999997 | 9.395882839999999 | 28610.83501531142 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 1 | ok | 501.022399 | 0.0294875 | 0.0309897 | 0.034356269999999994 | 33727.17681944628 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 2 | ok | 510.557456 | 0.0287105 | 0.03060995 | 0.03607202999999998 | 34499.29448942769 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 4 | ok | 504.702621 | 0.0290975 | 0.03061465 | 0.035328179999999994 | 34076.03180520505 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 8 | ok | 510.187 | 0.029355 | 0.03119585 | 0.03345085999999999 | 33897.06410748352 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 1 | 128 | ok | 536.473291 | 0.0308855 | 0.032570550000000004 | 0.03590725999999998 | 32374.44542574986 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.18565 | 0.0436015 | 0.04757455 | 0.052034409999999996 | 45964.92324777116 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 2 | ok | 500.557472 | 0.045630000000000004 | 0.0516985 | 0.05514234999999999 | 43088.7572383726 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 4 | ok | 507.335305 | 0.0478185 | 0.05131035 | 0.05583846 | 41799.71166558893 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 8 | ok | 504.1432 | 0.0531095 | 0.061627699999999994 | 0.062296769999999994 | 37178.92282250485 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 2 | 128 | ok | 636.662235 | 0.1333615 | 0.21438409999999986 | 3.736454989999991 | 7150.462170122366 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 1 | ok | 507.78402 | 0.045188 | 0.0502933 | 0.05160177 | 87954.84749948766 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 2 | ok | 503.649285 | 0.045266 | 0.05072085 | 0.053651479999999994 | 87060.27487540587 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 4 | ok | 510.900232 | 0.052478 | 0.05832889999999999 | 0.06236795 | 74881.01406864492 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 8 | ok | 508.980892 | 0.046952 | 0.05375119999999999 | 0.05718402999999999 | 83231.4788111382 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 4 | 128 | ok | 536.372557 | 0.12711 | 0.18828284999999984 | 4.450803469999993 | 13253.629539583488 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 1 | ok | 499.123785 | 0.0448795 | 0.0508276 | 0.052643539999999996 | 177022.79522534116 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 2 | ok | 503.056844 | 0.04879 | 0.052805399999999995 | 0.056900679999999995 | 163261.0413442261 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 4 | ok | 505.761862 | 0.0495785 | 0.057159999999999996 | 0.05863732 | 159904.76072451245 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 8 | ok | 503.25238 | 0.0557225 | 0.06228409999999998 | 0.06493628 | 141461.4238233769 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 8 | 128 | ok | 617.259633 | 0.107397 | 0.1337786 | 0.14630295 | 72789.26271027909 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 1 | ok | 534.929374 | 0.0456015 | 0.04968635 | 0.05112796 | 348078.0652080745 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 2 | ok | 509.321612 | 0.0486085 | 0.054344399999999994 | 0.05656907 | 325454.0898188197 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 4 | ok | 501.574634 | 0.055807999999999996 | 0.0646374 | 0.06633807 | 282047.3677400567 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 8 | ok | 508.597271 | 0.0537415 | 0.0618906 | 0.06528763 | 293906.18588023254 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 16 | 128 | ok | 528.867144 | 0.1169155 | 0.15023544999999988 | 2.319498229999992 | 78127.03710145567 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 1 | ok | 499.691849 | 0.0475905 | 0.0514221 | 0.05270829999999999 | 672223.4336248683 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 2 | ok | 508.434006 | 0.052291000000000004 | 0.059030849999999996 | 0.06281787999999999 | 597783.7913166675 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 4 | ok | 507.219521 | 0.057870500000000005 | 0.0658357 | 0.06722498 | 545848.5488616328 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 8 | ok | 510.305732 | 0.0536125 | 0.06397445 | 0.06453022 | 581851.8234327002 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 32 | 128 | ok | 528.93549 | 0.170215 | 0.27955055 | 0.28881347 | 170693.79560058078 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 1 | ok | 495.842271 | 0.0497645 | 0.055846599999999996 | 0.05910612 | 1274217.6005288004 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 2 | ok | 497.437311 | 0.0671545 | 0.0745779 | 0.07830308 | 950789.7943411963 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 4 | ok | 499.736072 | 0.0615015 | 0.06803105 | 0.07170837999999999 | 1045591.3783148922 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 8 | ok | 504.955458 | 0.057245000000000004 | 0.06688759999999999 | 0.06908462 | 1098526.4983472326 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 64 | 128 | ok | 527.39962 | 0.135243 | 0.17086055 | 0.20590675999999988 | 457163.29887322104 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 1 | ok | 492.01519 | 0.0621415 | 0.06904515 | 0.07068656999999999 | 2012861.5563256848 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 2 | ok | 510.494846 | 0.08444650000000001 | 0.0964622 | 0.09775117 | 1502714.277664031 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 4 | ok | 507.387266 | 0.09104699999999999 | 0.1006705 | 0.10411883999999999 | 1409845.9677193735 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 8 | ok | 508.204606 | 0.074879 | 0.0831107 | 0.09068665999999999 | 1688928.414505256 | - |
| `full_mlp_capacity_search_hd32_depth2` | `compile` | `bf16` | 128 | 128 | ok | 532.18915 | 0.224277 | 0.28729129999999986 | 0.33233360999999995 | 551163.091914713 | - |
