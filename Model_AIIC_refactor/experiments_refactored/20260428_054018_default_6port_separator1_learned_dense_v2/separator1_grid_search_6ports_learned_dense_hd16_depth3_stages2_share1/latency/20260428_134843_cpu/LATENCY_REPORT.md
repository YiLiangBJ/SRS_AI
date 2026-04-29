# Latency Report

- Device: `cpu`
- Runtime backends: `['pytorch', 'onnxruntime', 'openvino']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `pytorch`
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

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`174287.344` samples/s, p50=`0.732` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`bf16`, p50=`0.495` ms, throughput=`2017.695` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`419601.225` samples/s, p50=`0.305` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.146` ms, throughput=`6808.634` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`225327.755` samples/s, p50=`0.554` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.465` ms, throughput=`2133.543` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`701135.280` samples/s, p50=`0.182` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19681.638` samples/s

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`41762.722` samples/s, p50=`3.047` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.623` ms, throughput=`1599.525` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,656`
- MACs / sample: `19,968`
- FLOPs / sample estimate: `41,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5378134999999999 | 0.5525829 | 0.6397775499999999 | 1844.5107673131588 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.528349 | 0.53851455 | 0.5808145599999999 | 1887.1780068577032 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5213695 | 0.5890835999999998 | 0.62151353 | 1892.778274842384 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5397734999999999 | 0.54532345 | 0.54899854 | 1850.904462974877 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5772905 | 0.59273355 | 0.7480414599999997 | 3419.572676520052 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5657755 | 0.6222916999999998 | 0.6556313 | 3498.419676371691 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.563347 | 0.58651725 | 0.6511312699999999 | 3523.0407392805055 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5615 | 0.5750463499999999 | 0.6409741999999999 | 3542.4130457153715 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.569635 | 0.57671715 | 0.5778030200000001 | 7015.19709161153 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.5682860000000001 | 0.57498905 | 0.5767670100000001 | 7036.655415758322 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5757939999999999 | 0.66911515 | 0.67307096 | 6837.443907746883 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5657695 | 0.58107335 | 0.6611699 | 7011.8197544418645 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.570301 | 0.6202571499999998 | 0.6633577799999999 | 13883.564927393812 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.579002 | 0.5891378 | 0.5910775899999999 | 13798.818538257345 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.5634870000000001 | 0.6178296499999998 | 0.65379879 | 14050.127764836829 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.576192 | 0.6196383999999998 | 0.66903854 | 13756.104744346407 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.5841285 | 0.6770796 | 0.68178858 | 26865.478370737692 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.5743214999999999 | 0.66833235 | 0.67219194 | 27420.812122192627 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.561466 | 0.6196235499999998 | 0.6622024799999999 | 28176.352833870147 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.577868 | 0.6648868499999999 | 0.67511854 | 27290.920082327153 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.6118705 | 0.7072055500000001 | 0.7165506 | 51407.358365739514 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.610631 | 0.6983075 | 0.71542129 | 51612.44204366303 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.604514 | 0.6291479 | 0.66163529 | 52642.72396616599 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.60057 | 0.6189854 | 0.71113092 | 52871.402935407066 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6490815000000001 | 0.6666219 | 0.66909677 | 98319.35355025042 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.6386145000000001 | 0.6519850500000001 | 0.65574775 | 100095.71965991605 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.6435960000000001 | 0.65755 | 0.66186552 | 99324.91647783293 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.650856 | 0.6681028 | 0.67310668 | 98113.69071081653 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.739518 | 0.7657516 | 0.76725002 | 172230.99718222016 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.7308859999999999 | 0.76054155 | 0.8636992099999999 | 173399.88348611578 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.732064 | 0.75667775 | 0.75993422 | 174287.3444948401 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.7292000000000001 | 0.7766841499999999 | 0.8474647599999998 | 173672.28555305736 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.514645 | 0.51881305 | 0.5201436899999999 | 1942.728292235758 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.49523 | 0.5006159 | 0.50186639 | 2017.6950239443904 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5067585 | 0.51200585 | 0.51380407 | 1972.4601173495425 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5268025000000001 | 0.5307617 | 0.53256331 | 1899.758772430594 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.5306105000000001 | 0.53884115 | 0.54263977 | 3764.2984524027984 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.5318445 | 0.53903925 | 0.54062644 | 3761.5885609790935 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.5203450000000001 | 0.5272601499999999 | 0.52820119 | 3842.1903312740346 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.5276205 | 0.5318632 | 0.53521064 | 3792.0432845055934 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.5640354999999999 | 0.56877485 | 0.57330038 | 7094.310522582875 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.5544695 | 0.5610241499999999 | 0.56247761 | 7209.928013194744 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5639164999999999 | 0.5707983 | 0.57553103 | 7087.387451844303 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.5461655000000001 | 0.5503968 | 0.55176666 | 7320.382429954829 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.6005785 | 0.6075583 | 0.6095162000000001 | 13309.9238392848 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6118005 | 0.6177201999999999 | 0.62024599 | 13066.279486672918 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.6115630000000001 | 0.6171791 | 0.62153415 | 13084.008822547148 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.6076385 | 0.6136081999999999 | 0.7045411199999997 | 13079.650460729052 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.0554975 | 1.0955053 | 1.10691426 | 15091.143147679699 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.113659 | 1.13962005 | 1.2133827199999998 | 14318.20664318612 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.2097725000000001 | 1.24332175 | 1.2465042 | 13273.636260388592 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.1518435 | 1.17158345 | 1.18492204 | 13885.864615076454 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.5356745 | 1.5752955 | 1.5841944799999998 | 20802.156101875647 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.8355584999999999 | 1.8792111500000002 | 1.92487587 | 17411.846162595608 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.8486785000000001 | 1.8874989 | 1.90047819 | 17262.835452227704 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.9214365 | 1.96215775 | 1.9717335 | 16616.88369476492 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.670991 | 1.69313655 | 1.7098653099999999 | 38298.51384053236 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.892586 | 1.94667335 | 1.95711333 | 33716.813975602534 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.041795 | 2.0630228 | 2.08481114 | 31353.934926181817 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.161087 | 2.2227674 | 2.24050484 | 29558.838674241375 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.7459155 | 1.77919575 | 1.7957438499999998 | 73296.18270502816 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.1138055 | 2.1449219 | 2.14951571 | 60520.437368718514 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.2125275 | 2.2324810999999998 | 2.23820869 | 57842.582478258024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.1828575 | 2.21236955 | 2.22208501 | 58643.94848972131 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 219.847232 | 0.1455295 | 0.15305535 | 0.16134097999999997 | 6808.63394700459 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 221.844877 | 0.149982 | 0.15729374999999998 | 0.15941191000000002 | 6635.338138822155 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 217.260538 | 0.149401 | 0.15179475 | 0.1535615 | 6677.296031642904 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 220.181188 | 0.164302 | 0.18529184999999992 | 0.19824393999999998 | 6009.485371710708 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 217.62691 | 0.172062 | 0.17673434999999998 | 0.17828621 | 11578.696171654632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 219.227222 | 0.17577199999999998 | 0.1781801 | 0.17935679 | 11357.61424680959 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 217.816868 | 0.168318 | 0.16959174999999999 | 0.16976832 | 11885.733411557307 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 221.074703 | 0.17619800000000002 | 0.18089960000000002 | 0.18217563 | 11333.207988052985 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 217.02476 | 0.169999 | 0.1731145 | 0.17427209999999999 | 23468.080008317087 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 220.906664 | 0.1673585 | 0.16948335 | 0.17057358 | 23855.71870481578 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 218.288921 | 0.17217 | 0.17479795 | 0.17535349 | 23174.13321789525 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 218.450835 | 0.1725465 | 0.1758265 | 0.17842555 | 23133.23738437574 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 222.693537 | 0.174637 | 0.1764277 | 0.17826817 | 45738.765101367964 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 219.325188 | 0.17626350000000002 | 0.17769379999999999 | 0.17824686 | 45372.313292647224 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 218.370646 | 0.174509 | 0.17811385 | 0.17994603 | 45733.09639016094 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 221.523856 | 0.1762035 | 0.1805065 | 0.18190537999999998 | 45301.703683821295 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 218.280425 | 0.1780235 | 0.1805291 | 0.18167070999999999 | 89887.458654578 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 221.436186 | 0.1855795 | 0.1911344 | 0.1916723 | 86006.63455178932 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 218.564886 | 0.186363 | 0.18886015 | 0.1900694 | 85792.14515001453 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 220.486347 | 0.18036950000000002 | 0.1856292 | 0.18914948 | 88468.10509170438 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 222.430613 | 0.21751700000000002 | 0.22260805 | 0.22418908999999998 | 146584.9016085494 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 222.143558 | 0.2148925 | 0.22111015 | 0.22296557 | 148442.83466437075 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 219.671629 | 0.2104415 | 0.213444 | 0.21440888 | 151858.97692037778 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 219.709292 | 0.21087250000000002 | 0.21594395 | 0.21644639 | 151428.1233635707 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 220.206183 | 0.250091 | 0.25617865 | 0.25936967 | 254917.82285840152 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 223.589599 | 0.2492685 | 0.2519515 | 0.25364059 | 256497.00912464067 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 218.957953 | 0.24917 | 0.25465699999999997 | 0.25618456 | 256317.17712766485 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 221.708996 | 0.249166 | 0.25308945 | 0.25386426 | 256252.95241438728 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 223.202949 | 0.304551 | 0.31024070000000004 | 0.31097822999999997 | 419601.22541915375 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 224.117611 | 0.30893950000000003 | 0.3163837 | 0.31922974 | 413356.18417989 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 224.378458 | 0.3129435 | 0.3207173 | 0.3216804 | 407776.5537401712 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 222.688912 | 0.313346 | 0.31970244999999997 | 0.32083811999999995 | 408108.5038079074 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 222.435402 | 0.15121400000000002 | 0.15904154999999998 | 0.21160424999999994 | 6490.595840592524 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 220.81514 | 0.150917 | 0.15648145 | 0.15710243999999998 | 6585.186832945269 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 216.230026 | 0.153359 | 0.15561195 | 0.15626774 | 6510.217460793842 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 218.344892 | 0.1548215 | 0.1571902 | 0.15965175 | 6448.37706601165 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 215.910597 | 0.169188 | 0.1755689 | 0.17735041 | 11761.201044441697 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 217.2012 | 0.16913499999999998 | 0.17205555 | 0.17408973 | 11796.946242495667 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 220.562891 | 0.174653 | 0.17688685 | 0.18150506 | 11425.49503242327 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 220.303201 | 0.174336 | 0.1773523 | 0.17969997999999998 | 11457.021504371083 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 220.76226 | 0.20103 | 0.20956239999999998 | 0.21556585 | 19745.432017711653 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 217.263072 | 0.1948955 | 0.19924375 | 0.2000988 | 20468.29619548615 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 219.437196 | 0.1954245 | 0.19793725 | 0.20048496 | 20439.364581033904 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 216.605199 | 0.20145849999999998 | 0.2057699 | 0.2067651 | 19806.939777296713 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 220.231767 | 0.24947950000000002 | 0.25534629999999997 | 0.26537241 | 31926.01592772972 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 218.898412 | 0.24709799999999998 | 0.2489823 | 0.25055083 | 32380.827020893485 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 218.835316 | 0.24848599999999998 | 0.25442319999999996 | 0.25539929 | 32110.269233368927 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 217.793301 | 0.2496785 | 0.25183515 | 0.25267927 | 32024.938460077632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 224.493238 | 0.5396455 | 0.5465606 | 0.54829682 | 29609.09955730325 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 227.126421 | 0.6198295 | 0.6937985 | 0.70665348 | 25457.78183291956 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 230.146638 | 0.6145415 | 0.6238247499999999 | 0.63186589 | 26007.941004626944 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 228.649203 | 0.659679 | 0.6776677 | 0.68275226 | 24216.0067804819 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 230.768128 | 0.9098485000000001 | 0.9722159 | 0.98514487 | 34768.18670064272 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 230.545362 | 1.0314895000000002 | 1.0417478 | 1.05918607 | 30989.868501659425 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 237.11297 | 1.1279845000000002 | 1.22932705 | 1.3431202499999997 | 28031.1879903038 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 238.469178 | 1.231043 | 1.27065455 | 1.27721035 | 25918.407847756953 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 234.050528 | 1.0059010000000002 | 1.0706685999999999 | 1.07596549 | 62778.90509508885 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 237.359588 | 1.2281385 | 1.32229525 | 1.3330376099999999 | 51595.43068286101 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 237.612426 | 1.3306695 | 1.35090175 | 1.35234453 | 48221.06610207031 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 236.116462 | 1.3491695 | 1.3906387 | 1.40403898 | 47538.08220056398 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 231.144147 | 1.1769975000000001 | 1.21066205 | 1.22197168 | 108521.56978960261 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 233.716518 | 1.436633 | 1.46517345 | 1.48059974 | 89111.86285940502 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 235.961406 | 1.496368 | 1.5212127 | 1.5248289899999998 | 85378.49855907109 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 237.826501 | 1.572351 | 1.59490325 | 1.61621673 | 81360.08225707716 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 492.672527 | 0.46729200000000004 | 0.47235815 | 0.47524874 | 2138.8853549749747 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 502.665091 | 0.46595050000000005 | 0.47145309999999996 | 0.47226245 | 2145.6519157817893 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.854207 | 0.4694685 | 0.48456965 | 0.52973281 | 2117.3187697937547 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 549.367996 | 0.46539200000000003 | 0.48076179999999996 | 0.5382766800000001 | 2133.5426139506976 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.20824 | 0.482624 | 0.6353608499999996 | 3.1291231699999913 | 3337.904705090445 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 508.597103 | 0.479166 | 0.5052621999999999 | 0.55788326 | 4140.916035225282 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 509.507964 | 0.4773845 | 0.48365395 | 0.48482117 | 4191.797892296951 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 533.078232 | 0.4728025 | 0.53604895 | 0.5666482399999999 | 4174.922576060827 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.533134 | 0.4855565 | 0.6384899999999998 | 3.076759439999991 | 6686.891616152495 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.655355 | 0.477675 | 0.49511229999999995 | 0.55680348 | 8315.035059305948 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 509.856111 | 0.474379 | 0.4813226 | 0.48505132 | 8432.057976132386 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 547.515982 | 0.479475 | 0.5187395999999999 | 0.5627861999999999 | 8268.41969291172 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 496.028036 | 0.4858885 | 0.5427329 | 0.58024596 | 16255.612351762024 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 501.501095 | 0.48976949999999997 | 0.5069505999999999 | 0.56993244 | 16221.715158442561 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 508.322615 | 0.474107 | 0.5322833499999999 | 0.57049407 | 16660.421785234168 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 533.073834 | 0.478258 | 0.48621899999999996 | 0.48809077 | 16715.581333024988 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 491.785999 | 0.484964 | 0.55989115 | 2.854278569999992 | 27353.213169984047 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 514.524799 | 0.4839695 | 0.48732179999999997 | 0.48993057999999995 | 33076.841720614306 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.223528 | 0.4747695 | 0.48191385 | 0.48372497 | 33680.099427021516 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 535.107673 | 0.478503 | 0.48476735 | 0.48626522 | 33443.615047436215 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 495.218341 | 0.4929015 | 0.5022771 | 0.50520981 | 64775.549885715685 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 519.474261 | 0.488766 | 0.55291905 | 0.57238669 | 64423.19597435555 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 508.657424 | 0.4919005 | 0.5473283999999998 | 0.5935032 | 64200.683632954584 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 529.713723 | 0.487375 | 0.53156095 | 0.57308958 | 65018.62600770235 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.068269 | 0.5167745 | 0.5959740499999999 | 0.61717931 | 121901.01896299871 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 512.396046 | 0.5227459999999999 | 0.534183 | 0.5930963199999999 | 121671.26431956714 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 509.949146 | 0.5155805 | 0.5857750999999999 | 0.61408392 | 122124.61878514018 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 539.639491 | 0.5185109999999999 | 0.5315711 | 0.59332723 | 122785.54353706782 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.003433 | 0.559385 | 0.6861789 | 3.0208367299999916 | 192047.72385937907 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 506.021125 | 0.5757475 | 0.58969845 | 0.59416186 | 222079.2355810632 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 510.537743 | 0.5723505 | 0.6312248 | 0.6874304299999999 | 220990.59864557622 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 530.752692 | 0.5544725 | 0.66885995 | 0.6860588599999999 | 225327.75506309702 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.912057 | 0.4704165 | 0.47956945 | 0.48286093 | 2119.718477645322 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.519195 | 0.4715545 | 0.47948240000000003 | 0.48152796000000003 | 2117.3710433163724 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.426277 | 0.4714395 | 0.50813035 | 0.52827889 | 2100.6749300496253 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 499.285034 | 0.47063049999999995 | 0.5225860999999999 | 0.52592988 | 2090.388481558112 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.454373 | 0.4875045 | 0.5360795 | 0.56114654 | 4031.5839121579143 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 501.155215 | 0.4752865 | 0.5238398 | 0.54186949 | 4139.100772906705 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 498.658064 | 0.4798215 | 0.48913945000000003 | 0.48998511 | 4161.254608277384 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 514.968677 | 0.469963 | 0.479733 | 0.48088132 | 4244.389935159304 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 494.640773 | 0.5150035 | 0.56941665 | 0.57399203 | 7645.9257423104345 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 501.508575 | 0.5078315 | 0.5571007499999999 | 0.5716723899999999 | 7763.495964340709 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 518.11303 | 0.4982065 | 0.54716385 | 0.5687114099999999 | 7935.533944186929 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 503.744269 | 0.5043635 | 0.5124398 | 0.51554011 | 7916.044020329193 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.617967 | 0.553104 | 0.6046971999999999 | 0.61865427 | 14337.027531322283 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.637768 | 0.5523525 | 0.6015264499999999 | 0.6164801 | 14375.831754178163 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.781221 | 0.5521240000000001 | 0.61314015 | 0.62434967 | 14220.4343660118 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.184496 | 0.551366 | 0.61073015 | 0.62113877 | 14293.866384010225 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.857301 | 0.9296735 | 0.97464905 | 0.9770847899999999 | 17156.914783534423 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 500.16494 | 0.9387975 | 0.9613363 | 0.96902105 | 17095.280975901213 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.323616 | 0.9622775 | 0.99049625 | 0.9958208 | 16613.658021805095 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.63264 | 0.9790895 | 1.01258135 | 1.0367954799999999 | 16281.559284118564 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 491.101193 | 1.4641030000000002 | 1.49642525 | 1.5092105 | 21814.774945283774 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.524171 | 1.5999615 | 1.68065745 | 1.69237824 | 19898.867983248732 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.380136 | 1.6934685 | 1.7417761 | 1.74415257 | 18854.74021779841 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.754683 | 1.7757945 | 1.80924145 | 1.8275359199999999 | 18033.08864385628 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.439716 | 1.457831 | 1.49779325 | 1.5047663299999998 | 43673.669685008565 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 510.674465 | 1.7618355 | 1.7790674 | 1.83319696 | 36288.739703863925 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.080091 | 1.7852190000000001 | 1.8526753 | 1.9075315 | 35690.79869906593 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 506.432026 | 1.964724 | 2.0225797 | 2.05203355 | 32581.46246333491 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 490.450542 | 1.554718 | 1.6114787 | 1.61421935 | 81698.39317387457 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 507.2948 | 1.872755 | 1.9262713 | 1.93320204 | 68266.53559491833 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 504.072103 | 1.930051 | 2.08273815 | 2.10767769 | 65153.871812098674 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 504.958982 | 1.977819 | 2.0677840499999998 | 2.07028613 | 64289.9786629597 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.328868 | 0.0501715 | 0.0552475 | 0.05835392 | 19681.637701200463 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 12.1001 | 0.052701 | 0.054753750000000004 | 0.05899889 | 18903.255783356588 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 12.258443 | 0.053024 | 0.0558462 | 0.05969514 | 18643.283514751125 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 23.697617 | 0.0548 | 0.05825175 | 0.06009928 | 18130.69621148224 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 12.269488 | 0.0521245 | 0.054039899999999995 | 0.058008069999999995 | 38188.86993023658 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 12.271396 | 0.0541175 | 0.05688665 | 0.06120277 | 36628.31925829119 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 12.455627 | 0.054492 | 0.058289999999999995 | 0.06101221 | 36424.32591321249 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 13.052905 | 0.055325 | 0.0576444 | 0.06187685999999999 | 35893.53546658187 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.0695 | 0.05428 | 0.057341300000000005 | 0.060220429999999985 | 73218.58272985967 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 12.017374 | 0.0570665 | 0.058941099999999996 | 0.06302284 | 69767.26338607041 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 12.424265 | 0.0574615 | 0.06270365 | 0.06425346 | 68787.49993551172 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 13.659718 | 0.057381 | 0.05976035 | 0.06173384 | 69321.08660803258 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 12.048753 | 0.060344999999999996 | 0.06338429999999999 | 0.0640298 | 132315.03681004324 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 12.388486 | 0.0614885 | 0.0643835 | 0.06884537 | 128899.78235271749 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 12.366563 | 0.061398 | 0.06476325 | 0.07006056 | 128921.21946581494 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 13.024036 | 0.061621999999999996 | 0.0651247 | 0.07006559999999999 | 128882.29727539602 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 12.08923 | 0.069278 | 0.07022205000000001 | 0.07175587 | 231387.07876705393 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 12.207717 | 0.069574 | 0.07135694999999999 | 0.07411055999999999 | 230571.60429992984 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 12.477404 | 0.06873850000000001 | 0.07229925 | 0.07487313 | 232661.6198483628 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 12.97764 | 0.07058400000000001 | 0.07498964999999999 | 0.07703478 | 226931.0413298159 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 11.960712 | 0.0854445 | 0.0906387 | 0.09295503 | 371043.1437372439 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 12.247675 | 0.087229 | 0.09323914999999999 | 0.09635681999999998 | 362591.6053698004 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 12.415438 | 0.086972 | 0.09190305 | 0.09449500999999999 | 364747.10145736986 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.095744 | 0.0868025 | 0.0928966 | 0.09624508999999999 | 364409.0765190734 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.066451 | 0.1184365 | 0.122096 | 0.12401601 | 538957.6256356542 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 12.220854 | 0.14246799999999998 | 0.14771525 | 0.14872139 | 449976.65746089414 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 12.417103 | 0.14932600000000001 | 0.1581078 | 0.16217739999999997 | 427973.0114869294 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 12.986004 | 0.152475 | 0.15943855 | 0.16023059 | 419887.8846861903 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 11.990613 | 0.182099 | 0.1856878 | 0.18820518 | 701135.2804371315 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 12.185138 | 0.2086585 | 0.2202084 | 0.22256510000000002 | 611643.6499108481 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 12.558797 | 0.22741850000000002 | 0.23320525 | 0.23460957 | 565284.9835665473 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.340696 | 0.23732999999999999 | 0.2441588 | 0.24740057 | 540131.3363102446 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 298.522807 | 0.6239684999999999 | 0.6993227 | 0.70851473 | 1606.1449822103384 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 293.368844 | 0.628436 | 0.7282569 | 0.77711896 | 1576.0339949271877 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 293.096014 | 0.6232705000000001 | 0.6932355 | 0.71393409 | 1599.524698036929 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 292.874815 | 0.6463859999999999 | 0.7090344999999999 | 0.74419737 | 1551.41152850804 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 289.870959 | 0.9507245 | 1.0578900500000001 | 1.09870076 | 2100.7439490591437 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 296.974297 | 0.9770004999999999 | 1.0677423999999998 | 1.09663749 | 2049.430459176534 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 291.307675 | 0.9504055 | 1.0634662499999998 | 1.09191784 | 2081.9359639385375 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 291.453515 | 0.9603155 | 1.0438486 | 1.06341727 | 2089.806290630406 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 292.429082 | 1.072986 | 1.1780195999999998 | 1.25435119 | 3715.552117761601 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 288.328109 | 1.0448745000000002 | 1.21460705 | 1.26083785 | 3780.908382521534 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 288.905473 | 1.0876255000000001 | 1.47176375 | 1.51714333 | 3476.6351345130997 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 291.307372 | 1.056135 | 1.18898245 | 1.25249982 | 3744.4873423315735 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 286.360897 | 1.2089495000000001 | 1.3346424 | 1.39054147 | 6694.6952356314305 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 289.15794 | 1.2124424999999999 | 7.334499699999989 | 18.827415029999965 | 3659.5485397943876 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 288.653303 | 1.245879 | 1.3726234 | 1.4127657399999998 | 6415.873692747691 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 290.421736 | 1.1908915 | 1.32655045 | 1.3548665199999999 | 6702.740031530025 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 290.283133 | 1.5438435 | 1.69888005 | 1.7530231299999999 | 10335.862561968343 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 291.733684 | 1.4361855000000001 | 1.5912638 | 1.6499063399999998 | 11087.458328654078 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 289.261147 | 1.477338 | 1.67813905 | 1.7773264999999996 | 10740.28069374534 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 289.356182 | 1.5213785 | 1.65596915 | 1.7204843399999998 | 10645.708531252581 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 290.7389 | 1.3739645 | 1.5384181 | 1.6352187699999998 | 23189.877178004135 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 298.020209 | 1.431229 | 3.4161677999999913 | 13.868896439999999 | 16025.182131453186 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 291.858536 | 1.4122184999999998 | 1.57156735 | 1.62177861 | 22936.695180037 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 290.782996 | 1.374971 | 1.50268105 | 1.52652619 | 23140.469476158778 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 290.959747 | 3.218029 | 3.478756 | 3.54052773 | 20011.926858258434 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 290.449013 | 2.993595 | 3.32588995 | 3.4581321299999996 | 21133.550972364916 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 292.257793 | 3.0856315 | 3.3569953 | 3.52776515 | 20665.784142965585 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 289.045191 | 2.9814095 | 3.2287095999999997 | 3.3578177699999996 | 21370.225894039595 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 291.18525 | 3.2638135 | 3.63525375 | 3.65180458 | 39274.31040232763 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 281.752565 | 3.0470905 | 3.3689413999999998 | 3.40468175 | 41762.72228739128 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 291.02765 | 3.229775 | 3.5448409 | 3.58779157 | 39633.853052461265 | - |
| `separator1_grid_search_6ports_learned_dense_hd16_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 287.618877 | 3.1977905 | 3.40644265 | 3.4849008799999996 | 40104.07207223104 | - |
