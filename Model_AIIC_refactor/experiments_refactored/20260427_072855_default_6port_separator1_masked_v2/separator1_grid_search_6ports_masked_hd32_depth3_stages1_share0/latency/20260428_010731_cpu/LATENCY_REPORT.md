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

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`300035.840` samples/s, p50=`0.426` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.274` ms, throughput=`3638.052` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`612720.929` samples/s, p50=`0.208` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.085` ms, throughput=`11728.403` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`375186.435` samples/s, p50=`0.337` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.255` ms, throughput=`3909.980` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1003460.685` samples/s, p50=`0.127` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`32078.960` samples/s

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`68923.798` samples/s, p50=`1.866` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.369` ms, throughput=`2903.982` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,024`
- MACs / sample: `26,112`
- FLOPs / sample estimate: `53,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.2791715 | 0.29691039999999996 | 0.32079129 | 3540.3859615482925 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.286158 | 0.30645924999999996 | 0.35425748 | 3437.662257658562 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.2899965 | 0.3470818 | 0.34952024 | 3355.334034911311 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2857815 | 0.34173555 | 0.34902084 | 3430.5905761676872 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.30393349999999997 | 0.3498186 | 0.35701603 | 6482.72412365563 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.303309 | 0.3359174499999999 | 0.36170092 | 6506.811655781856 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.3105745 | 0.32700395 | 0.36222014999999996 | 6376.771411301309 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.31103349999999996 | 0.3182956 | 0.34790620999999994 | 6399.478621677736 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.3089015 | 0.3251243 | 0.38145249 | 12791.96623586096 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.31072299999999997 | 0.32212504999999997 | 0.32699909 | 12828.013919421344 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.30543750000000003 | 0.31373799999999996 | 0.31917178999999996 | 13063.777951830845 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.3072785 | 0.3490107 | 0.35400096999999997 | 12798.339696987789 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.316786 | 0.3263546 | 0.33400726 | 25154.783672633635 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.3141705 | 0.3618679 | 0.36598354 | 25031.473949557323 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.308921 | 0.3602915 | 0.36186436 | 25415.69271614204 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3187765 | 0.36654945 | 0.36992727 | 24602.886349119264 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.337366 | 0.34531055 | 0.35117009 | 47304.439308759145 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.3265935 | 0.33576849999999997 | 0.34492455 | 48795.94183670121 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.31491899999999995 | 0.32046939999999996 | 0.32344027 | 50707.837035660414 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.317707 | 0.3692064 | 0.37520864 | 49643.09407782089 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.3346405 | 0.40316785 | 0.40704923 | 93777.01583952407 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.327166 | 0.3354853 | 0.34512091 | 97529.98605138331 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.3340145 | 0.34778775 | 0.36180421999999995 | 95363.4584511842 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.3392355 | 0.37927479999999986 | 0.40564261 | 93059.58692594232 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.35931749999999996 | 0.3723883 | 0.37590212 | 177402.07959587808 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.357479 | 0.4136507999999998 | 0.44654478 | 175555.5565308642 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3584575 | 0.44378504999999996 | 0.44808887 | 174641.51690565358 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3619615 | 0.4260353999999999 | 0.45061531 | 173503.59385584734 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.43516350000000004 | 0.45475925 | 0.5099709799999999 | 291280.07580563973 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4256685 | 0.43937725 | 0.44357203 | 300035.84021872614 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.5301485 | 0.56931345 | 0.57791597 | 240509.84780724728 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.4355985 | 0.45327455 | 0.45595446 | 293021.7645120517 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.281419 | 0.28872275 | 0.29326447 | 3539.5556314436285 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2748505 | 0.29944449999999995 | 0.30877848999999996 | 3600.124391497975 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.27446499999999996 | 0.27848269999999997 | 0.28145159999999997 | 3638.051692348886 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2766765 | 0.28539404999999995 | 0.30657254999999994 | 3595.9516776013566 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.2984175 | 0.30183205 | 0.30864853 | 6695.900521952141 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.29405499999999996 | 0.3019851 | 0.30475969 | 6782.018400565403 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.2962855 | 0.3023703 | 0.3382784699999999 | 6707.996421418069 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.292673 | 0.2974609 | 0.30111331 | 6835.706394379518 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.3248805 | 0.34344684999999997 | 0.35660888999999996 | 12229.592997922071 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.32795549999999996 | 0.33498045 | 0.34524542999999996 | 12165.309525011573 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.3280015 | 0.33627465 | 0.33954219999999996 | 12163.03231206838 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.3304465 | 0.3358992 | 0.36024688999999993 | 12048.508500072141 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.6767965 | 0.7158897999999999 | 0.7493842599999999 | 11742.4184268136 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.711894 | 0.7435817 | 0.7461992900000001 | 11181.268602136806 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.7443960000000001 | 0.76162785 | 0.76603244 | 10737.963595135714 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.6945589999999999 | 0.7211368 | 0.7246283600000001 | 11478.571100407282 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8472175 | 0.87165075 | 0.873719 | 18910.57858711692 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.8906350000000001 | 0.92012335 | 0.92453555 | 17920.886008604266 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8462855 | 0.8814033499999999 | 0.88774796 | 18890.52097742767 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9074059999999999 | 0.9567928 | 0.96482609 | 17539.432865507693 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8507795 | 0.8854601 | 0.9177124 | 37416.083616902426 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.911046 | 0.9678365999999999 | 0.98789575 | 34760.87710209344 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.8656675 | 0.9168483 | 0.92261172 | 36735.85911651085 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.95296 | 0.9970437 | 1.0048354 | 33455.15960003437 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8729155 | 0.89661435 | 0.92442631 | 73025.74554607991 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.0000455 | 1.1008504000000001 | 1.10591096 | 62709.86124522149 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0824729999999998 | 1.1645010999999998 | 1.2841367399999997 | 58302.54366892405 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.0832775 | 1.1605287499999999 | 1.16890984 | 58587.94341415309 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9528840000000001 | 1.00878655 | 1.02474427 | 133033.06661589237 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.230472 | 1.2716968 | 1.2761851100000001 | 105029.87410659784 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.259153 | 1.3073977 | 1.31635134 | 102021.41891867659 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.2876915 | 1.3388472999999999 | 1.3707587799999998 | 99986.32374565516 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 183.020647 | 0.08481 | 0.08720185 | 0.08977782 | 11728.403025458845 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 178.684781 | 0.0857 | 0.0878542 | 0.08909998 | 11636.206568731699 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 178.333618 | 0.08494299999999999 | 0.08690935 | 0.09098772 | 11714.704766010487 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 181.162244 | 0.0874605 | 0.08940825 | 0.08981513 | 11412.757911209657 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 181.554066 | 0.09715750000000001 | 0.09942085 | 0.10102198 | 20499.28467746118 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 179.412582 | 0.095112 | 0.1024577 | 0.10510452 | 20836.688908442964 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 181.438099 | 0.09518399999999999 | 0.099825 | 0.10548405999999998 | 20856.405732174553 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 178.824751 | 0.09670200000000001 | 0.0994967 | 0.10220822 | 20610.93702884686 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 182.097652 | 0.09888250000000001 | 0.10133924999999999 | 0.10276866999999999 | 40348.66898819458 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 182.596204 | 0.09719 | 0.10084585 | 0.10119516 | 40974.4964490477 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 182.085181 | 0.0978265 | 0.1010244 | 0.1016933 | 40768.9513451104 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 179.608592 | 0.099575 | 0.1048262 | 0.10814288999999999 | 39816.580938249856 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 180.10963 | 0.101795 | 0.1068613 | 0.10859104 | 78033.12818423934 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 181.821164 | 0.10045699999999999 | 0.10300785 | 0.10544946 | 79293.73074029619 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 179.686746 | 0.1027735 | 0.11095265 | 0.11375431 | 77049.48252604417 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 178.430231 | 0.1026115 | 0.10515455 | 0.10712021999999999 | 77649.7742818124 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 182.486305 | 0.117974 | 0.12284505 | 0.12404775 | 135163.92596463536 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 182.84561 | 0.1193925 | 0.12133585 | 0.12424985999999999 | 133838.9769883955 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 182.50576 | 0.1143865 | 0.119874 | 0.12508872 | 139060.66260667474 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 179.032175 | 0.1172035 | 0.11902675 | 0.12351616 | 136170.0968765133 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 183.168524 | 0.1333805 | 0.13540359999999999 | 0.13854277 | 239708.63415518438 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 179.767364 | 0.13250800000000001 | 0.1379581 | 0.13880356 | 240400.18216323806 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 182.762659 | 0.1321775 | 0.1347104 | 0.13665347 | 241744.64694251947 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 183.64562 | 0.13357750000000002 | 0.14020755 | 0.14375512999999998 | 238135.06902047305 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 178.833748 | 0.1558405 | 0.15732315 | 0.16707740999999995 | 410133.9484659004 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 179.427341 | 0.15673399999999998 | 0.16046915 | 0.16271336 | 407383.4170506256 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 183.27942 | 0.15744000000000002 | 0.1618302 | 0.16282677 | 405293.4875795911 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 181.407654 | 0.15559 | 0.1581308 | 0.16011746 | 410763.9670659721 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 180.481986 | 0.20810250000000002 | 0.2134418 | 0.22584965999999998 | 612720.9289768365 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 182.480883 | 0.2099385 | 0.2163518 | 0.21874658 | 608099.1201565855 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 182.539379 | 0.278678 | 0.28324954999999996 | 0.28485137 | 459862.5830007054 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 183.22358 | 0.2103875 | 0.21695979999999998 | 0.220391 | 605431.8398685758 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 181.254352 | 0.0933335 | 0.0955469 | 0.09646671 | 10681.521672487028 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 177.233282 | 0.0928985 | 0.0951774 | 0.09814487 | 10730.75358863227 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 178.26559 | 0.09288 | 0.09691404999999999 | 0.09944639 | 10720.715080272426 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 177.346266 | 0.0943865 | 0.09753415 | 0.09914102 | 10549.079592805527 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 179.908595 | 0.1095775 | 0.11142185 | 0.11181724000000001 | 18204.95502465861 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 178.364481 | 0.109359 | 0.11276929999999999 | 0.11423973 | 18232.80705551288 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 180.985053 | 0.11474300000000001 | 0.11794125 | 0.12097066 | 17362.55799094369 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 181.987079 | 0.111393 | 0.11377359999999999 | 0.11878246999999999 | 17893.063181479676 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 179.109923 | 0.14185150000000002 | 0.1469852 | 0.14937204999999998 | 28090.550451190422 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 182.418131 | 0.141249 | 0.14392015 | 0.14543288 | 28255.447897589824 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 177.766685 | 0.1414845 | 0.1434867 | 0.14631566 | 28211.032629585614 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 178.120021 | 0.14246799999999998 | 0.14843784999999998 | 0.1499625 | 27902.828956216552 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 186.023994 | 0.3539355 | 0.35926790000000003 | 0.36250956 | 22593.936408801128 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 188.700574 | 0.395167 | 0.4334753999999999 | 0.45280959 | 20076.620414148554 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 190.542679 | 0.41498199999999996 | 0.4587903 | 0.47190582999999997 | 19035.453579881163 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 202.766375 | 0.524544 | 0.54350195 | 0.54861874 | 15197.388815048573 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 192.478648 | 0.45660199999999995 | 0.5520323 | 0.55590471 | 34156.818650989255 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 187.569284 | 0.46143449999999997 | 0.49196049999999997 | 0.49710724 | 34270.8935929879 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 192.543557 | 0.490512 | 0.57079145 | 0.57753193 | 32183.470239804265 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 189.271213 | 0.5045805 | 0.5296145 | 0.53198777 | 31554.538272302256 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 189.417518 | 0.49430549999999995 | 0.61297285 | 0.62560708 | 63325.40960952801 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 190.028196 | 0.556241 | 0.66247 | 0.66834829 | 56584.50549315306 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 188.513796 | 0.5446679999999999 | 0.65200225 | 0.6603171099999999 | 57900.185363824676 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 193.500456 | 0.5399594999999999 | 0.6285422999999999 | 0.6470445699999999 | 57670.88143490344 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 189.604869 | 0.5415525 | 0.6714118 | 0.67368017 | 114347.1881508298 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 188.753851 | 0.696465 | 0.7790669499999999 | 0.7945149399999999 | 89826.52308870072 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 191.675921 | 0.6913925000000001 | 0.7672123 | 0.7843126 | 91263.13743558284 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 195.546148 | 0.7547135 | 0.8163479499999999 | 0.82906833 | 84077.71008471145 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 189.203712 | 0.6273875 | 0.7528475 | 0.7613242299999999 | 197541.13755607352 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 191.434555 | 0.8439255 | 0.87612485 | 0.90403569 | 150792.13464455892 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 190.528014 | 0.86297 | 0.8986107 | 0.9167586799999999 | 148714.44379973118 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 192.153858 | 0.9771255 | 1.0196379999999998 | 1.0692457499999999 | 130935.55417491288 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.749325 | 0.25926950000000004 | 0.26499635 | 0.26984879 | 3843.1742036308924 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 509.016823 | 0.2594115 | 0.2643793 | 0.31429561999999983 | 3819.43746866629 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 512.661857 | 0.263237 | 0.2685611 | 0.26940264 | 3794.482049026378 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 544.99864 | 0.255482 | 0.26038195 | 0.26476536 | 3909.9800598836905 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 498.116942 | 0.263043 | 0.2682557 | 0.26861983 | 7596.272691300924 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 518.646373 | 0.258274 | 0.2637989 | 0.26722102 | 7725.693981707565 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 511.960838 | 0.2658325 | 0.26909515 | 0.27142290999999996 | 7517.957016231119 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 545.167644 | 0.26139999999999997 | 0.2657173 | 0.2683852 | 7648.987358212634 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 497.038344 | 0.263018 | 0.2674216 | 0.27011367999999997 | 15198.948536740227 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 518.910771 | 0.2617785 | 0.28693759999999996 | 0.30249789 | 15110.60091006616 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 516.833141 | 0.260532 | 0.26582505 | 0.2707788 | 15336.520057714393 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 535.685431 | 0.26568749999999997 | 0.27093625000000005 | 0.27177172 | 15050.20447207796 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.924923 | 0.267228 | 0.27138625 | 0.27265614 | 29925.54897484421 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 504.012583 | 0.267991 | 0.27328765 | 0.2760079 | 29796.006113544536 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 513.873412 | 0.26109950000000004 | 0.2667066 | 0.2674701 | 30523.56304603793 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.345264 | 0.26241000000000003 | 0.26856045 | 0.27322198 | 30414.786209145146 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 500.332042 | 0.27529950000000003 | 0.32039455 | 0.33149041 | 56523.92421017885 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 507.989185 | 0.269156 | 0.27577325 | 0.27805073 | 59348.39615410523 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 516.281596 | 0.27387300000000003 | 0.2807 | 0.31139813 | 58041.82130882131 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 535.541593 | 0.26826 | 0.31657185 | 0.32675911 | 58312.25102491071 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 496.558341 | 0.285117 | 0.34174135 | 0.35206566999999994 | 108652.62661952683 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.200888 | 0.277242 | 0.32691985 | 0.3402768 | 113015.6444731253 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 510.956787 | 0.27905 | 0.3276602 | 0.33802307 | 112446.85480524768 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 548.090065 | 0.2839815 | 0.33930065 | 0.35396836 | 109750.98393471955 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.406225 | 0.297631 | 0.3656543 | 0.37778150999999993 | 206732.60920068645 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.369768 | 0.3071015 | 0.3758786 | 0.38699677 | 203708.83768511645 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 517.24291 | 0.294729 | 0.32359074999999987 | 0.36266659999999995 | 214680.87084758555 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 538.08188 | 0.299928 | 0.37091035 | 0.38185725 | 204397.84968352184 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.548736 | 0.33704350000000005 | 0.3643179 | 0.42561752 | 375186.43541581737 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 518.418533 | 0.3515175 | 0.36660535 | 0.36919502 | 364558.6140757448 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.021911 | 0.4300295 | 0.5094495 | 0.5464067599999999 | 290074.58588113624 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.69251 | 0.3329385 | 0.3978039999999999 | 0.4518135699999999 | 374962.2474534058 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.430558 | 0.2667615 | 0.290151 | 0.3455446499999998 | 3679.8021620604004 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 505.934578 | 0.268712 | 0.2736565 | 0.27557693 | 3715.2371759115017 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 504.512495 | 0.269693 | 0.27477475 | 0.27521729 | 3704.442945873791 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 513.349744 | 0.2743915 | 0.28025045 | 0.28110406 | 3641.3937915401098 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 497.652731 | 0.27542999999999995 | 0.2796456 | 0.28124897 | 7243.105866973698 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 508.981646 | 0.277362 | 0.2822928 | 0.28470299 | 7194.4159245806495 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 509.4663 | 0.27940200000000004 | 0.28692365 | 0.29371073 | 7129.787224333907 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 499.542097 | 0.27682249999999997 | 0.2805376 | 0.2863518 | 7210.563302184181 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 493.981311 | 0.306514 | 0.312095 | 0.31258266 | 13041.40155942559 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 510.918742 | 0.30761499999999997 | 0.31527805000000003 | 0.31618852000000003 | 12982.98657507256 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.415834 | 0.30767350000000004 | 0.31567409999999996 | 0.31699056 | 12962.302965703628 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 506.248444 | 0.314122 | 0.33271554999999997 | 0.34293167999999996 | 12640.714058768324 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 492.773612 | 0.531814 | 0.57324775 | 0.58494726 | 14933.883218825295 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 506.894713 | 0.5695815 | 0.57843155 | 0.58143537 | 14037.751444203868 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 505.252017 | 0.559673 | 0.56666105 | 0.56907153 | 14285.086252100844 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.897061 | 0.5412955 | 0.55017395 | 0.55146338 | 14748.97702017199 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.604951 | 0.7298830000000001 | 0.8087064499999997 | 0.8547898199999999 | 21654.484936165827 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 513.492655 | 0.759447 | 0.80103585 | 0.81718735 | 20863.80317922633 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 502.485759 | 0.7523245000000001 | 0.8155502499999998 | 0.86547507 | 21004.345352711138 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.461171 | 0.7807120000000001 | 0.8181493 | 0.9167254499999998 | 20302.946868685387 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.523938 | 0.7515585 | 0.78046545 | 0.83012405 | 42277.26934076737 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 516.537311 | 0.7921024999999999 | 0.8442594999999999 | 0.87197709 | 40011.96157591312 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 504.398398 | 0.7642015 | 0.7917284 | 0.7985438699999999 | 41665.92449238664 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.735895 | 0.802488 | 0.8273758 | 0.8787447399999998 | 39676.22907596567 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.970636 | 0.7589985 | 0.78513215 | 0.8366402099999999 | 83820.83882447142 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 506.074797 | 0.8743175 | 0.9332770499999999 | 0.9964552599999998 | 72059.58012188743 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 508.749568 | 0.9243835 | 0.9666755 | 1.00501296 | 69095.32561701207 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 498.219131 | 0.9094275 | 0.96247965 | 1.0105649799999998 | 69661.77011907569 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 496.189843 | 0.7987869999999999 | 0.84026835 | 0.8431072 | 158852.58395818248 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 517.052822 | 0.9476635 | 1.03108735 | 1.04155302 | 133019.99365328357 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 508.21257 | 1.0351675 | 1.07386495 | 1.0927817899999999 | 124307.02716947615 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 511.439753 | 1.0673110000000001 | 1.17422455 | 1.2200471299999998 | 117873.95752456144 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.599908 | 0.031071 | 0.0317123 | 0.034368619999999996 | 32078.960436376514 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.144941 | 0.032235 | 0.03478809999999999 | 0.03916185999999999 | 30711.247933133014 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.329115 | 0.032118 | 0.03605165 | 0.03933722 | 30691.736495482477 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.7135 | 0.031896999999999995 | 0.033098949999999995 | 0.03659976999999999 | 31162.627782355223 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.195913 | 0.031447 | 0.032638799999999996 | 0.03520469999999999 | 63390.357185645626 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.114317 | 0.0326965 | 0.03742254999999999 | 0.040325049999999994 | 60349.192497629796 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.289274 | 0.032975000000000004 | 0.03374195 | 0.03851551 | 60446.99342699393 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.718608 | 0.032898 | 0.033398199999999996 | 0.045897199999999985 | 59923.26227033661 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.007806 | 0.033655 | 0.03505445 | 0.036002669999999994 | 119199.66957851592 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.928416 | 0.0346625 | 0.03636995 | 0.04153453999999999 | 114380.94459215473 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.178975 | 0.0344805 | 0.03868654999999999 | 0.041310419999999994 | 115178.37963600176 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.760778 | 0.034524 | 0.035617949999999995 | 0.040630059999999996 | 114970.28018257281 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.994895 | 0.037063 | 0.03773975 | 0.03852447 | 215592.28050280432 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.109331 | 0.0389445 | 0.04300369999999999 | 0.045940049999999996 | 204881.9265457317 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.392942 | 0.0383405 | 0.0390161 | 0.04968267999999997 | 206396.96431344887 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.004897 | 0.038526500000000005 | 0.04120969999999999 | 0.04811838 | 206925.3785699801 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.967491 | 0.044084 | 0.0455452 | 0.047395889999999996 | 363156.33698729135 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.382467 | 0.0451465 | 0.04866989999999999 | 0.052198470000000004 | 354721.8803097431 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.466844 | 0.045533500000000005 | 0.0469143 | 0.056154589999999976 | 348589.8017439077 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.864359 | 0.0454535 | 0.051637999999999996 | 0.05446984 | 346042.46373583114 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.005776 | 0.057186 | 0.058528199999999996 | 0.05904854 | 563895.1338316991 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.126479 | 0.0581075 | 0.0615439 | 0.06411991999999998 | 547130.1314822098 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.395398 | 0.058514 | 0.06133035 | 0.06508916 | 548495.0153801431 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.882313 | 0.0589225 | 0.06288414999999999 | 0.0672362 | 542937.9048704924 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.99469 | 0.07925650000000001 | 0.08189725 | 0.08368608 | 805189.2433762491 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.135258 | 0.1224085 | 0.13416134999999998 | 0.13594641000000002 | 521024.0532382377 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.396574 | 0.1272795 | 0.14554525 | 0.14927732999999999 | 496901.2770518102 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.900388 | 0.12165300000000001 | 0.137496 | 0.14270845 | 515833.42642514704 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.900198 | 0.127248 | 0.13080665 | 0.132442 | 1003460.6850375233 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.125958 | 0.270442 | 0.28435204999999997 | 0.28708355 | 479855.5874609536 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.234751 | 0.2669825 | 0.2874364 | 0.2969134 | 475787.9680657063 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.879979 | 0.2595525 | 0.2750026 | 0.29174554999999996 | 494618.5119479353 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 267.183024 | 0.3745095 | 0.5439785499999995 | 0.8636041399999999 | 2482.9940975754107 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 262.937897 | 0.3691695 | 0.5439905999999998 | 0.6342050199999999 | 2903.9817597420893 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 261.505023 | 0.38952149999999996 | 0.4334299 | 0.44933905999999996 | 2539.3028212060635 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 260.600314 | 0.3840565 | 0.43674485 | 0.44998929 | 2585.4918743161375 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 260.109879 | 0.5810995000000001 | 0.6962602999999998 | 0.7370928999999999 | 3410.8225604492404 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 257.687158 | 0.6706970000000001 | 0.77733625 | 0.78125889 | 2961.1236445604573 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 259.099981 | 0.565637 | 0.7128159 | 1.058927549999999 | 3429.8957603230006 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 258.196853 | 0.591663 | 0.6935528 | 0.72103715 | 3334.1116261239254 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 257.134515 | 0.6194035 | 0.70988695 | 0.72751922 | 6368.229970539613 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 260.309058 | 0.6096820000000001 | 0.8978429499999999 | 1.1636824299999997 | 6846.840826561579 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 259.869013 | 0.662465 | 0.75104775 | 0.7709579799999999 | 5943.433608369686 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 263.13822 | 0.633211 | 0.7610264499999999 | 0.8131963699999999 | 6194.644655359705 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 257.536743 | 0.704124 | 0.8165235999999999 | 0.8391675799999999 | 11213.40854540225 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 266.341539 | 0.7093035 | 0.85916795 | 0.9242192499999998 | 11011.376183076021 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 262.00041 | 0.7374025 | 1.27765965 | 1.39594038 | 9794.168689080883 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 260.207477 | 0.49038800000000005 | 0.8345727999999999 | 1.0467163599999998 | 13789.061351360417 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 259.859272 | 0.8868775 | 1.0239958 | 1.06724083 | 17816.1640890092 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 261.336721 | 0.9358315 | 1.07862735 | 1.12313136 | 16936.802312246124 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 260.388006 | 0.9574995 | 1.0494052 | 1.0646440099999999 | 16890.96174771457 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 262.411844 | 0.9723055 | 1.1177082 | 1.15106829 | 16503.896982675033 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 259.181464 | 1.5431835 | 1.7225493 | 1.88461097 | 20637.53179453619 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 260.368473 | 1.5147775 | 1.7591689499999998 | 1.93275329 | 21110.32359342145 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 258.718153 | 1.496477 | 1.68068375 | 1.8023330899999996 | 21360.680415637722 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 262.175552 | 1.475818 | 1.5947505999999998 | 1.6666369899999998 | 21926.965881312186 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 257.259814 | 1.9016255000000002 | 2.1800246999999997 | 2.2758828999999996 | 33846.790556356194 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 257.879182 | 2.1668085 | 2.6511533999999997 | 2.7053227599999996 | 29214.927253872527 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 258.000224 | 1.805552 | 2.05992125 | 2.1907626199999997 | 35055.645688732904 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 258.012811 | 1.9307565 | 2.2505244500000003 | 2.3302219699999998 | 33344.52806395314 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 263.299208 | 1.9864665000000001 | 2.22906045 | 2.23628963 | 64483.10224335907 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 261.377306 | 1.866097 | 2.03664565 | 2.1123881 | 68923.79829015349 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 262.33317 | 2.0212845 | 2.3396866 | 2.38989396 | 63119.00450205175 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 256.487476 | 1.8803845 | 2.1259442 | 2.36361592 | 67255.8468003446 | - |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd32_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
