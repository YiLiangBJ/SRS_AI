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

### separator1_grid_search_6ports_masked_depth2_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`363241.483` samples/s, p50=`0.351` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.208` ms, throughput=`4778.013` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`682711.088` samples/s, p50=`0.188` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.067` ms, throughput=`14705.679` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`441436.027` samples/s, p50=`0.286` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.200` ms, throughput=`4978.376` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`947470.461` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`36457.126` samples/s

### separator1_grid_search_6ports_masked_depth2_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`90988.997` samples/s, p50=`1.413` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.331` ms, throughput=`3008.071` samples/s

## Run References

### separator1_grid_search_6ports_masked_depth2_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_depth2_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,560`
- MACs / sample: `27,648`
- FLOPs / sample estimate: `56,568`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.21550049999999998 | 0.22411335 | 0.23140182 | 4620.230455247016 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.2132975 | 0.22081725 | 0.22176114 | 4676.550587510374 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.210553 | 0.2492165 | 0.2972031499999998 | 4609.647235682136 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.2169105 | 0.22428309999999999 | 0.2290902 | 4591.7768073210555 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.2204625 | 0.25282155 | 0.26409918 | 8859.273714968993 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.2257055 | 0.23231179999999998 | 0.2605491499999999 | 8790.433687473536 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.22352349999999999 | 0.22917215000000002 | 0.23171012 | 8931.418318160342 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.2279435 | 0.23942575 | 0.25882949999999993 | 8693.849901029213 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.23507499999999998 | 0.2429507 | 0.24851453 | 16975.293055093567 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.226572 | 0.2395688 | 0.25525806999999995 | 17496.329488777374 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.23110350000000002 | 0.24182084999999998 | 0.24605121 | 17225.19653518617 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.223725 | 0.23537529999999998 | 0.24126151999999998 | 17724.401333973896 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.237728 | 0.27191665 | 0.27313418 | 32626.99159235054 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.23587750000000002 | 0.24805115 | 0.27087412999999994 | 33609.05078293964 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.2332565 | 0.24659015 | 0.25329515999999996 | 33980.21468020031 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.238178 | 0.2466072 | 0.24874592999999998 | 33450.76503154031 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.24056 | 0.28342275 | 0.28461417 | 65049.2813355398 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.241511 | 0.2567259 | 0.28581469 | 65497.62681630036 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.2413615 | 0.25749215 | 0.27377348999999995 | 65656.76171495134 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.2385515 | 0.28033095 | 0.28531964 | 65225.26441914538 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2598245 | 0.2896216 | 0.32133026 | 120841.15716887901 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.25383500000000003 | 0.26408115 | 0.27092841 | 125284.74874300246 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.26055550000000005 | 0.2753755 | 0.31692598 | 121009.15585525491 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.2592755 | 0.30218369999999994 | 0.31774526 | 120693.05876869394 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.2845985 | 0.34792995 | 0.35112685 | 218548.55756244596 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.2973175 | 0.31544135 | 0.33980009999999994 | 212721.5603073268 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.28264599999999995 | 0.34119229999999995 | 0.34869549 | 220890.8029076961 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.284673 | 0.29771585 | 0.3016781 | 223326.9150108488 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.3509835 | 0.3636516 | 0.38884024999999994 | 363241.4829946513 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.43804350000000003 | 0.49538365 | 0.5256127499999999 | 285633.7854425224 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.44414549999999997 | 0.4847976 | 0.49056008999999995 | 283985.8892736355 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.372054 | 0.38433174999999997 | 0.39536692999999995 | 341709.17697903136 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.20876650000000002 | 0.21479395 | 0.21669857 | 4770.92863263277 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.212766 | 0.22136535 | 0.22196955 | 4679.453241452581 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.21195350000000002 | 0.21691259999999998 | 0.22366424 | 4704.641213240404 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.2079545 | 0.21871685 | 0.22090444 | 4778.013036521889 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.231519 | 0.23890669999999997 | 0.3109244099999999 | 8511.779962997589 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.231155 | 0.23724865 | 0.23848195 | 8646.77832712682 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.22545949999999998 | 0.2307438 | 0.23314979 | 8859.874179154807 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.226472 | 0.23349655 | 0.25443426999999996 | 8746.592327629156 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.4278865 | 0.434595 | 0.45000329999999994 | 9328.539645570532 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.4720745 | 0.4973039 | 0.50722896 | 8409.436278905177 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.5839399999999999 | 0.59446185 | 0.59943006 | 6842.098060949409 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.6158545 | 0.6330137 | 0.6580765099999999 | 6471.589656691229 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.5636615 | 0.5742371 | 0.57621137 | 14181.00126590253 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6046045 | 0.6237066 | 0.6567607999999999 | 13185.081159778198 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.6174394999999999 | 0.6596634 | 0.66341284 | 12821.253330080153 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.794879 | 0.8166897000000001 | 0.83081239 | 10048.244132139434 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.5797695 | 0.62643495 | 0.62947725 | 27176.04724936279 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.6490145 | 0.68324065 | 0.69119117 | 24459.76855738945 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.743365 | 0.76296095 | 0.7723024000000001 | 21501.213085003998 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.767109 | 0.78758145 | 0.7934369699999999 | 20852.64972925832 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.5818035 | 0.59910705 | 0.6040299299999999 | 54913.11029915976 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.6646555000000001 | 0.71128005 | 0.71609374 | 47794.43470251146 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.809858 | 0.85214415 | 0.85698317 | 39274.168206254966 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.8792985 | 0.91712205 | 0.9492390699999999 | 36212.72231244848 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.6204704999999999 | 0.64446595 | 0.65278654 | 102642.86780580148 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.7663745 | 0.82102055 | 0.8359337299999999 | 82415.8835805591 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0669055 | 1.10009935 | 1.10499131 | 59933.242982720934 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.1110069999999999 | 1.1422586 | 1.15082965 | 57596.75868480825 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.7124459999999999 | 0.7700221 | 0.7823191899999999 | 177055.0195153362 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.85632 | 0.89872725 | 0.90591473 | 147940.14498920142 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.2239035 | 1.25427045 | 1.26067789 | 104439.36832850697 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.21801 | 1.2510676 | 1.2562150699999999 | 104939.56948312065 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 133.62324 | 0.06803000000000001 | 0.0736519 | 0.07500622 | 14546.630971173812 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 132.188967 | 0.0674575 | 0.07197245 | 0.07739566999999999 | 14705.679068554053 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 130.583087 | 0.068163 | 0.06978975 | 0.07246313 | 14602.543587862483 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 135.030342 | 0.06860050000000001 | 0.07430234999999999 | 0.07633304999999999 | 14424.839804941545 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 133.244256 | 0.074577 | 0.0779078 | 0.08010486 | 26677.76461674723 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 133.138424 | 0.0749225 | 0.07699085 | 0.07857112 | 26608.426143523724 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 130.998981 | 0.07446900000000001 | 0.08157665 | 0.08205833 | 26595.21419120629 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 130.703571 | 0.07710349999999999 | 0.0791491 | 0.08317093 | 25900.361310040276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 132.745715 | 0.0776645 | 0.0793328 | 0.08220045 | 51482.781326371485 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 133.692212 | 0.0775225 | 0.08141949999999999 | 0.08782023999999998 | 51326.58696674243 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 133.89141 | 0.07766300000000001 | 0.0801783 | 0.08093705 | 51290.00780890369 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 134.452912 | 0.075984 | 0.08039035 | 0.08306456 | 52182.22136063576 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 134.160438 | 0.0859185 | 0.08849355 | 0.08933358999999999 | 92676.47338215838 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 133.657892 | 0.08706549999999999 | 0.09121125 | 0.09255185 | 91648.25491412215 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 133.352689 | 0.0853955 | 0.09033535 | 0.09181973 | 93022.84478277886 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 134.365484 | 0.0856845 | 0.08887115 | 0.09251543 | 92811.75293789803 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 135.698772 | 0.092182 | 0.09389365000000001 | 0.09447291 | 173621.3972746215 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 133.22679 | 0.0931245 | 0.09917785 | 0.10050442 | 170554.71000689254 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 133.991333 | 0.0916025 | 0.09359574999999999 | 0.09476743 | 174511.96094073288 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 131.040955 | 0.0934095 | 0.0950966 | 0.09584431 | 170942.25292914893 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 132.149112 | 0.10918900000000001 | 0.11162715000000001 | 0.112903 | 292524.5711498373 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 133.938753 | 0.108667 | 0.11311575 | 0.11450764999999999 | 293534.35544442025 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 133.870685 | 0.1090735 | 0.110676 | 0.1118383 | 293019.0947724478 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 133.787855 | 0.1089315 | 0.1112016 | 0.11153203 | 293561.3913929268 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 132.085501 | 0.134124 | 0.1382607 | 0.13882843 | 475522.1270104741 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 134.435275 | 0.13448500000000002 | 0.1392918 | 0.14166257 | 472987.32438313216 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 134.337362 | 0.1352255 | 0.13710044999999998 | 0.13945083 | 472469.4279059638 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 135.195761 | 0.1354085 | 0.1413048 | 0.14166563 | 469919.4293768369 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 132.72148 | 0.18751600000000002 | 0.18999885 | 0.19094301 | 682711.0884014852 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 134.189052 | 0.2861605 | 0.32023085 | 0.32752742 | 440347.95469727763 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 137.327495 | 0.437971 | 0.4713196 | 0.48136139 | 289760.6246731987 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 137.366179 | 0.1965535 | 0.2060081 | 0.21004823 | 647501.5646268276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 133.369901 | 0.07649 | 0.0782947 | 0.07869116999999999 | 13018.988976561654 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 131.777899 | 0.07540350000000001 | 0.08073179999999999 | 0.08276255 | 13166.602018650756 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 133.135828 | 0.0781505 | 0.08224485 | 0.08507488999999999 | 12721.225288065785 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 132.342888 | 0.076059 | 0.07770329999999999 | 0.08066347 | 13098.64405456476 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 133.30866 | 0.093948 | 0.0952997 | 0.09587111999999999 | 21256.37505258296 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 131.126249 | 0.09329799999999999 | 0.0951191 | 0.09539959999999999 | 21395.976187134343 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 133.371721 | 0.092762 | 0.0942908 | 0.09459689 | 21512.20010648539 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 132.310597 | 0.09374550000000001 | 0.0952664 | 0.09808765 | 21266.94869476229 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 136.356699 | 0.223886 | 0.22814345 | 0.22937512 | 17834.97811960297 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 137.235199 | 0.25429999999999997 | 0.26018795 | 0.26566441999999996 | 15671.70167097952 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 140.732034 | 0.3798525 | 0.39349245 | 0.44958836999999974 | 10444.19528075903 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 139.810517 | 0.3755565 | 0.38915655 | 0.39290293 | 10612.39065589615 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 139.522044 | 0.324648 | 0.33001095 | 0.3306158 | 24648.507659184834 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 137.849719 | 0.3327325 | 0.34026 | 0.34328387 | 23969.593611719785 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 141.558025 | 0.4539565 | 0.5028869 | 0.50448716 | 17294.87713632808 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 143.281702 | 0.49646 | 0.52763555 | 0.6187723899999997 | 15927.17331854442 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 140.482979 | 0.345885 | 0.35768859999999997 | 0.39029642 | 45997.0863145674 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 138.742607 | 0.38198 | 0.4180741 | 0.44584056999999994 | 41205.10510649717 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 142.454489 | 0.399175 | 0.43149285 | 0.44386225 | 39692.523848632765 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 144.888695 | 0.5570065 | 0.5782691 | 0.5876289100000001 | 28755.28782785072 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 137.836087 | 0.3653305 | 0.36963569999999996 | 0.3736734 | 87607.09967935801 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 141.924771 | 0.4131615 | 0.4421201499999999 | 0.49130004 | 76692.44283692824 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 141.782951 | 0.570932 | 0.6009431 | 0.6110668100000001 | 55793.518787735164 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 142.006275 | 0.623769 | 0.6592490999999999 | 0.67119048 | 50996.04681832317 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 141.142315 | 0.4111 | 0.4163679 | 0.41729606 | 155666.90279389982 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 142.34344 | 0.5326915000000001 | 0.57320655 | 0.58633115 | 119218.65879306904 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 144.736685 | 0.7834095000000001 | 0.8242428 | 0.84246908 | 81198.98214030925 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 143.994178 | 0.8799885000000001 | 0.91408625 | 0.9883903999999998 | 72343.7162089849 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 138.652901 | 0.4936915 | 0.5724099500000001 | 0.57291176 | 252211.91822690607 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 142.725553 | 0.6717795 | 0.7125078 | 0.71744219 | 189584.11259068677 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 143.402617 | 1.0392175 | 1.152019 | 1.16197917 | 120635.57910412326 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 145.935075 | 1.192836 | 1.244779 | 1.25334514 | 107164.79722143113 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.180812 | 0.209196 | 0.2156729 | 0.22070697 | 4764.171146178586 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 501.219153 | 0.20572649999999998 | 0.2108365 | 0.21378558999999997 | 4850.715354396174 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 536.32649 | 0.201245 | 0.20437205 | 0.2070843 | 4959.110153143272 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 538.123627 | 0.2000015 | 0.20636435 | 0.21286338999999999 | 4978.375926326408 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.899652 | 0.205215 | 0.20880195 | 0.21051542 | 9735.501987794796 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 505.006885 | 0.200319 | 0.2061672 | 0.20841666 | 9978.59093315292 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.331451 | 0.20556049999999998 | 0.20943055 | 0.2117416 | 9718.820904872766 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 542.242409 | 0.20049450000000002 | 0.20342785 | 0.20389023 | 9961.526591996391 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.583986 | 0.2064755 | 0.24175335 | 0.24329011 | 18968.084964122343 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.764308 | 0.2066245 | 0.21176979999999998 | 0.21489951 | 19323.193551927605 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 508.554353 | 0.202794 | 0.20601185 | 0.20634196 | 19729.333276775913 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 570.47122 | 0.2036595 | 0.2087955 | 0.2093288 | 19571.321516299875 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 539.177741 | 0.2070475 | 0.2130889 | 0.21402462 | 38517.92277464627 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 524.702409 | 0.2036175 | 0.20722565 | 0.208905 | 39258.75158544014 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.365057 | 0.207642 | 0.21552645 | 0.22337434999999997 | 38350.4512170276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 531.563128 | 0.2043555 | 0.20739559999999999 | 0.20899582 | 39160.53135358573 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.368888 | 0.21252949999999998 | 0.21733360000000002 | 0.21841182 | 75174.95561388714 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 503.895194 | 0.21684350000000002 | 0.2635674 | 0.26535446 | 71739.6010309698 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 504.809821 | 0.2123105 | 0.26006009999999996 | 0.26084925000000003 | 73055.62455253431 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 546.217202 | 0.2100445 | 0.2122517 | 0.21525302999999998 | 76183.88084307748 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.181777 | 0.226005 | 0.26392214999999986 | 0.29316254999999997 | 138939.31469398746 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 514.233532 | 0.22355 | 0.24638089999999993 | 0.27411995999999994 | 141537.42014967406 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 515.803081 | 0.226913 | 0.23477305 | 0.27339769999999997 | 139910.02561140462 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 539.430429 | 0.2262585 | 0.23268635 | 0.23504382000000001 | 140968.74071038025 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 494.992027 | 0.2481845 | 0.25345215 | 0.25573751 | 257299.3410885499 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 504.400703 | 0.26011799999999996 | 0.32523835 | 0.33035534 | 239556.58674684627 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 507.54768 | 0.243598 | 0.3093353 | 0.31679234 | 254952.53261878638 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 545.535804 | 0.24322349999999998 | 0.2740011999999999 | 0.3188593 | 258491.94506787675 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.66248 | 0.281982 | 0.35878024999999997 | 0.36068346 | 434849.9040816235 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.942773 | 0.4175335 | 0.4519845 | 0.49425190999999985 | 302581.6168997128 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 512.514831 | 0.36031100000000005 | 0.4050091 | 0.40667591 | 347182.57273678324 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 570.505627 | 0.285675 | 0.3067973499999999 | 0.36749813 | 441436.027263365 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.863994 | 0.22016200000000002 | 0.2328153 | 0.23915963999999998 | 4500.687840122606 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 498.818398 | 0.214843 | 0.2193593 | 0.22215866 | 4653.457053244855 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.698377 | 0.21406350000000002 | 0.2185031 | 0.22076676 | 4673.115129387809 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 546.807433 | 0.2181475 | 0.2234592 | 0.22476010999999999 | 4578.459410675356 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 490.813251 | 0.22422599999999998 | 0.22765715 | 0.22817152 | 8907.565480404604 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 547.099355 | 0.226188 | 0.22935285 | 0.23102869 | 8852.763319247997 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.909954 | 0.2213795 | 0.2271647 | 0.22963097 | 9013.487331768893 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 509.648608 | 0.2227535 | 0.22806565 | 0.23109706 | 8963.987181498329 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.183798 | 0.373729 | 0.3898995 | 0.39667925 | 10658.733726445198 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 497.864438 | 0.3757975 | 0.3967775 | 0.41052110999999997 | 10507.971636252481 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 526.876824 | 0.35218400000000005 | 0.36452435 | 0.37208745 | 11276.200957687748 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 504.899864 | 0.36027299999999995 | 0.3745752 | 0.37731918000000003 | 11042.353443538597 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 507.795902 | 0.529053 | 0.6173861 | 0.6239142099999999 | 14870.30196973507 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 502.731183 | 0.551257 | 0.6341371499999999 | 0.64279285 | 14194.65456279009 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.354513 | 0.5815995 | 0.6129559499999999 | 0.61752514 | 13680.213383968361 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 504.713411 | 0.603163 | 0.6274671500000001 | 0.7425485899999996 | 13188.72555201823 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 488.321448 | 0.5109395 | 0.59820515 | 0.60170653 | 30774.09899496793 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 497.655246 | 0.5773415 | 0.6068533 | 0.61190146 | 27576.82411987618 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.422097 | 0.5900099999999999 | 0.6582236499999999 | 0.66192263 | 26649.70235447122 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 512.312932 | 0.635273 | 0.68084915 | 0.72201471 | 24974.48544131102 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.787829 | 0.538384 | 0.56481495 | 0.5678615699999999 | 59139.914910968924 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.118217 | 0.5774595 | 0.60705585 | 0.61653506 | 54953.48479452531 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 505.17943 | 0.5893705 | 0.60430465 | 0.61060694 | 54211.13998946067 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 506.618297 | 0.6523695 | 0.67883715 | 0.69181604 | 48795.952253770665 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 494.254705 | 0.5793065 | 0.6203055000000001 | 0.62132792 | 109389.98503921033 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.307404 | 0.6596165 | 0.75271145 | 0.77780911 | 95180.28737724679 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 507.679147 | 0.6950035 | 0.7454628 | 0.7934673699999999 | 90822.26276174834 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 521.01539 | 0.7209855000000001 | 0.77718675 | 0.8343331099999999 | 87358.60634194544 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.468716 | 0.5879715000000001 | 0.6255194 | 0.6517620699999999 | 215321.86665890607 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 503.502818 | 0.7369795 | 0.7953213 | 0.80135581 | 171112.9291859837 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 506.638288 | 0.767477 | 0.8130947 | 0.83054323 | 166348.101388544 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.906703 | 0.8029995000000001 | 0.8527249 | 0.86301244 | 159202.39599605973 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.749438 | 0.027234500000000002 | 0.028580349999999997 | 0.031769479999999996 | 36457.12569104482 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.063167 | 0.0287695 | 0.030819749999999993 | 0.03493305 | 34454.078570458936 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.137829 | 0.0282445 | 0.02864605 | 0.03182854999999998 | 35359.67865124042 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.621397 | 0.028759 | 0.03276674999999998 | 0.05662357999999992 | 33440.364794251465 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.879518 | 0.0287475 | 0.03071579999999999 | 0.03331709 | 69075.22775829473 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.11501 | 0.0295515 | 0.030145599999999998 | 0.03370919999999999 | 67347.91325858164 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.286898 | 0.0294595 | 0.0301605 | 0.03733239999999999 | 67250.85240455423 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.976447 | 0.0296855 | 0.030424 | 0.03788218 | 66716.17006485478 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.796518 | 0.0303605 | 0.03299984999999999 | 0.03802099 | 130117.86075827484 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.000956 | 0.03124 | 0.032609349999999995 | 0.036531669999999995 | 128065.56957162067 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.310844 | 0.031004 | 0.033736749999999996 | 0.0397917 | 127307.8526666221 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.375785 | 0.0309325 | 0.031455699999999996 | 0.03731552 | 128523.71873098252 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.941378 | 0.0346005 | 0.035565200000000005 | 0.03852212999999999 | 230060.3793465595 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.845012 | 0.0360735 | 0.0373217 | 0.04233861 | 221918.44053473463 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.699738 | 0.0358005 | 0.039665449999999984 | 0.0462562 | 220721.43905963842 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.131556 | 0.035792500000000005 | 0.03710535 | 0.04018002 | 222437.6159803632 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.870578 | 0.0422805 | 0.04514964999999999 | 0.048323849999999995 | 379433.2026818338 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.964553 | 0.0437815 | 0.04778564999999999 | 0.049631549999999997 | 364090.9808747558 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.239953 | 0.043263499999999996 | 0.044140549999999994 | 0.12192313999999971 | 347782.84091630345 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.998198 | 0.0438785 | 0.04883439999999999 | 0.05139532 | 362462.3152461618 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.926288 | 0.056391 | 0.0575432 | 0.058653199999999996 | 570445.9782310685 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.948185 | 0.0574745 | 0.0602944 | 0.06440425 | 558318.2616481765 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.198487 | 0.057068 | 0.06254915 | 0.06898894999999998 | 556044.0247856624 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.04943 | 0.0573535 | 0.0634237 | 0.06606461999999999 | 554914.7061414492 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.90568 | 0.08215 | 0.08340755 | 0.08510168 | 780718.7833864017 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.823241 | 0.13287500000000002 | 0.16493615 | 0.17094605 | 471008.4718233165 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.164414 | 0.142313 | 0.1701791 | 0.17484993999999998 | 442688.0182476001 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.826125 | 0.1454515 | 0.16600025 | 0.17463890999999998 | 443923.94362054934 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.034775 | 0.1352125 | 0.13768835 | 0.13993095 | 947470.4611279517 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.166999 | 0.3372625 | 0.34913109999999997 | 0.35662724999999995 | 378571.9296722194 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.277027 | 0.3764685 | 0.41208344999999996 | 0.43018518 | 337659.08029369375 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.75433 | 0.40012349999999997 | 0.42087240000000004 | 0.43341107999999995 | 318716.16348605364 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 244.092526 | 0.359027 | 0.41603435 | 0.42918668 | 2755.1700766488316 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 249.189095 | 0.335781 | 0.37955324999999995 | 0.42134423999999987 | 2988.2891339471917 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 235.805315 | 0.33098700000000003 | 0.38618319999999995 | 0.40653070999999996 | 3008.070834533303 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 242.615307 | 0.3349585 | 0.38981635 | 0.41660084 | 2952.9725537147187 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 246.936033 | 0.4370465 | 0.7281017499999998 | 0.82629168 | 4571.320113593647 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 246.680566 | 0.5179020000000001 | 0.65026095 | 7.125437449999975 | 2567.5579045286972 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 239.04506 | 0.5438004999999999 | 2.445534899999999 | 2.73563759 | 2201.1762381440244 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 244.994737 | 0.4687755 | 0.6336324999999998 | 0.8161222799999994 | 4758.36114238544 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 238.931172 | 0.5719145 | 0.68676175 | 0.69333729 | 6864.122296694135 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 247.847443 | 0.9353955 | 2.3648313499999998 | 5.58782943999999 | 3163.955610209022 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 249.440001 | 0.574642 | 0.66936945 | 0.7224345099999998 | 6854.831094734006 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 244.312341 | 0.5833125 | 0.7446516999999999 | 24.577172229999935 | 2599.790319111393 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 240.675472 | 0.6684844999999999 | 0.73934225 | 0.9380381399999993 | 11791.743656926294 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 240.681754 | 0.691335 | 0.8660519499999999 | 1.5007786699999976 | 11086.774716710039 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 240.057907 | 0.6780345 | 0.78680325 | 0.8809018099999999 | 11546.408465110811 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 243.269801 | 0.6765645 | 0.7996095999999999 | 0.8283434399999999 | 11708.896460511834 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 249.831076 | 1.1709275 | 1.3451346999999996 | 1.41123396 | 13671.757875492223 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 239.959216 | 1.1627045 | 1.295787 | 1.3923799799999999 | 13813.51672986955 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 251.589542 | 1.117438 | 1.2816469999999998 | 1.32445118 | 14213.402332294956 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 241.184831 | 1.0996294999999998 | 1.29056855 | 1.3249882999999998 | 14310.609123564276 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 238.697299 | 1.1065325000000001 | 1.3563688999999999 | 1.40899794 | 28042.55843828188 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 250.047442 | 1.1122005000000001 | 1.2593773499999998 | 1.3193987699999998 | 28656.74173849419 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 245.615427 | 1.1836989999999998 | 1.3516527999999999 | 1.4120959999999998 | 26938.41783398688 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 236.113302 | 1.101812 | 1.229138 | 1.30330776 | 28840.062422873107 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 238.927429 | 1.3863159999999999 | 1.5991336 | 1.64091388 | 46018.67227760833 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 244.315148 | 1.4094125 | 1.6362732 | 1.7329415499999996 | 44877.84663862614 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 248.463873 | 1.3756249999999999 | 1.5896702499999997 | 1.8165651099999993 | 45691.79099284007 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 236.337811 | 1.461806 | 1.68553 | 1.73747234 | 43512.13944498498 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 239.990828 | 1.471202 | 1.6594262499999999 | 1.70805076 | 86684.02884557338 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 241.20187 | 1.413362 | 1.5768511 | 1.60332357 | 90988.99708444241 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 239.736554 | 1.4765570000000001 | 1.70133675 | 1.7236910300000001 | 86132.96652891762 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 239.388137 | 1.454226 | 1.68063435 | 1.7839221399999998 | 87781.44909972786 | - |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_depth2_stages1_share1` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
