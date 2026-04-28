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

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`139092.656` samples/s, p50=`0.909` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.336` ms, throughput=`2922.396` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`204482.904` samples/s, p50=`0.616` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.112` ms, throughput=`8847.611` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`164753.348` samples/s, p50=`0.766` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.299` ms, throughput=`3254.402` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`258856.984` samples/s, p50=`0.488` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.045` ms, throughput=`21804.497` samples/s

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`62921.738` samples/s, p50=`2.024` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.437` ms, throughput=`2282.449` samples/s

## Run References

### separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260427_072855_default_6port_separator1_masked_v2/separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `255,120`
- MACs / sample: `251,904`
- FLOPs / sample estimate: `507,384`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.34072749999999996 | 0.35527585 | 0.36634154999999996 | 2929.975003797247 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.34071850000000004 | 0.4125908 | 0.42666826 | 2876.722653442948 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.3398245 | 0.3468347 | 0.35999401 | 2936.9671056747197 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.3358905 | 0.39136779999999993 | 0.40258 | 2922.3956092525614 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3633555 | 0.3762063 | 0.37903002 | 5480.14330355533 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.377595 | 0.38551545000000004 | 0.38938745 | 5304.868399742418 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.3818435 | 0.39988914999999997 | 0.40151547 | 5234.689723450819 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.3642085 | 0.37569325 | 0.38113228 | 5492.02653857064 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.389621 | 0.40292415 | 0.41090799 | 10244.881349931133 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.385494 | 0.4483964 | 0.46051506000000003 | 10139.037637020223 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.3993105 | 0.4356878999999999 | 0.45742257999999997 | 9958.821765821593 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.3865465 | 0.40830405 | 0.45554914 | 10244.806830581527 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.4044105 | 0.41256645 | 0.41797536 | 19723.633434497493 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.408694 | 0.42205595 | 0.42775043 | 19502.87750330403 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.40709150000000005 | 0.46708715 | 0.47158566 | 19124.92015345836 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.4118185 | 0.4187625 | 0.42309410999999997 | 19415.22224289404 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.44490850000000004 | 0.46056375 | 0.46638464 | 35795.472722843035 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.478721 | 0.49127075 | 0.54115114 | 33294.83895036689 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.44392200000000004 | 0.5038624999999999 | 0.5219402 | 35601.27550469815 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.453569 | 0.5099251999999999 | 0.53408267 | 34820.297883295854 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.5427985 | 0.56498695 | 0.66255079 | 58305.432058374536 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.5838414999999999 | 0.5992985500000001 | 0.60380785 | 54808.93229390934 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.5263955 | 0.6201972 | 0.62546239 | 58817.48548264984 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.512809 | 0.55530395 | 0.59283024 | 61777.02540224951 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.6652645 | 0.7243195 | 0.7275955399999999 | 94851.79717923205 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.7761709999999999 | 0.8204273 | 0.9600874299999997 | 81275.16672838498 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.7180205 | 0.7776634 | 0.79541794 | 88182.889328352 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.6435185 | 0.6775659999999999 | 0.7872264499999999 | 98382.21817652679 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.9085755 | 0.97016055 | 0.9923197899999999 | 139092.65600773564 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.1043635 | 1.1373947 | 1.14663646 | 115599.81165900687 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.0304959999999999 | 1.0546896 | 1.06588461 | 124004.79131262733 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.9499865000000001 | 1.0229507999999998 | 1.05035237 | 133652.922531028 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.4994365 | 0.507243 | 0.5113518899999999 | 2000.8963215161864 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5157685 | 0.52495475 | 0.52995886 | 1936.0172624592003 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.506772 | 0.517773 | 0.5249739699999999 | 1969.5941926909145 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5255985000000001 | 0.537293 | 0.53837564 | 1902.7877172008396 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.7271 | 0.7574690000000001 | 0.76005672 | 2728.4350902290757 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.741635 | 0.7696086 | 0.8023530799999999 | 2677.9101043252185 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.76028 | 0.7736753000000001 | 0.79358106 | 2625.7859108520474 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.762759 | 0.79905985 | 0.8655498499999998 | 2611.7072231281513 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.870324 | 0.8936026 | 0.9033623599999999 | 4596.149975223305 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.884004 | 0.9314559499999999 | 0.9886359499999999 | 4494.715742490183 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.9167860000000001 | 0.93976695 | 0.94650376 | 4355.732688684896 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.9219280000000001 | 0.9569936 | 1.0488509799999997 | 4308.154168330871 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.916343 | 0.9525648 | 0.95835173 | 8682.622954406592 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.892382 | 0.9441409999999999 | 1.0072475799999998 | 8902.098812228563 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.945217 | 1.01128305 | 1.1763307799999996 | 8361.76874738342 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.9658865000000001 | 1.00293575 | 1.07953251 | 8227.126613504071 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.8912995 | 0.9320882500000001 | 1.0376374099999996 | 17826.615176568503 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.0186215 | 1.0803011 | 1.10410784 | 15663.126558591222 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.027061 | 1.05479755 | 1.05688409 | 15593.883805878368 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9954335000000001 | 1.04789565 | 1.07237691 | 15942.442680342308 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8927095 | 0.9372271499999999 | 0.9633235499999999 | 35526.89166986982 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.0925865 | 1.1701906 | 1.18141123 | 28948.736582147514 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.0783025 | 1.1475552500000001 | 1.15326186 | 29401.596833800842 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.12996 | 1.2081441 | 1.2264625999999998 | 28071.468696084143 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.007515 | 1.0788146 | 1.08177232 | 62651.35441460042 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.2891629999999998 | 1.3569537 | 1.37658909 | 49341.84379399182 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.3548585 | 1.3934328 | 1.41477602 | 47238.017475320754 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.3578535 | 1.3910027 | 1.40939686 | 47267.79375046797 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.1471065 | 1.25300645 | 1.2686859799999999 | 109588.89848020062 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.5808535 | 1.63341365 | 1.67138645 | 81033.04880983912 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.5962075 | 1.669619 | 1.7706007499999998 | 79971.32128454585 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.7459 | 1.80712105 | 1.8799885399999998 | 73542.51574996284 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 183.548349 | 0.11200550000000001 | 0.11962389999999999 | 0.12315072999999999 | 8847.6110476934 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 181.745826 | 0.13141350000000002 | 0.1381785 | 0.14022384 | 7559.647508755963 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 180.542225 | 0.128195 | 0.13259155 | 0.13373654 | 7773.948503499367 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 181.733044 | 0.13688650000000002 | 0.1419228 | 0.14265681 | 7281.855276912936 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 180.35811 | 0.1251265 | 0.1312593 | 0.13295193 | 15931.055489777997 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 184.162487 | 0.115929 | 0.1235892 | 0.12947387 | 17097.009801373777 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 183.409273 | 0.128827 | 0.1350706 | 0.1397976 | 15410.868391646261 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 183.403398 | 0.125876 | 0.13194465 | 0.13361116 | 15847.368290937397 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 180.009156 | 0.137853 | 0.14611695 | 0.14861577 | 28806.883692927262 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 184.270687 | 0.13973249999999998 | 0.15012885 | 0.15189741 | 28293.505508745522 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 182.714231 | 0.146297 | 0.1549985 | 0.15824491 | 27092.73422989513 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 180.055266 | 0.1423485 | 0.150619 | 0.15197049 | 27972.454405248976 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 183.606775 | 0.1506085 | 0.1590788 | 0.16249688999999998 | 52732.90939589707 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 184.42292 | 0.157026 | 0.16828995 | 0.17182656 | 50540.12867264059 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 180.724962 | 0.17767349999999998 | 0.19029634999999998 | 0.19783382 | 44523.13165913002 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 184.132388 | 0.160304 | 0.17074815 | 0.19759351999999997 | 49231.005539595826 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 181.804213 | 0.1862475 | 0.19474625 | 0.1996977 | 85401.97374636574 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 186.842916 | 0.2305525 | 0.2613767 | 0.26667113 | 67815.31110358739 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 184.743886 | 0.21667550000000002 | 0.2263192 | 0.22708746999999999 | 73414.9977296412 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 195.09843 | 0.29977 | 0.3205704 | 0.32511266 | 52795.94104805223 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 182.069209 | 0.252158 | 0.26035389999999997 | 0.27061682 | 126116.69435847708 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 184.268436 | 0.345512 | 0.3774995 | 0.38094417 | 91549.24379466352 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 183.404965 | 0.2914695 | 0.32314745 | 0.33127041999999995 | 108067.7214481534 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 185.715586 | 0.3073205 | 0.33866905 | 0.33987217 | 102591.22347471857 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 182.830456 | 0.3770425 | 0.38715725 | 0.3905598 | 169417.5219119064 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 185.087784 | 0.5762595 | 0.62572805 | 0.6440899299999999 | 109691.55386347191 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 187.725422 | 0.5143089999999999 | 0.5555679499999999 | 0.6057898599999999 | 122844.77480134272 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 183.862669 | 0.462908 | 0.48782605 | 0.48961834 | 136839.73172399547 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 184.816534 | 0.616248 | 0.6897944 | 0.70913175 | 204482.90427068935 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 190.278339 | 0.9288935 | 0.9516278499999999 | 0.9642372 | 137677.99182881118 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 186.473023 | 0.8044899999999999 | 0.8230294 | 0.83065171 | 159141.6694060584 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 187.083617 | 0.6624695 | 0.6837255999999999 | 0.68714727 | 192977.1979349028 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 184.049986 | 0.235583 | 0.25079219999999997 | 0.26059097000000003 | 4213.83777129109 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 187.124839 | 0.2690575 | 0.2782126 | 0.28227406 | 3703.7684510632926 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 183.348979 | 0.258015 | 0.26653725 | 0.26939211 | 3855.1452310311433 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 186.956415 | 0.2703845 | 0.2796695 | 0.28935725 | 3683.073580885968 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 184.604689 | 0.3510035 | 0.35839385 | 0.36768847 | 5686.085007084577 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 191.754442 | 0.4198735 | 0.4281025 | 0.43048869 | 4758.552474985004 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 191.816181 | 0.410588 | 0.41812265 | 0.42298567 | 4866.652267864823 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 193.306943 | 0.44517450000000003 | 0.4566955 | 0.47533316999999997 | 4478.151189907466 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 187.74987 | 0.4544135 | 0.46311465 | 0.46388169 | 8785.241391858093 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 188.888689 | 0.5526975000000001 | 0.6322801 | 0.64246363 | 7131.915765228172 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 193.668222 | 0.5313034999999999 | 0.54313575 | 0.54592704 | 7512.776885739867 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 195.133276 | 0.5823525 | 0.6002483000000001 | 0.6086696899999999 | 6844.2636737522935 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 190.877375 | 0.4899795 | 0.6016804 | 0.60796515 | 15982.662646867186 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 192.706656 | 0.571027 | 0.66428225 | 0.6853109 | 13788.846973858277 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 189.125427 | 0.5922495 | 0.71079605 | 0.71907965 | 13262.396751826838 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 192.371165 | 0.6313615 | 0.66349275 | 0.6942675999999999 | 12584.886237505056 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 189.591327 | 0.5267995000000001 | 0.64331185 | 0.64776831 | 29188.029872197294 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 190.443077 | 0.6311294999999999 | 0.7342281 | 0.74529941 | 24735.540739172786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 190.235018 | 0.645147 | 0.72691575 | 0.73795747 | 24426.69776999516 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 194.580111 | 0.717796 | 0.75339015 | 0.7768262899999999 | 22219.23743799305 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 188.967722 | 0.5548010000000001 | 0.6787033499999999 | 0.6823449 | 56113.095663744796 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 190.868227 | 0.7399135 | 0.7848234499999999 | 0.82112239 | 43079.438916155785 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 194.116548 | 0.75322 | 0.85275945 | 0.85670393 | 41822.986613978064 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 192.868096 | 0.770281 | 0.80263555 | 0.8313852099999999 | 41503.23471023427 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 192.227974 | 0.6832185 | 0.80253545 | 0.80912803 | 91710.46056104849 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 197.009269 | 0.964633 | 1.02840765 | 1.0833301499999999 | 65967.73847749758 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 192.538814 | 0.985369 | 1.02024895 | 1.02990901 | 64661.98482912635 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 196.203269 | 0.9806595 | 1.0275706 | 1.03733679 | 65084.78371045017 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 191.480744 | 0.8431495 | 0.9584711499999999 | 0.97023215 | 148658.8601563315 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 193.829486 | 1.3321065 | 1.39556885 | 1.41794835 | 95764.12959705791 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 195.738821 | 1.3226209999999998 | 1.36857225 | 1.37188151 | 96685.19930581236 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 196.506451 | 1.3520045 | 1.4008168 | 1.40587201 | 94760.10804962095 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 494.362035 | 0.2987935 | 0.36170754999999993 | 0.37500488 | 3254.4023602488032 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 513.916808 | 0.3050835 | 0.38115505 | 0.40179173 | 3194.6993548304654 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 510.921694 | 0.30403800000000003 | 0.326812 | 0.36803378999999997 | 3249.966281599828 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 535.730378 | 0.313001 | 0.33524595 | 0.33986117 | 3149.6517335588646 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.751561 | 0.2949545 | 0.33347085 | 0.3414258 | 6676.46460101748 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 517.100669 | 0.30377200000000004 | 0.3514888 | 0.36183048 | 6435.6512425344035 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 515.646086 | 0.30834249999999996 | 0.31604435 | 0.31707233 | 6485.313100539993 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 538.66493 | 0.302837 | 0.35386129999999993 | 0.36915994999999996 | 6432.031723809587 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.666409 | 0.321326 | 0.3324875 | 0.33546 | 12395.971458027798 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.799402 | 0.3206645 | 0.35164719999999994 | 0.3734113 | 12320.885891409009 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 508.587175 | 0.311717 | 0.3512112 | 0.35847617 | 12630.89235865116 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 548.002052 | 0.3142965 | 0.35550095 | 0.36239156 | 12613.543969426786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 492.528164 | 0.3441945 | 0.3749858999999999 | 0.39539006 | 22975.461230948244 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 514.752947 | 0.3471665 | 0.3835696 | 0.40039684 | 22881.31598685514 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 510.191152 | 0.34459850000000003 | 0.41955579999999987 | 0.45297401 | 22729.364861992977 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 532.60063 | 0.3441325 | 0.35956489999999997 | 0.4095081299999998 | 23023.111865962204 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 492.126533 | 0.378958 | 0.42721069999999994 | 0.45121103999999995 | 41789.75272428704 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 506.16143 | 0.406651 | 0.4852198 | 0.49899263999999993 | 38321.21281273583 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 514.490736 | 0.376732 | 0.4786185499999999 | 0.50144073 | 41071.1394232729 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 536.546358 | 0.380722 | 0.40299815 | 0.42424251999999996 | 41638.42627151523 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.470066 | 0.4388895 | 0.5403628500000001 | 0.54973065 | 71117.07862419744 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.116968 | 0.479919 | 0.52835225 | 0.5546219 | 66197.73843701347 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.446733 | 0.4432025 | 0.5249401499999999 | 0.53301404 | 70284.52449422269 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.878126 | 0.43913250000000004 | 0.46884174999999995 | 0.54026425 | 71916.2179656511 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 498.718309 | 0.544126 | 0.6080740499999998 | 0.6596718899999999 | 116067.25546134521 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.041138 | 0.719104 | 0.7652947999999999 | 0.78516769 | 87986.15581831276 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 509.564043 | 0.632425 | 0.67813055 | 0.7000385 | 100694.92399090863 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 558.044739 | 0.5516034999999999 | 0.5810259 | 0.65159993 | 114795.95771863984 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.598129 | 0.7661640000000001 | 0.85865405 | 0.8812144 | 164753.34827714847 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 510.642685 | 0.942984 | 1.006875 | 1.0808703999999998 | 133682.68238814003 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 507.618849 | 0.8981725 | 0.9625117 | 0.9979594599999999 | 140725.9086534494 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 540.266415 | 0.8695415 | 0.9148047499999999 | 0.94033998 | 146697.52021136545 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 494.879477 | 0.4641505 | 0.5086239 | 0.51033887 | 2110.1827713705593 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 507.116466 | 0.4841225 | 0.5242872999999999 | 0.5459894599999999 | 2042.9422372383017 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 517.193186 | 0.4729065 | 0.48188335 | 0.48425206 | 2111.6115959066324 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 511.166708 | 0.49514650000000004 | 0.5253484 | 0.53118205 | 1999.6124751023253 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 495.250153 | 0.587806 | 0.59735235 | 0.6188164599999999 | 3395.4145266359237 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 504.071057 | 0.6364385 | 0.6453444 | 0.65466576 | 3142.8584089801016 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 502.202074 | 0.642242 | 0.65182945 | 0.65496066 | 3115.253442845707 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 497.798653 | 0.6678649999999999 | 0.69595225 | 0.7527996699999998 | 2976.217846765317 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.616035 | 0.7403029999999999 | 0.852865 | 0.86920643 | 5325.160277339673 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 512.377579 | 0.8411075 | 0.9419476999999999 | 0.98001772 | 4666.94498770925 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 514.025166 | 0.8621655 | 0.89218395 | 0.91755086 | 4627.307406838419 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.103578 | 0.9047305000000001 | 0.92749185 | 0.9944525199999998 | 4404.4323300043825 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 491.419375 | 0.785391 | 0.8165481 | 0.82052309 | 10143.59038315612 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 505.951776 | 0.8621084999999999 | 0.9932350999999999 | 1.03059674 | 8982.491238253484 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 511.510503 | 0.8773425 | 0.90904925 | 0.94869033 | 9081.751176864402 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 513.288196 | 0.894412 | 0.9235838 | 0.95229531 | 8928.774119716065 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.802159 | 0.7575689999999999 | 0.78903035 | 0.8422391599999999 | 20989.336026754896 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 501.443241 | 0.9229315 | 0.97615395 | 0.98982578 | 17197.760008956593 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.499838 | 0.94101 | 1.02688275 | 1.0504299499999998 | 16727.3895300718 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 504.707367 | 0.9848375 | 1.070579 | 1.1304062199999998 | 16044.995944827837 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 493.850521 | 0.783964 | 0.8390473 | 0.88024642 | 40437.61690104352 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 511.418215 | 0.938861 | 1.0658978 | 1.07488421 | 33313.07967491348 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.989709 | 0.981323 | 1.0791302999999999 | 1.1076959 | 32018.868078581345 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 508.927281 | 1.0179935 | 1.10763165 | 1.16113754 | 30748.45545703793 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.566684 | 0.83832 | 0.8810578 | 0.9281028199999999 | 75412.39266931213 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.208212 | 1.1353435 | 1.22846465 | 1.2990302999999996 | 55491.36572557414 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 512.375054 | 1.096325 | 1.154674 | 1.18090641 | 58454.06960520633 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 505.578394 | 1.1580035 | 1.2199117 | 1.23229095 | 55513.2382766276 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.132241 | 0.9381710000000001 | 0.98521925 | 1.0350212099999998 | 134845.4821422337 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 507.178497 | 1.298083 | 1.3683809 | 1.40007796 | 98106.17976651403 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 503.574744 | 1.2538375 | 1.2995385499999998 | 1.30304796 | 102625.17605629568 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 511.547143 | 1.3510965000000001 | 1.39823035 | 1.43069777 | 94424.51980521106 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.789016 | 0.056625499999999995 | 0.06046819999999999 | 0.06859509999999998 | 17482.988178353055 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.318051 | 0.0448605 | 0.052118199999999996 | 0.06376837999999999 | 21804.49652327303 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.350806 | 0.046841 | 0.0541414 | 0.06171522999999998 | 20774.73991064369 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.188426 | 0.050498 | 0.053178899999999994 | 0.05885484 | 19603.32284163495 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.658141 | 0.0530575 | 0.0559159 | 0.06277039999999998 | 37393.354153952925 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.460589 | 0.054894 | 0.05979644999999999 | 0.06300863999999999 | 36204.062023350896 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.199637 | 0.053316 | 0.05781745 | 0.060984079999999996 | 37403.99796372635 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 10.936293 | 0.052185999999999996 | 0.05393895 | 0.05948749 | 38143.451417830234 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.734865 | 0.058757000000000004 | 0.062294999999999996 | 0.06985697999999999 | 67739.18112136118 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.086565 | 0.0754935 | 0.08381005 | 0.08585145 | 52588.82976443623 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.65031 | 0.08218249999999999 | 0.0903757 | 0.09694641999999999 | 48028.74424285446 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.944486 | 0.0800285 | 0.08944374999999999 | 0.09581049 | 49428.44651582588 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.84713 | 0.0752095 | 0.0781498 | 0.08005208 | 105480.53106282974 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.034574 | 0.123769 | 0.1353187 | 0.13924261000000002 | 64186.04710870742 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.607717 | 0.12735249999999998 | 0.14128559999999998 | 0.14922445999999998 | 61997.788848860706 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.963447 | 0.125007 | 0.15017919999999998 | 0.15367728 | 62080.98246879615 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.675846 | 0.10094249999999999 | 0.10439355 | 0.1068697 | 158073.30768198732 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 9.887111 | 0.1631415 | 0.17760789999999999 | 0.18231747 | 96542.79060073872 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.379446 | 0.16907250000000001 | 0.1805545 | 0.18196887 | 93860.72265482238 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 10.870116 | 0.170443 | 0.18122615 | 0.18849454 | 93398.99126754455 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 10.069793 | 0.15643600000000002 | 0.1609699 | 0.16499164 | 204097.74700323917 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.161716 | 0.3128855 | 0.37914115 | 0.38721482 | 99663.76559979306 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.063982 | 0.3038115 | 0.325305 | 0.33516229 | 104666.68175611329 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.75021 | 0.3258495 | 0.34183605 | 0.34696844 | 98473.65829640572 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.823738 | 0.26712899999999995 | 0.28000415 | 0.2905481 | 238358.76286141056 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.282395 | 0.627312 | 0.6526941 | 0.66030302 | 101678.80928776186 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.494729 | 0.621993 | 0.6670792 | 0.67159711 | 102461.67044990374 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 10.91108 | 0.6266449999999999 | 0.66879765 | 0.68376922 | 101509.24259100691 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.751687 | 0.4875955 | 0.5426232 | 0.54786879 | 258856.98382188345 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.031296 | 0.868288 | 0.8943858 | 0.91030098 | 146960.72926873786 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.541575 | 0.7830125 | 0.86615645 | 0.8701156800000001 | 161229.96698766042 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 11.366399 | 0.747435 | 0.83338775 | 0.84300471 | 168598.4759066587 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 352.185813 | 0.454415 | 0.49087 | 0.49730837 | 2218.5219248535154 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 364.168565 | 0.45520000000000005 | 0.4975835 | 0.5162204199999999 | 2186.564505708289 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 283.554821 | 0.43693550000000003 | 0.48893219999999993 | 0.50610593 | 2282.44884484351 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 295.496297 | 0.4384805 | 0.468604 | 0.5383019899999999 | 2288.5901149751057 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 274.540166 | 0.600742 | 0.69445855 | 0.7302327099999999 | 3263.7649859441062 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 274.304764 | 0.6140375 | 0.6882905 | 0.7052742599999999 | 3222.6125307292223 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 298.45249 | 0.6020645 | 0.68659775 | 0.7258577399999999 | 3256.3104120883618 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 272.7508 | 0.60233 | 0.6736317499999999 | 0.73087182 | 3290.5493907843947 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 267.959971 | 0.707337 | 0.8343801 | 0.84385327 | 5552.685279989574 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 279.262525 | 0.6799895 | 0.7965776 | 0.82085205 | 5792.245700575303 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 265.440241 | 0.48308 | 0.7693712999999999 | 4.866502479999984 | 6149.155956092444 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 274.062776 | 0.7105235000000001 | 3.580013899999999 | 16.678965439999974 | 2389.052862859383 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 303.669912 | 0.796643 | 0.91260885 | 0.94740505 | 9945.696002541325 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 270.956387 | 0.753197 | 0.8993356999999998 | 0.94460493 | 10411.309993449466 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 269.581617 | 0.8012935 | 0.9519281999999998 | 1.0109321599999999 | 9860.775707781828 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 269.347014 | 0.782463 | 0.9352231999999999 | 0.9778936699999999 | 10055.193206892463 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 268.156467 | 1.5684305 | 1.67856335 | 1.7439926199999998 | 10222.059549502335 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 284.900365 | 1.6127509999999998 | 1.8255648 | 1.8341372699999998 | 9875.240287741295 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 352.711681 | 1.46065 | 1.63669935 | 1.68850039 | 10995.383780538956 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 276.299846 | 1.6219185 | 1.8580272999999998 | 1.95263551 | 9904.069553705418 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 282.510639 | 1.6627025 | 1.8767996999999998 | 1.92908062 | 19134.496724287772 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 262.591843 | 1.6806155 | 1.9457725999999995 | 2.0206682299999996 | 18892.976436821486 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 282.776564 | 1.691479 | 1.9029755999999998 | 1.97395324 | 18887.4313694194 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 267.348463 | 1.7234275000000001 | 1.8687334999999998 | 1.9686539699999996 | 18731.201556356806 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 289.532982 | 1.9245255000000001 | 2.2386961999999997 | 2.32905944 | 32952.43137352386 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 313.733713 | 2.0699565 | 2.2743092 | 2.3464233699999997 | 31237.483347858928 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 280.748368 | 1.9654880000000001 | 2.14102775 | 2.22432227 | 33229.139213686416 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 330.07577 | 1.9366240000000001 | 2.2224256000000002 | 2.25795954 | 32447.754020715274 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 269.5481 | 2.082064 | 2.3016374 | 2.4109409499999996 | 61884.34145342919 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 271.514666 | 2.0350535 | 2.3187096499999997 | 2.6074400199999994 | 62310.1827378518 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 272.75785 | 2.023714 | 2.25554855 | 2.5931517699999995 | 62921.73786379284 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 267.579302 | 2.048376 | 2.34604855 | 2.42955609 | 62016.43111776371 | - |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `separator1_grid_search_6ports_masked_hd128_depth3_stages1_share0` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
