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

### full_mlp_capacity_search_hd64_depth2::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2837902.134` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.026` ms, throughput=`36924.186` samples/s

### full_mlp_capacity_search_hd64_depth2::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4072954.247` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.016` ms, throughput=`62738.327` samples/s

### full_mlp_capacity_search_hd64_depth2::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3008177.543` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.026` ms, throughput=`37971.473` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth2/MODEL_COMPLEXITY.md`
- Trainable parameters: `3,600`
- MACs / sample: `3,456`
- FLOPs / sample estimate: `7,128`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.028375499999999998 | 0.03321225 | 0.034674479999999994 | 33964.991603854076 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0293975 | 0.033273699999999996 | 0.03589740999999999 | 33457.885889872705 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0281705 | 0.0335701 | 0.034775179999999996 | 34103.17848444111 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0289265 | 0.03909159999999998 | 0.04164494 | 32716.603545432896 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.032219 | 0.0343233 | 0.04019148999999998 | 31571.81546147576 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0293045 | 0.0346495 | 0.03644852 | 65593.68516474182 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.030502 | 0.035842849999999996 | 0.03851983999999999 | 63191.512621556765 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.02876 | 0.03172115 | 0.033469399999999996 | 68235.6312819428 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.029987 | 0.03370375 | 0.03503352 | 65192.23560473947 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0313995 | 0.0341217 | 0.036383799999999994 | 64451.38657490508 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.030928999999999998 | 0.035386799999999996 | 0.03900243 | 126866.20182890315 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0292175 | 0.0339441 | 0.03719407999999999 | 131403.0229265424 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.029935 | 0.03582429999999999 | 0.038599 | 129198.63256167296 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.030435 | 0.03738364999999999 | 0.042576989999999995 | 127000.41529135799 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0310395 | 0.0334586 | 0.03902605999999998 | 128300.44885912031 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.029323000000000002 | 0.0352155 | 0.03730376999999999 | 257672.0234533076 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0298685 | 0.0350797 | 0.04146759999999999 | 256168.04623833232 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.032141 | 0.03675774999999999 | 0.03968744 | 249973.44032196578 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.030727 | 0.035775799999999996 | 0.0385589 | 250065.48589911987 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0292025 | 0.030990049999999998 | 0.0315238 | 272710.6621687588 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.029969 | 0.03386785 | 0.03721376999999999 | 519169.35499048297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.0308755 | 0.03620595 | 0.0410682 | 501420.90147956775 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.034179 | 0.038530550000000004 | 0.0404068 | 478586.2561991876 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.031310000000000004 | 0.0384813 | 0.08105906999999984 | 464828.19368926005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.029693999999999998 | 0.03113795 | 0.0363861 | 531744.8342651053 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.032021999999999995 | 0.037375649999999996 | 0.03960028 | 966545.4457586777 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.032600000000000004 | 0.0381946 | 0.04185764999999999 | 936320.2730309915 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.032088 | 0.037012199999999995 | 0.04179903999999999 | 947308.9026314466 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0321255 | 0.0372735 | 0.038575719999999994 | 965490.940074391 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.031647 | 0.0343267 | 0.0361126 | 997180.4722148126 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.035061999999999996 | 0.040758 | 0.0410029 | 1763214.604706571 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.052275 | 0.062343749999999996 | 0.13314648999999984 | 1125103.8084060724 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.046283500000000005 | 0.0547426 | 0.05689204 | 1336638.7295248876 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.034959000000000004 | 0.04111574999999999 | 0.10112548999999976 | 1661641.067874403 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.034988 | 0.03870534999999999 | 0.043106919999999986 | 1796095.960916952 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0444735 | 0.0493843 | 0.0498904 | 2837902.134058063 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0687595 | 0.07776285 | 0.08179027 | 1856545.8616142317 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.07225200000000001 | 0.07979645 | 0.09147516999999997 | 1747497.5289292755 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0607955 | 0.0695278 | 0.10884839999999985 | 2020189.2664819038 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2067405 | 0.2322617 | 0.23631748000000002 | 612530.9280264674 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.027193000000000002 | 0.0318897 | 0.034024769999999996 | 35755.839107305415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0269725 | 0.0302161 | 0.032701459999999995 | 36435.76593080994 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0262415 | 0.029782549999999998 | 0.04301097999999997 | 36127.27204413885 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0261345 | 0.029776599999999997 | 0.031747809999999994 | 36924.18578477926 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.027225 | 0.02904385 | 0.03484758999999998 | 36059.711998292216 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.046517 | 0.05563614999999999 | 0.0628851 | 41331.690536406815 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.04875 | 0.058000199999999995 | 0.060562559999999994 | 39505.7982660115 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.045858499999999996 | 0.0530459 | 0.06549840999999999 | 42500.99664837141 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.047077499999999994 | 0.06714514999999997 | 0.08128686999999998 | 39192.754670600574 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.1184125 | 0.16929394999999997 | 0.18336099999999997 | 15835.453172664713 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.047066 | 0.0538711 | 0.05558291 | 84942.24564363075 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0461535 | 0.0582334 | 0.06018672 | 81257.87185633609 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.046023999999999995 | 0.05409834999999999 | 0.06298779 | 83829.35863415149 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.052084 | 0.0635717 | 0.07117129999999999 | 73851.34389136762 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.12670599999999999 | 0.16578454999999997 | 0.19561152999999998 | 30352.79969671483 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0478 | 0.05573189999999999 | 0.057470719999999996 | 167325.72084966328 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0533695 | 0.06744865 | 0.1392837199999999 | 137064.78504129764 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.048908 | 0.05746265 | 0.05930282 | 158452.05023088443 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.09801299999999999 | 0.1113155 | 0.11636157999999999 | 80008.1288258887 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.1280015 | 5.98130985 | 6.778436939999997 | 11305.170326381114 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.046338000000000004 | 0.05860484999999999 | 0.05994615 | 324013.22784002655 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0516575 | 0.05991845 | 0.07245313999999999 | 298039.05205699103 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0546575 | 0.06572979999999999 | 0.08918543999999995 | 280630.26751781825 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.055576 | 0.0641786 | 0.08505417 | 277062.6099159323 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.1451145 | 0.1661307 | 0.20057385999999988 | 108252.65465955283 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.049945500000000004 | 0.06658579999999997 | 0.13215469999999982 | 583097.3914773756 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.057296 | 0.06969209999999999 | 0.07793445999999997 | 532004.5632691415 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.05965 | 0.06907795 | 0.08814084 | 515916.1752296391 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.0690705 | 0.07397855 | 0.0757882 | 458710.59890975954 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.1270225 | 0.16790785 | 0.19528250999999996 | 240906.67636717547 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.059066 | 0.0693808 | 0.07078002 | 1042250.5576854742 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.071308 | 0.08384105 | 0.08830917999999999 | 884557.7501494763 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.07614 | 0.0873757 | 0.09491307999999998 | 823080.1975906899 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.075284 | 0.08702115 | 0.09328525999999998 | 826837.704936247 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.141102 | 0.16579669999999994 | 0.19465434999999998 | 444577.50896275206 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.074196 | 0.09805414999999996 | 0.15480924999999984 | 1573771.5237846058 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.08651149999999999 | 0.10600480000000001 | 0.16700297999999997 | 1406807.983459455 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1061485 | 0.11931344999999999 | 0.12230210999999999 | 1211041.137742873 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.10021250000000001 | 0.10686 | 0.13926885999999988 | 1262964.378681714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.16557 | 0.1901407 | 0.19708194 | 764713.7192628399 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 1 | ok | 42.190972 | 0.018061 | 0.01981115 | 0.022140389999999992 | 53677.72964674686 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 2 | ok | 41.518931 | 0.018000000000000002 | 0.0204805 | 0.023434869999999997 | 54397.13827534961 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 4 | ok | 42.277681 | 0.0181775 | 0.01975805 | 0.02364926999999999 | 53784.0870097446 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 8 | ok | 42.036084 | 0.0181575 | 0.0192239 | 0.02344616999999999 | 54340.3247595169 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 1 | 128 | ok | 43.336371 | 0.0182815 | 0.021081249999999996 | 0.02367367 | 53531.700402344264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 1 | ok | 42.925314 | 0.019277 | 0.020091099999999997 | 0.023360299999999997 | 103752.30589499851 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 2 | ok | 41.306089 | 0.019171 | 0.019502 | 0.022043419999999994 | 104007.94620709021 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 4 | ok | 42.514438 | 0.018973 | 0.0195116 | 0.023983349999999983 | 105333.45411569407 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 8 | ok | 42.604765 | 0.019471500000000003 | 0.0199486 | 0.02517904999999999 | 102043.6277326008 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 2 | 128 | ok | 44.832005 | 0.019426 | 0.02177885 | 0.02408418999999999 | 100160.15608958725 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 1 | ok | 42.588232 | 0.0190005 | 0.019719699999999996 | 0.02712272999999999 | 207853.3220676833 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 2 | ok | 42.566312 | 0.0193225 | 0.019811 | 0.02220488999999999 | 206190.66863510024 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 4 | ok | 42.819089 | 0.019133999999999998 | 0.020672649999999997 | 0.022761829999999997 | 210271.99734216192 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 8 | ok | 42.560045 | 0.019358 | 0.019809900000000002 | 0.022575979999999995 | 206495.30997527216 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 4 | 128 | ok | 44.732907 | 0.019401 | 0.0200205 | 0.023708479999999987 | 204856.53384793297 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 1 | ok | 42.647695 | 0.0196255 | 0.0199939 | 0.024126049999999986 | 405174.4833265635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 2 | ok | 41.705533 | 0.01924 | 0.01955455 | 0.020856409999999995 | 415286.70355796884 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 4 | ok | 42.396825 | 0.02141 | 0.02184395 | 0.022239969999999998 | 386284.5797127202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 8 | ok | 42.590805 | 0.0195475 | 0.0203573 | 0.026442219999999985 | 407241.98420511966 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 8 | 128 | ok | 43.146428 | 0.019373 | 0.01986535 | 0.024317409999999994 | 408956.1394540435 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 1 | ok | 41.90802 | 0.020515 | 0.02104685 | 0.022462519999999996 | 779537.9094157461 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 2 | ok | 42.911028 | 0.020610499999999997 | 0.021885449999999997 | 0.025511779999999998 | 769572.6370753163 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 4 | ok | 42.633306 | 0.020470500000000003 | 0.020860100000000003 | 0.024420089999999988 | 779156.7770569496 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 8 | ok | 42.865216 | 0.020436 | 0.0209458 | 0.021928159999999995 | 782077.1382233358 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 16 | 128 | ok | 42.612543 | 0.020181499999999998 | 0.02064015 | 0.021594489999999997 | 790914.762127442 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 1 | ok | 40.813118 | 0.0218965 | 0.0224652 | 0.02582591999999999 | 1470299.0404460886 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 2 | ok | 41.725262 | 0.021774 | 0.02244295 | 0.025848999999999997 | 1461490.190203814 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 4 | ok | 41.732363 | 0.0217875 | 0.022461 | 0.026795449999999985 | 1463393.6647859009 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 8 | ok | 42.669479 | 0.021996 | 0.02257235 | 0.023743819999999995 | 1462119.2321680852 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 32 | 128 | ok | 48.740697 | 0.0216555 | 0.022085599999999997 | 0.02427375999999999 | 1488603.3458220952 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 1 | ok | 42.103821 | 0.0247905 | 0.025321149999999997 | 0.02909569 | 2578428.5444932017 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 2 | ok | 43.644932 | 0.0366825 | 0.0408508 | 0.04341110999999999 | 1716577.038099964 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 4 | ok | 42.381725 | 0.033229499999999995 | 0.03714324999999999 | 0.04151389 | 1898990.3305786103 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 8 | ok | 41.420151 | 0.024501000000000002 | 0.029875549999999994 | 0.034545379999999994 | 2494395.4053236633 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 64 | 128 | ok | 47.283239 | 0.024695 | 0.02604295 | 0.028159719999999992 | 2572268.6927569737 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 1 | ok | 45.309949 | 0.031285 | 0.032890899999999994 | 0.03700046 | 4072954.2473594206 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 2 | ok | 44.902285 | 0.0519065 | 0.05559295 | 0.05744753999999999 | 2460296.9578428115 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 4 | ok | 44.191631 | 0.107442 | 0.11287695 | 0.116522 | 1198484.3667077522 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 8 | ok | 43.82698 | 0.044761999999999996 | 0.04823345 | 0.050301139999999994 | 2848408.1626476664 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `fp32` | 128 | 128 | ok | 84.309497 | 0.1992565 | 0.22333035 | 0.23815806999999997 | 637340.5690077016 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 1 | ok | 41.020125 | 0.0159395 | 0.017343349999999997 | 0.020233059999999997 | 62030.12182715928 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 2 | ok | 42.341716 | 0.015982 | 0.0166686 | 0.022637089999999978 | 61577.11301863324 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 4 | ok | 41.08711 | 0.015961999999999997 | 0.016818 | 0.02302837 | 61373.33893058184 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 8 | ok | 40.990427 | 0.0157235 | 0.0169443 | 0.020028069999999995 | 62738.32722052898 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 1 | 128 | ok | 44.558582 | 0.0160675 | 0.017254799999999997 | 0.020602469999999998 | 61337.27527555771 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 1 | ok | 41.995257 | 0.028536 | 0.03128205 | 0.03177448 | 69617.9090678719 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 2 | ok | 44.079027 | 0.033989500000000006 | 0.0357323 | 0.041644709999999995 | 58446.89077152818 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 4 | ok | 43.371436 | 0.035194500000000004 | 0.037120549999999995 | 0.042704639999999995 | 56572.7653050545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 8 | ok | 44.895601 | 0.0340295 | 0.035643749999999995 | 0.04101701999999998 | 58593.738555910444 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 2 | 128 | ok | 69.42924 | 0.1204625 | 0.16902075 | 0.19553040999999996 | 15850.364946727714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 1 | ok | 43.05281 | 0.031275 | 0.034221049999999996 | 0.042102119999999986 | 125156.91548278653 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 2 | ok | 42.785808 | 0.034751000000000004 | 0.0370253 | 0.039365269999999994 | 114446.85545819944 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 4 | ok | 49.208262 | 0.0369285 | 0.03920005 | 0.042910319999999995 | 107377.82282873988 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 8 | ok | 43.070078 | 0.0377135 | 0.04305484999999999 | 0.04612164 | 104217.25545942092 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 4 | 128 | ok | 104.780353 | 0.12683699999999998 | 0.17239355 | 0.18550029999999998 | 29816.19508378649 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 1 | ok | 42.463278 | 0.033146499999999995 | 0.03650584999999999 | 0.04071331999999999 | 237581.60928278868 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 2 | ok | 43.066843 | 0.036028000000000004 | 0.03749045 | 0.03991672999999999 | 221737.4794546367 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 4 | ok | 42.670069 | 0.0374465 | 0.04110145 | 0.04269358 | 213046.54427852854 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 8 | ok | 42.967125 | 0.038451 | 0.0412617 | 0.04383668999999999 | 207160.71735613202 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 8 | 128 | ok | 77.108323 | 0.10248 | 0.14848639999999996 | 0.17506447 | 73975.63168716594 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 1 | ok | 43.375329 | 0.034537 | 0.03617315 | 0.038661829999999994 | 461245.03873305215 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 2 | ok | 43.829005 | 0.038925 | 0.0406669 | 0.04546248 | 414201.0915234264 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 4 | ok | 43.091105 | 0.0404945 | 0.044314349999999995 | 0.047185899999999996 | 391578.7082993149 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 8 | ok | 44.813062 | 0.04203 | 0.04608044999999999 | 0.04990549999999999 | 377453.92231243366 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 16 | 128 | ok | 84.435221 | 0.1176885 | 0.16133229999999998 | 0.18791441999999994 | 130799.4643107939 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 1 | ok | 45.268915 | 0.037868 | 0.0397844 | 0.04010976 | 841082.6836845729 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 2 | ok | 43.211356 | 0.0468555 | 0.0504344 | 0.05267507 | 678565.0215847292 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 4 | ok | 44.455558 | 0.0500005 | 0.0554838 | 0.05687065 | 635230.6879945371 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 8 | ok | 44.88572 | 0.050994 | 0.055419949999999996 | 0.05794903 | 626782.4124855057 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 32 | 128 | ok | 80.520816 | 0.129184 | 0.1691993 | 0.18824235999999997 | 239345.10392663593 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 1 | ok | 44.29246 | 0.045919 | 0.04793035 | 0.04851802 | 1385138.6740240615 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 2 | ok | 45.259576 | 0.054373000000000005 | 0.0576297 | 0.06209017999999999 | 1163836.966808097 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 4 | ok | 45.267662 | 0.060908500000000004 | 0.06407045 | 0.07252149999999997 | 1042364.9697616437 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 8 | ok | 44.437206 | 0.06336549999999999 | 0.0682623 | 0.10440732999999985 | 988942.6935523101 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 64 | 128 | ok | 100.037963 | 0.156841 | 0.22131584999999995 | 0.24043734999999997 | 389697.4230892557 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 1 | ok | 43.957133 | 0.06369649999999999 | 0.0678667 | 0.06919292 | 1996022.9243232862 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 2 | ok | 44.413577 | 0.0729835 | 0.07940944999999999 | 0.08420082 | 1737635.0298533842 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 4 | ok | 44.463323 | 0.0851775 | 0.08891109999999999 | 0.09120231999999999 | 1520870.2609742077 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 8 | ok | 46.416184 | 0.081859 | 0.0876945 | 0.09128731 | 1568315.3254793407 | - |
| `full_mlp_capacity_search_hd64_depth2` | `jit` | `bf16` | 128 | 128 | ok | 84.17865 | 0.2207325 | 0.29515079999999994 | 0.31663531 | 551246.6442860529 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 1 | ok | 535.333696 | 0.027319 | 0.03139525 | 0.03603454 | 35806.564345828396 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 2 | ok | 516.995962 | 0.026119 | 0.028621349999999997 | 0.03004463 | 37971.47279192089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 4 | ok | 514.151993 | 0.026543 | 0.0302455 | 0.03443094999999999 | 36735.29063859895 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 8 | ok | 532.322594 | 0.026577499999999997 | 0.029838999999999997 | 0.03179665999999999 | 37151.29791774405 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 1 | 128 | ok | 720.931287 | 0.0270905 | 0.0292793 | 0.030300759999999996 | 36699.53721883567 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 1 | ok | 487.175771 | 0.026938 | 0.0298589 | 0.03172269 | 73340.45224656473 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 2 | ok | 510.048902 | 0.0272315 | 0.030380449999999996 | 0.03308662 | 72244.78266240808 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 4 | ok | 509.631649 | 0.0283365 | 0.031740950000000004 | 0.0320295 | 69408.48699215545 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 8 | ok | 547.11583 | 0.026615 | 0.028878150000000002 | 0.029232730000000002 | 74116.3478428437 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 2 | 128 | ok | 836.124085 | 0.028637 | 0.03110195 | 0.03302797 | 69711.52673123349 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 1 | ok | 497.547192 | 0.026915 | 0.02989145 | 0.03224090999999999 | 145753.89745921805 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 2 | ok | 516.516156 | 0.0298245 | 0.0316733 | 0.033266159999999996 | 133236.6035591494 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 4 | ok | 512.665433 | 0.0282515 | 0.030251649999999998 | 0.03486643999999999 | 141100.8405377071 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 8 | ok | 536.944465 | 0.0284395 | 0.0294939 | 0.030468359999999996 | 141397.31656172627 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 4 | 128 | ok | 739.371483 | 0.0292905 | 0.032450549999999995 | 0.033989280000000004 | 133836.55879440028 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 1 | ok | 499.396355 | 0.0274495 | 0.0298967 | 0.03028169 | 289361.4804890788 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 2 | ok | 507.174209 | 0.029061 | 0.0325238 | 0.03460761999999999 | 275357.31053007656 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 4 | ok | 514.249032 | 0.028067 | 0.0309846 | 0.03449081 | 281386.75836123165 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 8 | ok | 539.523861 | 0.029529 | 0.03409369999999999 | 0.03732971999999999 | 269047.015293305 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 8 | 128 | ok | 744.686357 | 0.029706 | 0.03175 | 0.03224804 | 269853.6381330176 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 1 | ok | 496.682606 | 0.0304345 | 0.03352685 | 0.035959979999999996 | 521036.5240090374 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 2 | ok | 508.188323 | 0.028879000000000002 | 0.0330461 | 0.036806729999999996 | 540409.4547336153 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 4 | ok | 510.521428 | 0.029199 | 0.03198865 | 0.038052419999999997 | 535887.7315202465 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 8 | ok | 541.594352 | 0.029568499999999998 | 0.03233075 | 0.03260986 | 537527.4810924708 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 16 | 128 | ok | 835.437792 | 0.030445 | 0.0342885 | 0.03678794999999999 | 512746.88762639207 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 1 | ok | 500.224258 | 0.0312725 | 0.0343261 | 0.03533789 | 1017295.938382385 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 2 | ok | 510.553983 | 0.0301135 | 0.033819249999999995 | 0.039516159999999995 | 1043665.6669219298 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 4 | ok | 506.918001 | 0.031616000000000005 | 0.03435965 | 0.03805673999999999 | 1009594.3008401719 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 8 | ok | 547.561983 | 0.032107 | 0.03478535 | 0.03647789 | 1000440.1936852214 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 32 | 128 | ok | 851.590847 | 0.030981 | 0.035105899999999995 | 0.036869849999999996 | 1012839.64153073 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 1 | ok | 499.30313 | 0.036362 | 0.04150075 | 0.04449294999999999 | 1723175.1440601347 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 2 | ok | 516.489429 | 0.06281300000000001 | 0.06847595 | 0.07094742999999999 | 1015394.6521702156 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 4 | ok | 520.7931 | 0.045838000000000004 | 0.0492115 | 0.05213102 | 1392140.8428890752 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 8 | ok | 549.336808 | 0.0348585 | 0.03756325 | 0.041867869999999995 | 1828755.3491093963 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 64 | 128 | ok | 731.0438 | 0.035757 | 0.0391573 | 0.040105119999999994 | 1782575.877848988 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 1 | ok | 501.409138 | 0.0426835 | 0.0448371 | 0.04737972999999999 | 3008177.542638567 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 2 | ok | 512.675873 | 0.08664749999999999 | 0.09325235 | 0.09798193 | 1469834.4047813714 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 4 | ok | 512.099219 | 0.095005 | 0.10334805 | 0.10916382999999999 | 1341226.4509974534 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 8 | ok | 542.162706 | 0.0594735 | 0.0638849 | 0.06600745999999999 | 2148406.553714192 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `fp32` | 128 | 128 | ok | 841.419122 | 0.13682699999999998 | 0.14852755 | 0.16096943999999996 | 925354.8085050517 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 1 | ok | 500.448175 | 0.0308125 | 0.03321735 | 0.03648806999999999 | 32238.014550950247 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 2 | ok | 500.167131 | 0.0285305 | 0.029747600000000003 | 0.03160784999999999 | 35000.40250462881 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 4 | ok | 506.485386 | 0.0304375 | 0.03345115 | 0.03436699 | 32785.74580686704 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 8 | ok | 510.832365 | 0.029200499999999997 | 0.031218999999999997 | 0.03493285999999999 | 33907.822263333575 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 1 | 128 | ok | 597.901693 | 0.030673 | 0.0327294 | 0.03438715 | 32478.85788745816 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 1 | ok | 496.546673 | 0.0444585 | 0.0488288 | 0.050952109999999995 | 44707.56335732341 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 2 | ok | 517.151588 | 0.0465835 | 0.05033715 | 0.05242876 | 42723.59500253564 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 4 | ok | 506.276203 | 0.047477500000000006 | 0.050931699999999996 | 0.05308876999999999 | 41934.27724595796 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 8 | ok | 509.881656 | 0.047092999999999996 | 0.05484535 | 0.056702459999999996 | 41934.87513890927 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 2 | 128 | ok | 593.637352 | 0.126466 | 0.18540879999999998 | 0.20353778999999997 | 14083.632270164628 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 1 | ok | 511.791378 | 0.045167 | 0.05107435 | 0.05261181 | 88047.93329488575 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 2 | ok | 500.774509 | 0.0542045 | 0.05991955 | 0.06143634 | 72990.32987614635 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 4 | ok | 507.923051 | 0.0477685 | 0.0527958 | 0.05505110999999999 | 83029.2732157113 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 8 | ok | 500.515954 | 0.048577 | 0.05667745 | 0.05756603 | 80578.03458892713 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 4 | 128 | ok | 621.341821 | 0.14149299999999998 | 0.29838509999999996 | 0.32915245 | 23308.113565986376 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 1 | ok | 541.526743 | 0.044385 | 0.0477678 | 0.05336266999999999 | 180237.7425943691 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 2 | ok | 506.215423 | 0.046457 | 0.0517199 | 0.05538408 | 169007.49083451251 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 4 | ok | 500.49797 | 0.0486095 | 0.0535639 | 0.057412029999999996 | 162577.50374436312 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 8 | ok | 502.735496 | 0.0487505 | 0.05298625 | 0.058130329999999994 | 162704.4737628867 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 8 | 128 | ok | 537.474403 | 0.121169 | 0.15180195 | 0.16523778999999997 | 64839.1955530686 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 1 | ok | 528.785332 | 0.0461595 | 0.0509989 | 0.05355177999999999 | 345311.46878664894 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 2 | ok | 512.045117 | 0.056165999999999994 | 0.0614176 | 0.06472462 | 282241.2068634005 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 4 | ok | 504.788207 | 0.050576499999999996 | 0.0559415 | 0.05667323 | 314133.1241206727 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 8 | ok | 507.712509 | 0.056732 | 0.06371639999999999 | 0.06713722 | 279267.59281109367 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 16 | 128 | ok | 644.023082 | 5.992046 | 6.016985 | 6.058054869999999 | 3342.1043240705544 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 1 | ok | 532.349631 | 0.053955 | 0.0582936 | 0.06010541999999999 | 589241.7710545292 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 2 | ok | 511.860651 | 0.053043 | 0.0604626 | 0.06121429 | 595877.4220555089 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 4 | ok | 505.345895 | 0.053478 | 0.059562 | 0.06456395 | 593255.5006835416 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 8 | ok | 504.795795 | 0.054355 | 0.06341784999999998 | 0.06909601 | 580156.4029142706 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 32 | 128 | ok | 527.559677 | 0.13239250000000002 | 0.16262904999999997 | 0.18863711999999994 | 236312.7995726283 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 1 | ok | 514.195531 | 0.0555895 | 0.0601243 | 0.06419303 | 1132982.3920373996 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 2 | ok | 513.081148 | 0.06691849999999999 | 0.07509489999999999 | 0.08135121999999999 | 948151.5193535502 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 4 | ok | 506.244414 | 0.0663625 | 0.07331085 | 0.07612132999999999 | 958882.8056431452 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 8 | ok | 508.122093 | 0.065291 | 0.07079144999999999 | 0.07307962999999999 | 972045.785786625 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 64 | 128 | ok | 532.925936 | 0.133184 | 0.14926735 | 0.18694395999999996 | 469424.06361634907 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 1 | ok | 489.940773 | 0.0590175 | 0.06703764999999999 | 0.0695609 | 2136842.0279432163 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 2 | ok | 505.996246 | 0.0825215 | 0.09126954999999999 | 0.09533466999999998 | 1546972.116311032 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 4 | ok | 509.664209 | 0.0893105 | 0.0992027 | 0.09969306 | 1436057.519488871 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 8 | ok | 510.404413 | 0.0677845 | 0.07723624999999999 | 0.08121970999999999 | 1847108.4815469536 | - |
| `full_mlp_capacity_search_hd64_depth2` | `compile` | `bf16` | 128 | 128 | ok | 528.465894 | 0.18529099999999998 | 0.2138298 | 0.31666341999999986 | 673213.1738982209 | - |
