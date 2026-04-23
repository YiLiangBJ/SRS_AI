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

### full_mlp_capacity_search_hd512_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`8`, batch=`128`, precision=`bf16`, throughput=`385855.362` samples/s, p50=`0.334` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.060` ms, throughput=`16419.671` samples/s

### full_mlp_capacity_search_hd512_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`504102.766` samples/s, p50=`0.253` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.039` ms, throughput=`25102.645` samples/s

### full_mlp_capacity_search_hd512_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`4`, batch=`128`, precision=`bf16`, throughput=`455155.194` samples/s, p50=`0.274` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`14537.593` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.084557 | 0.09417355 | 0.10322172999999997 | 11683.490503542083 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0654645 | 0.08032399999999999 | 0.08976276999999999 | 14656.219322700928 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.06607450000000001 | 0.07200904999999999 | 0.07321493 | 14887.376993047596 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.059813500000000006 | 0.06686519999999999 | 0.06987522999999998 | 16419.67102860701 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.1141005 | 0.1356936 | 0.18182051999999996 | 8526.866194198696 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.09923950000000001 | 0.12399335 | 0.22234851999999983 | 18985.72918679945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0754135 | 0.08454695 | 0.08812463 | 25891.180920063827 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.070842 | 0.08210895 | 0.20710348999999956 | 25815.165480373762 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.0579825 | 0.06530255 | 0.06947286 | 33355.43686949886 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.1409715 | 0.2856809999999999 | 0.31229132 | 13194.79850487099 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.130816 | 0.16141184999999997 | 0.27407670999999983 | 28830.43709969839 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.091563 | 0.11415494999999996 | 0.19468526999999988 | 41457.53069826504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0864105 | 0.09824685 | 0.14909660999999982 | 44835.288600269196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.071258 | 0.08012399999999999 | 0.08352511 | 55209.941322874365 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.223988 | 0.24528999999999995 | 0.25649250999999995 | 17731.63612022936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.14338450000000003 | 0.16919659999999997 | 0.17555026999999998 | 54266.02889611772 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.106686 | 0.16016925 | 0.2439173399999997 | 64701.65581242472 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.1152415 | 0.12582649999999998 | 0.12981645 | 68002.63510211022 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0873235 | 0.0935027 | 0.09778310999999999 | 90902.12863242066 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.271914 | 11.9917934 | 12.00230795 | 4996.689256157486 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.216345 | 0.26255944999999997 | 0.3526990299999997 | 70770.47551301739 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.1756825 | 0.2143241 | 0.2871606799999999 | 84823.91033349262 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.13296000000000002 | 0.18758995 | 0.19766557999999998 | 111440.51068728426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1137145 | 0.12912005 | 0.14268725999999995 | 138053.23539836815 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.329217 | 0.36021434999999996 | 0.37534370999999994 | 48311.4659534606 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.2927735 | 0.34432514999999997 | 0.4539406499999997 | 105472.56256533532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.22425 | 0.2752042 | 0.27831184999999997 | 133345.23439550312 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.172737 | 0.21883995 | 0.26321140999999987 | 178267.77647270908 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.159749 | 0.1776933 | 0.19664555999999994 | 199933.87187187842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.45183799999999996 | 0.46643189999999995 | 0.47120058 | 70758.64500378957 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.4229305 | 0.4456803 | 0.48844888999999997 | 149936.42461320735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.3117895 | 0.34120395 | 0.34707566 | 203049.71792904264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.2401605 | 0.295465 | 0.30813225 | 254456.66940870232 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.2017015 | 0.2225181 | 0.22716178999999997 | 312864.2852414975 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.4171735 | 0.4317139 | 0.43563084 | 153123.7410656484 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6957344999999999 | 0.7116256 | 0.7214962899999999 | 184202.4576119205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4859805 | 0.5033966 | 0.51348531 | 262468.00966079126 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.3690735 | 0.3758283 | 0.39361559999999995 | 346140.8138689971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.6734555 | 0.6922543999999999 | 0.7132938799999999 | 189990.87687558102 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 1.0110074999999998 | 1.0474548000000001 | 1.05256697 | 127195.6225149528 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.26861199999999996 | 0.2960438 | 0.30737186999999994 | 3676.761214581299 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.196367 | 0.2228343 | 0.22915029999999997 | 4987.789890348427 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.18369049999999998 | 0.2036518 | 0.20844604 | 5377.686772977081 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.157776 | 0.18406504999999998 | 0.19448500999999999 | 6194.371447218256 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.6438845 | 0.69764245 | 0.7009540400000001 | 1547.4324567396686 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.3047425 | 0.34977844999999996 | 0.36944381 | 6474.381423031486 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.19749850000000002 | 0.2638307 | 0.27043790999999995 | 9711.504228486056 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1824815 | 0.21816755 | 0.24153135999999994 | 10655.734735287042 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1481515 | 0.17067865 | 0.2258001499999998 | 13159.880839910971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.38182499999999997 | 0.4127357 | 0.42523314999999995 | 5199.645051430209 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.264291 | 0.2886118 | 0.29620735 | 15023.087104685304 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.2178965 | 0.23834224999999998 | 0.25929727999999996 | 18113.82054619871 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.180938 | 0.1970509 | 0.20116606 | 22185.337865502057 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.161988 | 0.17992465 | 0.18864548999999997 | 24638.30657962281 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.5342359999999999 | 0.5636827 | 0.57141309 | 7471.66148232982 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.27190000000000003 | 0.29892124999999997 | 0.30873831999999996 | 29149.7810487071 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2181755 | 0.2352274 | 0.23700087 | 36772.51328177214 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.16744399999999998 | 0.19334625 | 0.20312639999999998 | 47105.624352665676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.170321 | 0.18601535 | 0.18740534 | 46068.50248113437 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.6763334999999999 | 0.72186695 | 0.8135858599999997 | 11716.405345914565 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.3065945 | 0.33013889999999996 | 0.34837003 | 51808.52553324897 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.22287800000000002 | 0.24955745 | 0.26436015999999996 | 70606.41021477059 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.19100699999999998 | 0.21618445 | 0.22240505 | 83023.56893830912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.18185 | 0.20888705 | 0.23586247999999993 | 86518.43751162592 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.5397055 | 0.55991605 | 0.56360689 | 29734.66015829469 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.3061205 | 0.34041425 | 0.34516036 | 102964.74101149986 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.2730305 | 0.29534965 | 0.29936306 | 116537.53579341156 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.2086945 | 0.23252155 | 0.23813724999999997 | 151657.57463709053 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1980035 | 0.2252416 | 0.2689936199999999 | 156411.00863691812 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.734974 | 0.7706871 | 0.8381510099999998 | 43360.205640111264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.3473925 | 0.38098725 | 0.4480789099999998 | 180785.4551773819 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.32475200000000004 | 0.3530722 | 0.36477458999999995 | 194986.4712902224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.256029 | 0.28283929999999996 | 0.3204432099999999 | 247012.99535368558 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.2582075 | 0.2693168 | 0.3021829399999999 | 247549.2623031983 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.9358005 | 0.9954795999999999 | 1.0938040399999998 | 68118.62944466075 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.42563399999999996 | 0.451593 | 0.4793904899999999 | 299005.0606606517 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.3753725 | 0.40340515 | 0.41182818 | 338786.18520985503 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3446005 | 0.3579715 | 0.37154740999999997 | 372269.1801661263 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.33447 | 0.34177835 | 0.34661132 | 385855.36235857947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 1.180812 | 1.2185354000000002 | 1.2307586899999998 | 108483.5693752172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 1 | ok | 50.493681 | 0.07150300000000001 | 0.07767725 | 0.08317869999999998 | 13981.582899741145 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 2 | ok | 51.49385 | 0.043046 | 0.04823205 | 0.04882224 | 22902.487164301067 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 4 | ok | 49.927046 | 0.0392275 | 0.0440826 | 0.04649946 | 25102.644714236514 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 8 | ok | 50.430596 | 0.0399815 | 0.04246615 | 0.045637029999999995 | 25259.656640435358 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 1 | 128 | ok | 56.166345 | 0.084577 | 0.10144734999999996 | 0.12639161999999995 | 11551.28702129734 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 1 | ok | 50.287641 | 0.0721105 | 0.08110575 | 0.08494489999999999 | 28074.730440475676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 2 | ok | 51.395688 | 0.052207500000000004 | 0.05867479999999999 | 0.06043696999999999 | 37752.374624363874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 4 | ok | 51.392209 | 0.043051 | 0.04601305 | 0.04919238 | 46246.81648477023 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 8 | ok | 50.005408 | 0.037834 | 0.042218900000000004 | 0.04540699999999999 | 52418.26420063196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 2 | 128 | ok | 67.88841 | 0.11198749999999999 | 0.1439077 | 0.15573090999999994 | 17533.696257467604 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 1 | ok | 51.377239 | 0.0765985 | 0.08366885 | 0.08637286 | 51827.489092904914 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 2 | ok | 50.461867 | 0.0542865 | 0.06173159999999999 | 0.06482913 | 71976.85512246682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 4 | ok | 50.212659 | 0.0471705 | 0.053589399999999995 | 0.057141439999999995 | 83106.55631778119 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 8 | ok | 51.405574 | 0.048247 | 0.05308565 | 0.0542589 | 82416.55221067979 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 4 | 128 | ok | 62.015152 | 0.143017 | 0.1650865 | 0.18656455999999993 | 27232.396094111893 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 1 | ok | 50.278443 | 0.0857655 | 0.0935618 | 0.09613507 | 92192.72797590635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 2 | ok | 52.178188 | 0.0749405 | 0.08210514999999999 | 0.08760111999999999 | 105491.07413648962 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 4 | ok | 50.135899 | 0.056666499999999995 | 0.0669271 | 0.07018827999999999 | 137053.18520168232 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 8 | ok | 51.226777 | 0.0530205 | 0.0590021 | 0.061082149999999995 | 147529.39892096998 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 8 | 128 | ok | 64.090529 | 0.195933 | 0.2159404 | 0.23292541999999997 | 40433.54872621686 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 1 | ok | 52.197376 | 0.116224 | 0.1260064 | 0.13405954999999997 | 135904.92634972278 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 2 | ok | 52.372426 | 0.103409 | 0.1095407 | 0.11186814999999999 | 153795.31266490943 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 4 | ok | 50.41891 | 0.0829835 | 0.0880123 | 0.08910885 | 192275.84662059575 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 8 | ok | 50.028691 | 0.0763885 | 0.08028 | 0.08216216 | 211233.9495203669 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 16 | 128 | ok | 94.337367 | 0.2658425 | 0.2927677 | 0.29706763999999997 | 59564.897780306645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 1 | ok | 52.063776 | 0.17588399999999998 | 0.18592904999999998 | 0.18713423999999998 | 181092.6177820727 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 2 | ok | 52.747479 | 0.157686 | 0.1642318 | 0.16560121 | 202159.00766207906 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 4 | ok | 50.318126 | 0.1337415 | 0.13932809999999998 | 0.14029033 | 238322.9216006959 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 8 | ok | 51.122204 | 0.1300195 | 0.13608535 | 0.13875405 | 246208.20134906704 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 32 | 128 | ok | 64.285566 | 47.5869675 | 58.402736749999995 | 60.46697036 | 725.0619842801776 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 1 | ok | 54.201943 | 0.295985 | 0.30943529999999997 | 0.3378630399999999 | 213779.3054684479 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 2 | ok | 54.171649 | 0.2482415 | 0.25705695 | 0.27690830999999994 | 256863.82276139368 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 4 | ok | 52.718565 | 0.2250445 | 0.23271825000000002 | 0.23470992 | 283282.5829635094 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 8 | ok | 51.919136 | 0.200538 | 0.2084146 | 0.21151244 | 318080.9539883972 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 64 | 128 | ok | 84.178654 | 24.6675105 | 32.36935405 | 45.44796400999997 | 2593.3920176884894 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 1 | ok | 55.66035 | 0.5336974999999999 | 0.54266095 | 0.5696629799999999 | 240028.25132518102 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 2 | ok | 53.790101 | 0.454587 | 0.4664159 | 0.47993338999999996 | 281141.7041606468 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 4 | ok | 53.52048 | 0.321621 | 0.33266415 | 0.33465460999999996 | 398210.8387017132 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 8 | ok | 54.042039 | 0.26488 | 0.27577455 | 0.27747495 | 482600.44423370896 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `fp32` | 128 | 128 | ok | 161.737591 | 11.360308 | 12.497531500000001 | 12.64271423 | 11213.397818173444 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 1 | ok | 51.247724 | 0.10323550000000001 | 0.11441829999999999 | 0.11752739 | 9592.419992982184 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 2 | ok | 50.288835 | 0.0924625 | 0.10096615 | 0.102555 | 10707.876285480546 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 4 | ok | 48.851165 | 0.082958 | 0.09352490000000001 | 0.09493286 | 11856.013927496682 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 8 | ok | 50.250895 | 0.084959 | 0.0974476 | 0.10190039999999999 | 11521.063845818671 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 1 | 128 | ok | 58.754608 | 0.2969335 | 0.33762875 | 0.35658267 | 3292.563061472548 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 1 | ok | 50.277502 | 0.0988175 | 0.11141374999999999 | 0.11358761 | 19849.72566686642 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 2 | ok | 51.657064 | 0.0985375 | 0.10807885 | 0.11457649999999998 | 20093.25682356955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 4 | ok | 50.329608 | 0.090146 | 0.10254465 | 0.10695985999999999 | 21967.59340620716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 8 | ok | 51.633271 | 0.0877385 | 0.0998175 | 0.10530740000000001 | 22290.470690037186 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 2 | 128 | ok | 69.38131 | 0.29375450000000003 | 0.3120727 | 0.31631947 | 6777.857417602435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 1 | ok | 51.631911 | 0.100424 | 0.10783775 | 0.11241135999999999 | 39415.8181590645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 2 | ok | 50.910532 | 0.098907 | 0.10919904999999999 | 0.11214094999999999 | 39715.746459440496 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 4 | ok | 50.893759 | 0.095683 | 0.10759205000000001 | 0.11126283 | 41356.93766765069 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 8 | ok | 51.511024 | 0.0952305 | 0.10538644999999999 | 0.10797538 | 41503.9366483911 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 4 | 128 | ok | 115.294568 | 0.29769 | 0.3290607 | 0.34210436 | 13273.518434760157 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 1 | ok | 51.838845 | 0.108126 | 0.12176554999999999 | 0.13562708999999995 | 72785.40841803279 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 2 | ok | 52.315645 | 0.1070035 | 0.1178316 | 0.12270565999999998 | 73901.45942449608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 4 | ok | 50.382307 | 0.101338 | 0.11090615 | 0.11680399999999998 | 79451.37238341801 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 8 | ok | 52.455479 | 0.10132450000000001 | 0.11384015 | 0.11855290999999998 | 77887.97411626844 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 8 | 128 | ok | 63.114128 | 0.334118 | 0.35866549999999997 | 0.37332598999999994 | 23968.321070041726 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 1 | ok | 50.901468 | 0.117714 | 0.12776905 | 0.13120751 | 134354.35928673286 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 2 | ok | 51.028697 | 0.12113 | 0.1292254 | 0.13173754999999998 | 131352.98997990135 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 4 | ok | 51.791582 | 0.116336 | 0.1262267 | 0.13152487 | 137365.9630152447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 8 | ok | 51.870457 | 0.1162485 | 0.12938715 | 0.13157712 | 135569.6909553325 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 16 | 128 | ok | 74.155954 | 0.416215 | 0.454862 | 0.46137859 | 37942.76599349244 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 1 | ok | 53.285436 | 0.1234315 | 0.1422664 | 0.14641948 | 254192.79093357865 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 2 | ok | 52.910828 | 0.1678115 | 0.1845701 | 0.19085658 | 193195.69588968912 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 4 | ok | 50.565118 | 0.1426365 | 0.15373045 | 0.16139567 | 221060.4574100087 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 8 | ok | 52.013712 | 0.14171499999999998 | 0.1521537 | 0.1578173 | 225127.68257221885 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 32 | 128 | ok | 74.073672 | 0.3927505 | 0.43854945 | 0.44112681 | 80905.77248528172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 1 | ok | 52.618786 | 0.178042 | 0.18900619999999999 | 0.19811454999999997 | 357912.2086091308 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 2 | ok | 53.689032 | 0.218452 | 0.23316974999999998 | 0.24251133 | 294166.990326962 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 4 | ok | 52.750385 | 0.2213425 | 0.23868240000000002 | 0.24748414999999999 | 290687.6925919513 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 8 | ok | 52.9134 | 0.242354 | 0.25603615 | 0.26370340999999997 | 272273.1798495384 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 64 | 128 | ok | 86.8132 | 0.5064295 | 0.5545453499999999 | 0.5699785099999999 | 128027.7512953708 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 1 | ok | 56.296339 | 0.2525315 | 0.267084 | 0.2740329 | 504102.76638995623 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 2 | ok | 53.790499 | 0.289587 | 0.30066414999999996 | 0.30622111999999996 | 443313.96305296756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 4 | ok | 54.561353 | 0.28646150000000004 | 0.30289309999999997 | 0.31247421999999997 | 443492.35259804054 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 8 | ok | 53.120548 | 0.27163499999999996 | 0.27977240000000003 | 0.28332406 | 469529.23971331125 | - |
| `full_mlp_capacity_search_hd512_depth4` | `jit` | `bf16` | 128 | 128 | ok | 87.220594 | 1.0478049999999999 | 1.08141545 | 1.09208086 | 122447.8777171208 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 1 | ok | 496.346746 | 0.08256749999999999 | 0.087287 | 0.08958407 | 12037.25772009524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 2 | ok | 497.645009 | 0.0937355 | 0.09961874999999999 | 0.10612393999999999 | 10566.323225940747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 4 | ok | 511.497906 | 0.071597 | 0.078032 | 0.08130424 | 13875.194842423572 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 8 | ok | 514.259888 | 0.06403400000000001 | 0.07187124999999998 | 0.1758688999999996 | 14537.592615368154 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 1 | 128 | ok | 649.183897 | 0.08617949999999999 | 0.0914916 | 0.0938696 | 11551.228311413723 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 1 | ok | 496.443888 | 0.084013 | 0.09103615 | 0.09449553 | 23567.546828126357 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 2 | ok | 500.975844 | 0.0824085 | 0.0889786 | 0.09087258 | 24057.927640489877 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 4 | ok | 510.469184 | 0.07137399999999999 | 0.07779765 | 0.15216931999999972 | 26633.47780219476 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 8 | ok | 515.262023 | 0.0582655 | 0.06366575 | 0.06713308999999999 | 33830.4492311523 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 2 | 128 | ok | 615.564431 | 0.192765 | 0.2286585 | 0.23784756999999998 | 11755.695487517507 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 1 | ok | 503.042562 | 0.1149455 | 0.12300589999999999 | 0.12626066 | 34502.87141521635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 2 | ok | 505.345927 | 0.1236915 | 0.130358 | 0.13403254999999997 | 33151.8710418738 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 4 | ok | 513.213081 | 0.0961345 | 0.1029775 | 0.10902121999999999 | 41218.73069804127 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 8 | ok | 513.486048 | 0.088839 | 0.09496945 | 0.09591748 | 44712.35091504944 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 4 | 128 | ok | 560.694507 | 11.997446 | 12.014994699999999 | 12.145389179999999 | 385.24507955160647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 1 | ok | 501.546751 | 0.1441005 | 0.1550519 | 0.1576126 | 55090.1281382608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 2 | ok | 509.514253 | 0.12838850000000002 | 0.18398394999999998 | 0.18747097 | 54008.07420709396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 4 | ok | 507.804324 | 0.125214 | 0.13294555 | 0.13440303 | 63476.92978594627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 8 | ok | 514.679355 | 0.1053335 | 0.11376575 | 0.11449071 | 75867.16645427022 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 8 | 128 | ok | 655.253363 | 0.2712215 | 0.3078667 | 0.31787743 | 28964.03398321143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 1 | ok | 497.537547 | 0.199489 | 0.24417199999999992 | 0.26581931 | 78227.33724704702 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 2 | ok | 508.923871 | 0.2152695 | 0.29283469999999967 | 0.39878752 | 78113.03131385434 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 4 | ok | 506.225717 | 0.194187 | 0.25413435 | 0.25689348999999995 | 83024.23238648694 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 8 | ok | 500.705531 | 0.16351549999999998 | 0.1769634 | 0.28465983999999966 | 94614.15990786473 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 16 | 128 | ok | 547.008609 | 0.39838 | 0.4360206 | 0.45497114 | 39708.72459249666 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 1 | ok | 537.977441 | 0.281814 | 0.2897557 | 0.29614939 | 113510.22056667566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 2 | ok | 510.798387 | 0.22601949999999998 | 0.25166885 | 0.4881491599999998 | 132661.0733441618 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 4 | ok | 520.035045 | 0.188346 | 0.3465721 | 0.34857864 | 145945.33307415876 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 8 | ok | 502.98681 | 0.1728035 | 0.24690515000000002 | 0.24920040999999998 | 161873.45866116084 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 32 | 128 | ok | 678.701956 | 0.510478 | 0.57421825 | 0.7101642999999995 | 61515.15675792055 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 1 | ok | 498.377896 | 0.40449199999999996 | 0.41587085 | 0.41824846 | 158106.0320316892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 2 | ok | 500.078827 | 0.29941799999999996 | 0.3166582 | 0.3887583999999997 | 210206.60813376078 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 4 | ok | 508.767536 | 0.230819 | 0.38603119999999996 | 0.39357468 | 254974.8786000859 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 8 | ok | 522.664893 | 0.2581915 | 0.31731425 | 0.33391408999999994 | 234469.3029125044 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 64 | 128 | ok | 543.524408 | 0.6882355 | 0.7166413500000001 | 0.7475118399999999 | 92995.87609787444 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 1 | ok | 496.798773 | 0.668669 | 0.68815605 | 0.7081993999999999 | 191178.20709433052 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 2 | ok | 505.756166 | 0.45628 | 0.4681665 | 0.47136136 | 280370.0078039239 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 4 | ok | 511.749323 | 0.3339135 | 0.3548943 | 0.45828796 | 377724.07669333264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 8 | ok | 507.706112 | 0.273458 | 0.3324186 | 0.33609962 | 451392.0260046947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `fp32` | 128 | 128 | ok | 629.179941 | 1.0513555 | 1.15201465 | 1.19235027 | 120467.67886998608 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 1 | ok | 503.001307 | 0.2797795 | 0.29546825 | 0.30084204999999997 | 3575.2809366377987 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 2 | ok | 500.486325 | 0.19515749999999998 | 0.26819645 | 0.27603787999999996 | 4699.330711921347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 4 | ok | 506.906179 | 0.1776525 | 0.22497375 | 0.23223730999999997 | 5232.763251685133 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 8 | ok | 512.535284 | 0.188194 | 0.2107358 | 0.24304738999999992 | 5446.912052959672 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 1 | 128 | ok | 553.232156 | 0.7624095 | 0.79355355 | 0.8887908299999996 | 1311.7052722078326 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 1 | ok | 501.517037 | 0.286922 | 0.3028721 | 0.30670613 | 6964.949934198635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 2 | ok | 510.76459 | 0.184851 | 0.255492 | 0.26245621999999996 | 9908.432214532619 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 4 | ok | 505.446155 | 0.2025215 | 0.2166988 | 0.22077389 | 10485.91191519414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 8 | ok | 508.704376 | 0.1889175 | 0.22167285 | 0.26576795999999986 | 10541.497765518718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 2 | 128 | ok | 549.40858 | 0.5998815 | 0.6507509 | 0.6687893699999999 | 3321.8308297192643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 1 | ok | 498.401172 | 0.264595 | 0.27364605000000003 | 0.27427234 | 15079.447956489761 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 2 | ok | 502.635364 | 0.2054915 | 0.2224332 | 0.22922673 | 19318.850184538485 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 4 | ok | 503.983688 | 0.219412 | 0.23407024999999998 | 0.23756272 | 19710.779786043426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 8 | ok | 518.553451 | 0.1955045 | 0.24185005 | 0.24759609999999999 | 19266.128133431037 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 4 | 128 | ok | 665.965121 | 0.93852 | 0.9763215999999999 | 1.0292878599999997 | 4256.05544652842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 1 | ok | 507.122543 | 0.278803 | 0.29385395 | 0.30569763 | 28615.074736136816 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 2 | ok | 500.3887 | 0.218121 | 0.237452 | 0.27191047 | 37960.78461145714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 4 | ok | 524.380463 | 0.22051300000000001 | 0.24260959999999998 | 0.24638203 | 38249.01735883841 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 8 | ok | 518.382754 | 0.16526849999999998 | 0.2152349 | 0.2690020799999998 | 44859.76109709928 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 8 | 128 | ok | 654.437782 | 0.8190390000000001 | 0.8606430999999999 | 0.88429362 | 9752.66961654039 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 1 | ok | 498.26613 | 0.28234 | 0.2910623 | 0.30089414999999997 | 56410.46397183538 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 2 | ok | 518.660664 | 0.214167 | 0.22482885 | 0.22771827 | 74656.01076539676 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 4 | ok | 509.099311 | 0.2397035 | 0.282309 | 0.29039531 | 64563.96885046759 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 8 | ok | 517.322859 | 0.195385 | 0.22998944999999998 | 0.23180258 | 79573.67607307092 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 16 | 128 | ok | 580.331672 | 0.867773 | 0.89182515 | 0.89500112 | 18488.905882247716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 1 | ok | 496.623199 | 0.25012 | 0.2622084 | 0.26628333 | 127719.00796495647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 2 | ok | 504.705411 | 0.2197195 | 0.30471565 | 0.30734752 | 138630.07664423718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 4 | ok | 511.834631 | 0.210818 | 0.25320015 | 0.2557331 | 153745.03691802698 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 8 | ok | 516.919714 | 0.20826499999999998 | 0.24272624999999998 | 0.25569069 | 153743.84027706177 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 32 | 128 | ok | 549.949322 | 0.78235 | 0.8134171499999999 | 0.8191415900000001 | 40882.73594627845 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 1 | ok | 499.97363 | 0.294923 | 0.3049672 | 0.3176044 | 216808.48425801023 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 2 | ok | 511.239223 | 0.26817 | 0.28193575 | 0.28679863 | 237295.10865075636 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 4 | ok | 510.023514 | 0.2265355 | 0.27654799999999996 | 0.27950313 | 267007.2789521833 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 8 | ok | 507.968492 | 0.2267805 | 0.29422709999999996 | 0.30522307 | 263903.46543186234 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 64 | 128 | ok | 615.494535 | 1.1437270000000002 | 1.18934765 | 1.3214427999999998 | 55542.45582877719 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 1 | ok | 503.117467 | 0.34477250000000004 | 0.3666164 | 0.37209807 | 371501.55821684824 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 2 | ok | 509.15635 | 0.33244300000000004 | 0.37062765 | 0.38191639 | 380711.24117158883 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 4 | ok | 505.488844 | 0.2741165 | 0.3243551999999999 | 0.36107996 | 455155.19440887356 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 8 | ok | 513.957649 | 0.30731149999999996 | 0.32117535 | 0.32384219999999997 | 415900.79726233304 | - |
| `full_mlp_capacity_search_hd512_depth4` | `compile` | `bf16` | 128 | 128 | ok | 637.248413 | 0.7459245000000001 | 0.8156821 | 0.8617458399999999 | 169633.71495529794 | - |
