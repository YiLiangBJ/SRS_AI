# Latency Report

- Device: `cpu`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8, 64]`

## CPU Thread Scaling Highlights

### full_mlp_capacity_search_hd64_depth4::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`838156.663` samples/s, p50=`0.151` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.080` ms, throughput=`12386.460` samples/s

### full_mlp_capacity_search_hd64_depth4::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1512666.216` samples/s, p50=`0.084` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.042` ms, throughput=`23611.301` samples/s

### full_mlp_capacity_search_hd64_depth4::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1118263.337` samples/s, p50=`0.114` ms
- Lowest batch-1 p50 latency: threads=`64`, precision=`fp32`, p50=`0.067` ms, throughput=`14839.543` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,120`
- MACs / sample: `14,848`
- FLOPs / sample estimate: `30,040`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0818085 | 0.08698329999999999 | 0.09398380999999999 | 12129.975109291077 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.088122 | 0.09186145 | 0.09942622999999998 | 11269.827853379538 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.0836615 | 0.08909045 | 0.09378397999999999 | 11856.902365286025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.083093 | 0.08662375 | 0.09303287999999998 | 11948.65614659185 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 1 | 64 | ok | 0.0 | 0.0801505 | 0.0838365 | 0.08686037 | 12386.45951882073 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.1018535 | 0.10870674999999999 | 0.11317086999999999 | 19439.13809971928 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.1022035 | 0.10693875 | 0.11577316 | 19401.50245234991 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0993845 | 0.10629834999999999 | 0.11422075999999999 | 19942.426215515807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.103255 | 0.11103099999999999 | 0.11463785 | 19333.198000947326 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 2 | 64 | ok | 0.0 | 0.090141 | 0.09922375 | 0.10219264 | 21847.417635235513 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.100562 | 0.1053162 | 0.11109514999999998 | 39544.33849637398 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.1035335 | 0.10788465 | 0.11365811999999999 | 38514.28080646593 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.102335 | 0.10988085 | 0.11853691 | 38643.57944656211 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.1432715 | 0.15075355 | 0.16113691 | 30136.05827590405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 4 | 64 | ok | 0.0 | 0.088532 | 0.0926728 | 0.09704119 | 44844.32628462738 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.1032135 | 0.10986974999999999 | 0.11746364999999998 | 76807.79138235783 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.107045 | 0.1130051 | 0.11830331999999999 | 74148.59341045158 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.10404 | 0.1111669 | 0.11527309999999999 | 76200.65079165807 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.10144 | 0.10733085 | 0.11917784000000001 | 78059.6250640089 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 8 | 64 | ok | 0.0 | 0.1043995 | 0.1096441 | 0.11477603999999998 | 76038.81857727189 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.106077 | 0.1103999 | 0.12619952999999995 | 149457.56245924946 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.108243 | 0.11249849999999999 | 0.12106185999999998 | 146842.66249241924 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.1261285 | 0.1326446 | 0.13928635999999997 | 125884.2978535783 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.11143549999999999 | 0.11583589999999999 | 0.11997147999999999 | 142975.50615116372 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 16 | 64 | ok | 0.0 | 0.091592 | 0.0973249 | 0.10022838999999999 | 173092.50972455356 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.110001 | 0.11506445 | 0.12181608999999997 | 288820.86355272046 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.13838499999999998 | 0.1455707 | 0.1540624 | 229432.09390770318 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.14023950000000002 | 0.14482955 | 0.15262732 | 227010.18225796244 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.122837 | 0.1314907 | 0.13350355 | 258571.2741212657 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 32 | 64 | ok | 0.0 | 0.140496 | 0.1448871 | 0.15281422 | 226522.33984552027 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.129774 | 0.136449 | 0.14287018999999998 | 489207.3968769916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.160968 | 0.1685264 | 0.17505055 | 395302.08182135434 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1545725 | 0.1613425 | 0.17125564999999998 | 411715.3076137354 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.14429750000000002 | 0.15299749999999998 | 0.15909953000000002 | 439887.0919806659 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 64 | 64 | ok | 0.0 | 0.149319 | 0.1730894 | 0.17519327999999998 | 413476.2238250168 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.150975 | 0.16155635 | 0.16522682 | 838156.6630049697 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.2288645 | 0.239584 | 0.25234501 | 563750.1434700072 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.2119255 | 0.221188 | 0.22828529 | 610854.4249912762 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.1833665 | 0.1913759 | 0.19535707 | 698373.3248451876 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `fp32` | 128 | 64 | ok | 0.0 | 0.1903305 | 0.19761309999999999 | 0.19928617 | 670197.560627485 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.154706 | 0.16229025 | 0.16893376 | 6411.050394189844 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.1587975 | 0.16765124999999997 | 0.16934621 | 6253.116396884297 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.16118549999999998 | 0.17092439999999998 | 0.17930558999999996 | 6128.5903273929725 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.1791285 | 0.18510705 | 0.19501661999999997 | 5596.801495107165 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 1 | 64 | ok | 0.0 | 0.27553300000000003 | 0.2948871 | 0.29861508999999997 | 3655.041563305137 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.191441 | 0.1975038 | 0.2028823 | 10402.20868016464 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1964135 | 0.20308379999999998 | 0.20633338 | 10161.135283322934 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.198814 | 0.2087494 | 0.21343484 | 9989.273518096268 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.21799849999999998 | 0.2337995 | 0.23850607 | 9124.899340954145 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 2 | 64 | ok | 0.0 | 0.3707555 | 0.42969625 | 0.44449224 | 5292.629484180331 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.22956700000000002 | 0.23899265 | 0.24360605999999999 | 17325.36401239491 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.23397600000000002 | 0.24323939999999997 | 0.24666582 | 17095.885951977143 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.23919449999999998 | 0.25726155 | 0.26363269 | 16551.01369993608 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.2609545 | 0.27749475 | 0.27955842999999997 | 15358.088421736793 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 4 | 64 | ok | 0.0 | 0.4223625 | 0.50265445 | 0.50999604 | 8956.452786126312 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.22627999999999998 | 0.23547164999999998 | 0.24150944 | 35097.451144128645 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.2545305 | 0.26262775 | 0.26575934 | 31327.67795648304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.24640299999999998 | 0.2609482 | 0.26588040999999996 | 32543.399267073837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.2736305 | 0.2952454 | 0.30005377 | 29136.697145594324 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 8 | 64 | ok | 0.0 | 0.4792555 | 0.56296525 | 0.5846053099999999 | 17158.862194132023 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.2376995 | 0.2475346 | 0.25121365 | 67060.91780242714 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.2686635 | 0.27854555 | 0.28458326 | 59630.149957901114 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.2591555 | 0.27191355 | 0.27537607 | 61448.69916944402 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.28063499999999997 | 0.3051366 | 0.30915925 | 56823.59781734879 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 16 | 64 | ok | 0.0 | 0.4994865 | 0.5931382 | 0.60933283 | 32830.45556607133 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.256021 | 0.2628106 | 0.26786111 | 125052.59047613539 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.31350199999999995 | 0.32369745 | 0.32924014 | 103668.55814270522 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.2804735 | 0.2976985 | 0.30249351 | 114730.7877917837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.298929 | 0.317625 | 0.32426681999999996 | 106952.00034331593 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 32 | 64 | ok | 0.0 | 0.4437795 | 0.5267636 | 0.53407196 | 69854.15543815691 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.2896215 | 0.2994696 | 0.302318 | 220323.70509561908 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.33447899999999997 | 0.37976075 | 0.3819246 | 182463.0435360243 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.3169885 | 0.34690275 | 0.36157081999999996 | 199495.25207534287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.321721 | 0.35630475 | 0.36002895 | 196386.89839175087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 64 | 64 | ok | 0.0 | 0.4872665 | 0.6002953 | 0.6236501 | 130345.78088018841 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.3562645 | 0.3644947 | 0.36550021 | 358734.96375011216 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.44496 | 0.52272275 | 0.5371503 | 296809.02013742534 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.3841735 | 0.4419705 | 0.44556843 | 327676.2755003821 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.37552050000000003 | 0.39858309999999997 | 0.43490576 | 343860.57098585094 | - |
| `full_mlp_capacity_search_hd64_depth4` | `eager` | `bf16` | 128 | 64 | ok | 0.0 | 0.5173544999999999 | 0.6118384 | 0.6124497799999999 | 247868.7355340514 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 1 | ok | 62.23468 | 0.0445885 | 0.0502264 | 0.05186763999999999 | 21975.838883697706 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 2 | ok | 62.657953 | 0.045815499999999995 | 0.05135175 | 0.053130739999999996 | 21536.908230372555 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 4 | ok | 62.350391 | 0.0481365 | 0.054511399999999995 | 0.055984559999999996 | 20399.19594529262 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 8 | ok | 62.201527 | 0.0462335 | 0.052201449999999996 | 0.05406968 | 21283.143707617277 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 1 | 64 | ok | 61.024319 | 0.041651 | 0.04693605 | 0.04775503 | 23611.30131326058 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 1 | ok | 62.141817 | 0.0495625 | 0.05336394999999999 | 0.05488725 | 40304.29744571515 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 2 | ok | 62.726102 | 0.048484 | 0.054123199999999996 | 0.05518613 | 40385.48755581779 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 4 | ok | 62.251874 | 0.051497 | 0.05502699999999999 | 0.05943178999999999 | 38427.10186640434 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 8 | ok | 62.518174 | 0.0484675 | 0.05182894999999999 | 0.05284982 | 41225.6212288863 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 2 | 64 | ok | 60.722442 | 0.0454465 | 0.046978849999999996 | 0.04960161 | 43809.28765660177 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 1 | ok | 62.772478 | 0.047494499999999995 | 0.05292544999999999 | 0.05760195 | 82752.27444626315 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 2 | ok | 63.130062 | 0.0501855 | 0.0530198 | 0.05487492 | 78965.58245605862 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 4 | ok | 63.413865 | 0.0534365 | 0.0567015 | 0.0584322 | 74922.27750237598 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 8 | ok | 63.444516 | 0.048802 | 0.05236244999999999 | 0.055126619999999994 | 81384.08281969804 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 4 | 64 | ok | 61.375503 | 0.046088000000000004 | 0.0494313 | 0.053006349999999994 | 85971.58639069788 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 1 | ok | 62.910113 | 0.051172 | 0.05410155 | 0.05586763 | 154618.95125830837 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 2 | ok | 63.422102 | 0.050060499999999994 | 0.0535533 | 0.05645906 | 158519.49135850806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 4 | ok | 63.341673 | 0.054572499999999996 | 0.057954 | 0.06030558 | 147985.52849516846 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 8 | ok | 63.197524 | 0.0522615 | 0.05599165 | 0.057539379999999994 | 152155.81973187102 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 8 | 64 | ok | 60.807122 | 0.047217499999999996 | 0.049486949999999995 | 0.05134808 | 168613.946986932 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 1 | ok | 63.447108 | 0.0544805 | 0.058058549999999993 | 0.060165949999999996 | 291330.9381074705 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 2 | ok | 62.785535 | 0.055236 | 0.0577436 | 0.05992331 | 293220.7365704903 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 4 | ok | 63.11113 | 0.060669 | 0.06345069999999998 | 0.06798149999999999 | 262173.6225152085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 8 | ok | 62.884771 | 0.0507045 | 0.05440785 | 0.05652682999999999 | 312658.52764409455 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 16 | 64 | ok | 61.244256 | 0.049579 | 0.052183049999999995 | 0.05790550999999998 | 320017.40894704673 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 1 | ok | 63.712385 | 0.0557435 | 0.059053549999999996 | 0.060211110000000005 | 570771.3689665899 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 2 | ok | 63.899543 | 0.06600500000000001 | 0.07110525 | 0.07284087 | 479616.8820346307 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 4 | ok | 63.553524 | 0.0663195 | 0.07391865 | 0.07510199 | 470525.95979943825 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 8 | ok | 63.981478 | 0.0655275 | 0.06973209999999999 | 0.07335989 | 485934.03202548233 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 32 | 64 | ok | 68.795509 | 0.089058 | 0.0941137 | 0.09510369 | 357716.1611690164 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 1 | ok | 63.219552 | 0.0630985 | 0.06502145 | 0.06950692 | 1008787.1667140586 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 2 | ok | 64.164573 | 0.093196 | 0.113203 | 0.11547439999999999 | 647201.6718837188 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 4 | ok | 64.207703 | 0.077135 | 0.08543815 | 0.08584903 | 823110.0492991476 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 8 | ok | 64.084881 | 0.0861745 | 0.0907823 | 0.09392067999999999 | 756986.8705358473 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 64 | 64 | ok | 70.132272 | 0.107518 | 0.11403775 | 0.11494654 | 590152.9658045085 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 1 | ok | 63.831491 | 0.0840215 | 0.08885719999999998 | 0.09189401999999999 | 1512666.21601819 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 2 | ok | 65.495679 | 0.12988650000000002 | 0.16093795 | 0.16318289 | 911531.0674000313 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 4 | ok | 64.945238 | 0.119759 | 0.12587505 | 0.13062348 | 1063611.2660364327 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 8 | ok | 66.119667 | 0.1130205 | 0.12149375 | 0.12777207 | 1124536.0410286973 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `fp32` | 128 | 64 | ok | 75.45671 | 0.131696 | 0.13809375 | 0.14950328999999996 | 965432.3915238658 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 1 | ok | 61.99922 | 0.0895945 | 0.09290369999999999 | 0.09709859 | 11125.339183778366 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 2 | ok | 62.930865 | 0.1000965 | 0.10848909999999999 | 0.11170561 | 9878.190062698848 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 4 | ok | 62.790253 | 0.1010015 | 0.1111026 | 0.11864654 | 9765.49148742107 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 8 | ok | 62.225056 | 0.108704 | 0.11953565000000001 | 0.12018212 | 9180.266466414454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 1 | 64 | ok | 60.469954 | 0.2053385 | 0.22570255 | 0.23213198999999998 | 4893.607109119217 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 1 | ok | 63.195694 | 0.122477 | 0.12633405 | 0.13272895 | 16264.72994936952 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 2 | ok | 62.537933 | 0.119983 | 0.12486625 | 0.12750749 | 16592.3412739301 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 4 | ok | 62.360655 | 0.1259045 | 0.1352144 | 0.13825696 | 15748.316701798536 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 8 | ok | 65.844107 | 0.152038 | 0.1653125 | 0.16752452 | 13138.18572953915 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 2 | 64 | ok | 60.671113 | 0.2932325 | 0.3083771 | 0.31440348 | 6979.946195782744 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 1 | ok | 62.812167 | 0.1371925 | 0.144705 | 0.14855557 | 28971.370057538585 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 2 | ok | 62.986988 | 0.153662 | 0.15988059999999998 | 0.16364643999999998 | 26045.59046200059 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 4 | ok | 62.696221 | 0.1513875 | 0.16340369999999999 | 0.16512781999999998 | 26107.595143256287 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 8 | ok | 63.155305 | 0.181414 | 0.20023844999999998 | 0.20622483 | 21839.869205391304 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 4 | 64 | ok | 60.714833 | 0.383207 | 0.4415540499999999 | 0.46672262 | 10887.510082514806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 1 | ok | 63.082898 | 0.1451365 | 0.1525823 | 0.15650395 | 54805.571753646764 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 2 | ok | 63.031848 | 0.165864 | 0.1724447 | 0.17890943999999998 | 47963.873610396644 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 4 | ok | 63.433753 | 0.16801149999999998 | 0.17838915 | 0.18122708 | 47229.95153498524 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 8 | ok | 62.562624 | 0.182376 | 0.199948 | 0.21041290999999998 | 43386.21136308739 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 8 | 64 | ok | 61.945778 | 0.3878595 | 0.4106216 | 0.4409165699999999 | 21651.32021090118 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 1 | ok | 63.286211 | 0.154964 | 0.16365955 | 0.16575877 | 102419.10075261396 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 2 | ok | 62.987503 | 0.17216700000000001 | 0.18714804999999998 | 0.19121325 | 91078.548758958 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 4 | ok | 63.000514 | 0.1719 | 0.18157555 | 0.19647947999999996 | 91810.924581916 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 8 | ok | 63.148796 | 0.194816 | 0.2117203 | 0.21465212 | 81719.20036536655 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 16 | 64 | ok | 60.709246 | 0.3852725 | 0.46185315 | 0.47275247000000004 | 41538.3175922148 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 1 | ok | 63.053493 | 0.170412 | 0.1801273 | 0.18527238 | 186448.41636541655 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 2 | ok | 64.376574 | 0.2124935 | 0.22384295 | 0.22826633000000002 | 149007.24389403238 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 4 | ok | 63.607886 | 0.2015315 | 0.2118573 | 0.21451066 | 157832.81342218138 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 8 | ok | 63.791864 | 0.211089 | 0.2340749 | 0.23951898 | 150410.941493528 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 32 | 64 | ok | 68.834692 | 0.3252735 | 0.47081195000000003 | 0.47222043999999996 | 86942.32745914742 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 1 | ok | 63.802369 | 0.212157 | 0.21773095 | 0.22381256999999996 | 300771.0736427635 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 2 | ok | 64.503218 | 0.2970715 | 0.30683565 | 0.30791483 | 227494.87407838806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 4 | ok | 64.055401 | 0.2570435 | 0.26848045 | 0.27492634 | 254868.60766633623 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 8 | ok | 64.105613 | 0.2398535 | 0.2671869 | 0.27268924 | 265457.3976132891 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 64 | 64 | ok | 68.986652 | 0.367095 | 0.44562955000000004 | 0.44903992 | 168475.72743870458 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 1 | ok | 63.89825 | 0.27985899999999997 | 0.28953035 | 0.29051234000000004 | 455418.6097564186 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 2 | ok | 64.791283 | 0.36758599999999997 | 0.44594845 | 0.45160102 | 361316.75115268514 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 4 | ok | 64.643949 | 0.2979505 | 0.3712009 | 0.37305134 | 406136.36510745424 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 8 | ok | 65.077573 | 0.29884849999999996 | 0.34998085 | 0.36226691 | 423625.93618022325 | - |
| `full_mlp_capacity_search_hd64_depth4` | `jit` | `bf16` | 128 | 64 | ok | 74.24909 | 0.40400349999999996 | 0.48173299999999997 | 0.4907766 | 301763.1407356806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 1 | ok | 1367.459034 | 0.067373 | 0.07201894999999998 | 0.07568517999999999 | 14720.498942479355 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 2 | ok | 1352.328051 | 0.069506 | 0.07487854999999999 | 0.07888979 | 14263.8050811097 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 4 | ok | 1327.827511 | 0.0673405 | 0.07010519999999999 | 0.07198421999999999 | 14804.339921856774 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 8 | ok | 1416.609568 | 0.0691985 | 0.07381135 | 0.07490658 | 14368.976345503761 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 1 | 64 | ok | 1593.644581 | 0.066883 | 0.06935285 | 0.07544716 | 14839.542989562458 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 1 | ok | 1298.59582 | 0.070465 | 0.0735885 | 0.07936111 | 28134.43646583087 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 2 | ok | 1323.836726 | 0.069217 | 0.07192865 | 0.07410931999999999 | 28718.137226746927 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 4 | ok | 1410.589067 | 0.0735285 | 0.07579735 | 0.08582096999999998 | 27001.645480275565 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 8 | ok | 1445.09022 | 0.07374249999999999 | 0.07574465 | 0.08062551999999999 | 27023.492062119443 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 2 | 64 | ok | 1511.317272 | 0.071098 | 0.0738214 | 0.07917842999999998 | 27976.887733505595 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 1 | ok | 1377.758534 | 0.0727415 | 0.0751526 | 0.08082319999999998 | 54676.0308482166 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 2 | ok | 1376.102478 | 0.074244 | 0.07629135 | 0.07947523999999999 | 53720.32019459648 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 4 | ok | 1403.843488 | 0.07283 | 0.07559065 | 0.07961737999999999 | 54634.13568114425 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 8 | ok | 1349.043544 | 0.0735565 | 0.0753479 | 0.07824858 | 54181.38029775378 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 4 | 64 | ok | 1662.204093 | 0.071114 | 0.0731308 | 0.07439607 | 56102.63415893035 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 1 | ok | 1360.255879 | 0.0726055 | 0.07396715 | 0.07454496999999999 | 110184.45980415813 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 2 | ok | 1361.692404 | 0.0733555 | 0.074827 | 0.0791267 | 108927.92321017126 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 4 | ok | 1345.65698 | 0.075732 | 0.0773375 | 0.08291161999999999 | 105324.77288687567 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 8 | ok | 1375.138925 | 0.0727125 | 0.075222 | 0.08018261999999998 | 109292.60362304981 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 8 | 64 | ok | 1574.807191 | 0.077152 | 0.07910685 | 0.08264268999999999 | 103359.41349734405 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 1 | ok | 1387.961415 | 0.079257 | 0.08124995 | 0.08652518999999997 | 201352.53531786392 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 2 | ok | 1367.012047 | 0.0750475 | 0.0769165 | 0.08028991999999999 | 212417.5587540329 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 4 | ok | 1417.07431 | 0.091597 | 0.09825949999999999 | 0.10198916999999999 | 173618.5336048438 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 8 | ok | 1396.700306 | 0.0745115 | 0.07809189999999999 | 0.08187633 | 213288.2868471512 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 16 | 64 | ok | 1551.675563 | 0.0779865 | 0.0800476 | 0.08461913999999998 | 204141.62529396394 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 1 | ok | 1343.419015 | 0.080018 | 0.08278115 | 0.08957615999999999 | 397552.9621298503 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 2 | ok | 1316.810069 | 0.1075055 | 0.10940555 | 0.11328710999999998 | 296988.94050309184 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 4 | ok | 1371.135871 | 0.10571 | 0.1078224 | 0.11102021 | 302285.0672877113 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 8 | ok | 1404.486787 | 0.0955925 | 0.0990607 | 0.10395623999999999 | 332917.3253921454 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 32 | 64 | ok | 1582.135432 | 0.13200250000000002 | 0.13527440000000002 | 0.13647681 | 242093.38146956736 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 1 | ok | 1315.326969 | 0.091782 | 0.0944568 | 0.09842329999999999 | 694246.4757228517 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 2 | ok | 1346.81655 | 0.13095800000000002 | 0.13545000000000001 | 0.13780546999999999 | 486865.2874178795 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 4 | ok | 1403.378641 | 0.124535 | 0.1275781 | 0.13370111 | 512764.3059638975 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 8 | ok | 1412.938112 | 0.1153265 | 0.1191223 | 0.12937352 | 551151.7952994675 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 64 | 64 | ok | 1555.497727 | 0.1556995 | 0.1620742 | 0.16699489 | 408984.46855919115 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 1 | ok | 1353.410377 | 0.11374000000000001 | 0.11754479999999999 | 0.12232020999999998 | 1118263.3370375806 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 2 | ok | 1370.110743 | 0.184438 | 0.18982829999999998 | 0.19945340999999997 | 702790.6168228526 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 4 | ok | 1400.046223 | 0.17654199999999998 | 0.18318954999999998 | 0.18735237 | 722630.7281328836 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 8 | ok | 1480.974483 | 0.14256649999999998 | 0.14496905 | 0.15116678999999997 | 900491.4713608535 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `fp32` | 128 | 64 | ok | 1550.7726 | 0.171329 | 0.17972335 | 0.18353464 | 743785.5266541935 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 1 | ok | 1391.555022 | 0.123999 | 0.12869589999999997 | 0.13096767999999998 | 8039.4124154253805 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 2 | ok | 1363.072638 | 0.130611 | 0.13827895 | 0.14227254 | 7597.860442499392 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 4 | ok | 1410.157115 | 0.1323035 | 0.14073245 | 0.1446609 | 7474.885133440153 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 8 | ok | 1355.375876 | 0.144914 | 0.15606135 | 0.16139521999999998 | 6888.888429629661 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 1 | 64 | ok | 1631.876198 | 0.2402725 | 0.26849955 | 0.27833114 | 4084.6155894745666 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 1 | ok | 1297.345951 | 0.158601 | 0.16446444999999998 | 0.16840053 | 12566.789343664237 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 2 | ok | 1403.510976 | 0.1728805 | 0.18126935 | 0.18383757 | 11516.569867977498 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 4 | ok | 1335.099272 | 0.1710305 | 0.18517755 | 0.18997895 | 11526.18703589808 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 8 | ok | 1518.270409 | 0.189917 | 0.21272729999999998 | 0.22068042 | 10470.125381845473 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 2 | 64 | ok | 1546.804747 | 0.342951 | 0.39640735 | 0.40898108 | 5675.001730875529 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 1 | ok | 1371.543607 | 0.1942645 | 0.2011747 | 0.20551294999999997 | 20494.938416296674 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 2 | ok | 1410.173908 | 0.2107095 | 0.21543245 | 0.21869414 | 18966.933711609858 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 4 | ok | 1363.662145 | 0.204483 | 0.21880239999999998 | 0.22140334 | 19360.180870553766 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 8 | ok | 1421.495277 | 0.2353765 | 0.2544222 | 0.25780324 | 16940.026462015336 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 4 | 64 | ok | 1616.694118 | 0.4372975 | 0.5461622 | 0.5531096799999999 | 8918.951702092743 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 1 | ok | 1333.730755 | 0.19040649999999998 | 0.20105705 | 0.20379919999999999 | 41759.81279911118 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 2 | ok | 1368.782012 | 0.2196515 | 0.22712545 | 0.23031363 | 36380.87428334225 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 4 | ok | 1340.181808 | 0.3597325 | 0.3765536 | 0.38118672 | 24728.961310674516 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 8 | ok | 1383.233043 | 0.231362 | 0.25060095 | 0.25843859999999996 | 34261.85824327461 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 8 | 64 | ok | 1624.172161 | 0.5250215 | 0.79444995 | 3.4136901699999904 | 12179.365639567568 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 1 | ok | 1372.937293 | 0.2087445 | 0.21965105000000001 | 0.22227198999999997 | 76253.27984419929 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 2 | ok | 1373.680866 | 0.244203 | 0.25053505 | 0.26202016999999994 | 65603.8407112506 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 4 | ok | 1372.21689 | 0.230416 | 0.2462598 | 0.26001624999999995 | 68686.68482857091 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 8 | ok | 1394.054338 | 0.244981 | 0.2622493 | 0.27013320999999996 | 64610.51248573925 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 16 | 64 | ok | 1503.692554 | 0.41062750000000003 | 0.4956526 | 0.50694458 | 37569.98113423397 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 1 | ok | 1365.542584 | 0.2211595 | 0.22987915 | 0.2330669 | 144284.60715900545 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 2 | ok | 1382.357452 | 0.2782825 | 0.28637515 | 0.28794576 | 116403.60301162311 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 4 | ok | 1366.772031 | 0.2940695 | 0.3810032 | 0.38932622 | 109070.70126462025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 8 | ok | 1396.556986 | 0.2709025 | 0.28306945 | 0.2844146 | 118279.03413932119 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 32 | 64 | ok | 1553.346287 | 0.4566345 | 0.5447599 | 2.1689317699999937 | 62369.41160814032 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 1 | ok | 1378.04513 | 0.2674745 | 0.27665179999999995 | 0.28244456 | 238393.49602944462 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 2 | ok | 1404.605549 | 0.3191165 | 0.36379895 | 0.36659097 | 193076.38928568433 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 4 | ok | 1365.872333 | 0.311934 | 0.3234863 | 0.33047583 | 210050.3858363025 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 8 | ok | 1378.45704 | 0.2830385 | 0.31583 | 0.31789753000000004 | 221607.7378219796 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 64 | 64 | ok | 1623.950944 | 0.46135550000000003 | 0.5708712 | 0.7414704299999993 | 134277.19008195103 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 1 | ok | 1362.075959 | 0.3414645 | 0.35018065 | 0.36090847 | 373667.09589728125 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 2 | ok | 1337.598444 | 0.42647999999999997 | 0.51333815 | 0.5288938999999999 | 308239.37330311816 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 4 | ok | 1349.74518 | 0.34303 | 0.39210575 | 0.39816177999999997 | 360177.91888748243 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 8 | ok | 1412.83007 | 0.3347285 | 0.3884038 | 0.39678671 | 376493.35500993824 | - |
| `full_mlp_capacity_search_hd64_depth4` | `compile` | `bf16` | 128 | 64 | ok | 1557.717983 | 0.4643615 | 0.5482478 | 0.5553296799999999 | 279941.76161576784 | - |
