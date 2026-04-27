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

### full_mlp_capacity_search_hd512_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`496973.277` samples/s, p50=`0.239` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.065` ms, throughput=`15405.643` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`562968.066` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.040` ms, throughput=`24829.447` samples/s

### full_mlp_capacity_search_hd512_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`603264.964` samples/s, p50=`0.209` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.064` ms, throughput=`15412.149` samples/s

### full_mlp_capacity_search_hd512_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`440131.189` samples/s, p50=`0.291` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.029` ms, throughput=`31819.449` samples/s

### full_mlp_capacity_search_hd512_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`387335.274` samples/s, p50=`0.310` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.212` ms, throughput=`4653.332` samples/s

## Run References

### full_mlp_capacity_search_hd512_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `349,328`
- MACs / sample: `348,160`
- FLOPs / sample estimate: `697,560`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0883815 | 0.10215025 | 0.10389925 | 11067.835872848274 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0871885 | 0.10363444999999998 | 0.17546465999999988 | 10845.89993529336 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.06475700000000001 | 0.07121859999999999 | 0.07309791 | 15162.753968547597 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0647445 | 0.08243179999999996 | 0.09427007 | 15405.642902127332 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.10141900000000001 | 0.12085554999999999 | 0.16596382999999984 | 18903.07711850566 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0742915 | 0.08985755 | 0.09452888999999999 | 26346.774667365382 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.066785 | 0.08124984999999998 | 0.12999471999999984 | 29178.899920458323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.06251899999999999 | 0.07208724999999999 | 0.13097102999999985 | 31284.187751427107 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.1244835 | 0.13823285 | 0.14228121 | 31667.57156752421 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.084482 | 0.1300168 | 0.21029719999999982 | 42000.87109806658 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0751395 | 0.08576045 | 0.13291408999999982 | 50929.294097065635 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.07149900000000001 | 0.0808415 | 0.11355910999999991 | 53784.7668094204 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.142644 | 0.15952064999999999 | 0.16974381 | 55021.40470196418 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.1004465 | 0.15256404999999998 | 0.1856378799999999 | 72316.75025880357 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.09545000000000001 | 0.10747084999999999 | 0.1524441399999999 | 80988.21823895168 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.081512 | 0.0957335 | 0.10284793999999998 | 94600.80117418515 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.21344049999999998 | 0.25797644999999997 | 0.3718145999999997 | 71351.16178867017 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.16912850000000001 | 0.2150338999999999 | 0.3165931199999999 | 89491.22449052647 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.193639 | 0.20349925 | 0.20736632 | 91217.19772559039 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.1135195 | 0.1358642 | 0.15036927999999994 | 136221.75676004725 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.296165 | 0.33240775 | 0.3443126 | 107258.3681805716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.21653450000000002 | 0.24052384999999998 | 0.31691281999999993 | 143981.49693782852 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.173959 | 0.1910865 | 0.19499951999999998 | 181110.9025820302 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.142946 | 0.1605045 | 0.19413387999999998 | 215721.65973548478 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.422916 | 0.47799805 | 0.5052675 | 148961.07563303103 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.310929 | 0.3499467 | 0.35449667 | 202353.5616282 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.24006650000000002 | 0.25785635 | 0.26325807 | 264764.9044852333 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.206407 | 0.2217134 | 0.24281648999999994 | 306822.46600119426 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6856530000000001 | 0.70165865 | 0.71071673 | 186803.09129928116 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.48631 | 0.4989652 | 0.50617581 | 263398.74429587385 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.37059699999999995 | 0.3777103 | 0.38210574 | 345569.81654994324 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.307743 | 0.32150185000000003 | 0.33215945999999996 | 414978.089481077 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.1217065 | 0.13927425 | 0.14443039 | 8237.058592504143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.126739 | 0.14521409999999998 | 0.22349429999999998 | 7683.520109462502 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1033135 | 0.1237575 | 0.12880924 | 9510.163797355146 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.105338 | 0.11326825 | 0.11520064999999999 | 9425.173734227443 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.13502 | 0.16179209999999997 | 0.27116098999999977 | 14112.796241705908 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.1182025 | 0.1334479 | 0.13845754 | 16617.622523392627 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1183115 | 0.13203120000000002 | 0.13522787 | 16620.680015178004 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1138905 | 0.13333845 | 0.13604868 | 17266.85685533931 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.1247825 | 0.15499569999999996 | 0.2660830599999997 | 30000.11550044468 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.117722 | 0.13388695 | 0.13761236 | 33315.392994205955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.11380599999999999 | 0.133026 | 0.13887628999999999 | 34150.91185495971 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.113468 | 0.12271594999999999 | 0.12956536999999999 | 35251.98737485324 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.1342755 | 0.1512685 | 0.2554086599999998 | 57092.12671026601 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.131616 | 0.1435117 | 0.1541083 | 60235.59345311381 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1239275 | 0.1372271 | 0.14632204999999998 | 63552.609962697796 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.1213895 | 0.13503379999999998 | 0.14794787999999998 | 65001.321964385446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1406385 | 0.15918715 | 0.26329933999999977 | 109215.35999362182 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.145582 | 0.16763909999999999 | 0.2470551199999999 | 104981.352031212 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.1455245 | 0.16370115 | 0.16817106999999998 | 107907.92432633082 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1532235 | 0.1784866 | 0.18569060999999998 | 102939.66075977447 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.14997749999999999 | 0.1719248 | 0.17602604 | 207665.96306842557 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.16472599999999998 | 0.1764016 | 0.18843952999999997 | 192889.51385091376 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.17443399999999998 | 0.19132395 | 0.19729878 | 182776.01119685842 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.17715350000000002 | 0.1885272 | 0.19158323 | 179659.3703253474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.1848235 | 0.2012612 | 0.21772863999999995 | 339484.9059175932 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.23607299999999998 | 0.24642055 | 0.24894382999999998 | 272617.2886898669 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.2206965 | 0.23515329999999998 | 0.241321 | 289574.7829591756 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.217699 | 0.23490815 | 0.23844688 | 294442.9415585969 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.238704 | 0.36135375 | 0.39643876999999994 | 496973.277436264 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.29243850000000005 | 0.3115811 | 0.31965349 | 434384.19694002124 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.29301299999999997 | 0.30699204999999996 | 0.31968478 | 436997.46145443403 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.283995 | 0.29644005 | 0.30190803 | 451516.06395386404 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 51.829305 | 0.064526 | 0.06962925 | 0.07297740999999999 | 15434.990444197414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 50.47811 | 0.0404415 | 0.046425749999999995 | 0.050662399999999996 | 23983.932683816456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 49.326391 | 0.0402505 | 0.044391 | 0.0458936 | 24426.84842847428 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 50.006918 | 0.0399045 | 0.04580375 | 0.047856749999999997 | 24829.44653177325 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 50.20066 | 0.057626 | 0.06280954999999999 | 0.06509920999999999 | 34422.61400511589 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 51.655447 | 0.05156 | 0.0568413 | 0.05977567 | 38345.21971235717 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 50.344846 | 0.0420305 | 0.04572175 | 0.0462852 | 47796.257935373724 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.90919 | 0.0354935 | 0.0419832 | 0.04466899 | 55257.07525405822 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 50.75974 | 0.089131 | 0.10434689999999998 | 0.10921299000000001 | 44031.284227443604 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 51.979729 | 0.057082 | 0.06353975 | 0.0656026 | 69034.40882038834 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 50.200198 | 0.047419 | 0.05287554999999999 | 0.05399175 | 84372.13002926449 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 49.071565 | 0.047213000000000005 | 0.051744599999999995 | 0.053073209999999996 | 85210.84785219668 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 50.342975 | 0.097773 | 0.13777055 | 0.14048986 | 74159.46732737808 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 52.028808 | 0.06910250000000001 | 0.0770948 | 0.08067129999999999 | 114062.21751778941 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.530153 | 0.056234000000000006 | 0.06258115 | 0.06455835 | 140276.99797404945 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 51.752807 | 0.054983000000000004 | 0.0587374 | 0.06401256 | 145698.15276597015 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 52.547288 | 0.1195845 | 0.1253653 | 0.12958891 | 132695.0040165119 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 52.356585 | 0.1023725 | 0.11032375 | 0.11519474999999998 | 154210.5058220249 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 51.49767 | 0.0836615 | 0.09008094999999999 | 0.09309914999999999 | 189748.14491540196 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 50.688881 | 0.0761505 | 0.0823859 | 0.08452217999999999 | 208800.63350112207 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 52.949649 | 0.17723899999999998 | 0.19223284999999998 | 0.19557802 | 178306.3968088504 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 51.375327 | 0.15315600000000001 | 0.160334 | 0.16767089 | 208257.51458193085 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 51.087192 | 0.1392255 | 0.1464744 | 0.14998884 | 228803.86060754006 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 52.204242 | 0.130709 | 0.1379312 | 0.14292143 | 243412.38606969168 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 54.10775 | 0.29764 | 0.3103075 | 0.33619998999999995 | 212626.22356424807 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 53.116301 | 0.2504645 | 0.25996425 | 0.26064478 | 254507.58786536456 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 53.161931 | 0.2199275 | 0.2253254 | 0.22825301999999997 | 290462.99256396585 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 52.349966 | 0.20336500000000002 | 0.2142184 | 0.22416542999999997 | 314596.459747727 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 54.575657 | 0.5517160000000001 | 0.5636365 | 0.5925454099999999 | 231629.36637642063 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 53.560249 | 0.459888 | 0.46807615 | 0.47267805 | 278049.0687094005 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 53.431395 | 0.32514350000000003 | 0.33412325 | 0.34995497 | 392871.78346969874 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 53.961483 | 0.24688749999999998 | 0.2567797 | 0.2603642 | 517291.8124749286 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 52.243296 | 0.08490400000000001 | 0.09626654999999999 | 0.11603587999999992 | 11493.4813571135 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 51.352253 | 0.0824705 | 0.09247019999999999 | 0.09795005 | 11963.253670027143 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 51.021778 | 0.072362 | 0.08533774999999999 | 0.08927248 | 13512.48364313855 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 51.507685 | 0.07963 | 0.0887587 | 0.0934438 | 12556.636709879913 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 50.511604 | 0.090325 | 0.10440485 | 0.12206891999999994 | 21433.488883306654 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 51.42757 | 0.08563799999999999 | 0.0955386 | 0.10183139 | 23036.448037720802 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 52.342803 | 0.09122649999999999 | 0.1041071 | 0.1058476 | 22152.45675175868 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.639194 | 0.08487249999999999 | 0.0952946 | 0.09735687999999999 | 23539.231778339526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 50.773406 | 0.0889035 | 0.10289374999999999 | 0.10508638 | 44093.95628392892 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 52.368922 | 0.0885335 | 0.0980841 | 0.10299881 | 44444.2074086716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 52.049024 | 0.0854545 | 0.09703774999999999 | 0.10196245 | 46111.391750164796 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 50.568592 | 0.0832705 | 0.09201789999999999 | 0.09642102 | 48145.50727791561 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 52.714378 | 0.093355 | 0.10223265 | 0.11297091 | 84279.20943573174 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 52.93171 | 0.0944955 | 0.1090465 | 0.11245648 | 83321.75508111685 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 50.926812 | 0.0975945 | 0.10893755 | 0.11269545999999998 | 82097.61036433073 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 52.191414 | 0.0865885 | 0.09404544999999999 | 0.10353235 | 91421.2577279532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 53.023405 | 0.10224549999999999 | 0.1143473 | 0.12084997 | 154039.14712375216 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 51.859895 | 0.10757549999999999 | 0.12029135 | 0.12139939999999999 | 146804.0752811298 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.183587 | 0.11340549999999999 | 0.12350355 | 0.12664461999999999 | 139911.33468942394 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.664569 | 0.1119955 | 0.1195249 | 0.1215455 | 142613.5792371742 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 53.931975 | 0.1207415 | 0.1365662 | 0.14752948 | 258642.58230047382 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 52.483206 | 0.1332575 | 0.1442898 | 0.14611889 | 238567.04703200224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 53.501828 | 0.137464 | 0.14902664999999998 | 0.15346005 | 233471.73343131365 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 52.904555 | 0.1348705 | 0.1409405 | 0.14394468 | 238206.79118628916 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 53.803956 | 0.1513595 | 0.15927825 | 0.16392663999999998 | 419063.2940103545 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 55.272706 | 0.19743149999999998 | 0.20384195 | 0.20727757 | 324661.8519990394 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 53.735274 | 0.196389 | 0.20697975 | 0.21503227999999996 | 324073.9990569446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 63.533795 | 0.398655 | 0.4898415 | 0.49569570999999996 | 154950.90162782217 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 55.185651 | 0.2259825 | 0.23383945 | 0.23957304 | 562968.0661642294 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 54.109975 | 0.2528005 | 0.2606508 | 0.2909438599999999 | 502681.8074427068 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.635477 | 0.257392 | 0.2652314 | 0.26704924 | 500014.297283813 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 55.239352 | 0.2463185 | 0.2563072 | 0.25772924 | 522543.634638797 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 497.982784 | 0.079372 | 0.084271 | 0.0884128 | 12535.04799419176 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 499.354782 | 0.0944695 | 0.10093980000000001 | 0.10198717 | 10504.437809841524 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 506.35724 | 0.068048 | 0.071661 | 0.07471697999999999 | 14649.700750562772 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 515.209387 | 0.0639685 | 0.07174605 | 0.07443512 | 15412.148595305953 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.381303 | 0.083699 | 0.08860275 | 0.09025991 | 23696.424162201074 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 508.920122 | 0.075549 | 0.08251929999999999 | 0.08661912999999999 | 26139.314705586363 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 507.543608 | 0.069742 | 0.07455595 | 0.07849458999999999 | 28451.65523194643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 512.527665 | 0.059118000000000004 | 0.06162695 | 0.06595211 | 33675.067922612005 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.917702 | 0.110214 | 0.11931805 | 0.12066118999999999 | 35987.553344800166 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 525.755262 | 0.12552950000000002 | 0.133409 | 0.16665685999999996 | 32108.439189505938 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 508.604933 | 0.092556 | 0.09888995 | 0.10111360999999999 | 42939.74973425662 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 502.239334 | 0.08571300000000001 | 0.0898484 | 0.09249400999999999 | 46607.67222874859 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.940939 | 0.1330015 | 0.14307799999999998 | 0.14693995999999998 | 59458.219675230284 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 511.815086 | 0.17007 | 0.18021835 | 0.185461 | 55176.774660949064 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 503.27966 | 0.12385650000000001 | 0.1304239 | 0.13171268 | 64226.23047422241 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 504.015069 | 0.10576650000000001 | 0.11564685 | 0.11900161 | 75221.83388950476 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.015587 | 0.2062865 | 0.21714624999999999 | 0.22108208 | 77497.18951598645 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 501.437687 | 0.1988915 | 0.4057865 | 0.41102333 | 74547.17251893033 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.119487 | 0.197477 | 0.2596793 | 0.26283708 | 82301.81722412432 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 499.448221 | 0.162041 | 0.16955499999999998 | 0.17075223 | 99063.23330010589 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 491.951741 | 0.2775705 | 0.29344685 | 0.29488954 | 115031.15726736297 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.645585 | 0.2181765 | 0.4507781499999998 | 0.48627407 | 131136.6845859482 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.008236 | 0.2214345 | 0.34509154999999997 | 0.35173867 | 150241.56966382323 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 511.086934 | 0.1798815 | 0.2479753 | 0.24950512 | 163765.47596542048 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.098467 | 0.390611 | 0.40090345 | 0.40501506 | 163748.64031060663 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 494.789152 | 0.297142 | 0.36265035 | 0.36931874 | 209679.19803472926 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 497.246543 | 0.225371 | 0.3922975 | 0.39430649 | 255566.00802780752 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 508.011916 | 0.218864 | 0.34202295000000005 | 0.34566562 | 275851.60562875203 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.214717 | 0.676914 | 0.6928277500000001 | 0.69671647 | 188552.49487063562 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.728833 | 0.461537 | 0.4746832 | 0.47935925 | 276978.375389398 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 519.782167 | 0.330742 | 0.3577358 | 0.4018244299999999 | 382600.59242114855 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 499.897893 | 0.26522049999999997 | 0.38486754999999995 | 0.4715518299999999 | 452628.24602523446 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.578457 | 0.1116525 | 0.1257887 | 0.13737918999999998 | 8743.499645188784 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.617952 | 0.14659899999999998 | 0.1544133 | 0.15896232 | 6787.880104316142 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 509.771086 | 0.1285265 | 0.1427882 | 0.14959638 | 7673.180224741311 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 523.745479 | 0.125666 | 0.1343507 | 0.13629464 | 7918.594316476441 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.328215 | 0.1285355 | 0.13630865 | 0.14064562999999997 | 15468.10507669396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 504.946984 | 0.13362649999999998 | 0.15095165 | 0.15615426 | 15111.960739126 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 510.701553 | 0.126674 | 0.13681274999999998 | 0.14356876999999998 | 15979.658533872718 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 511.076532 | 0.12882949999999999 | 0.13888545 | 0.13929646999999998 | 15495.417152899947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.357788 | 0.127255 | 0.13284284999999998 | 0.13963144 | 31265.99939812951 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 503.147935 | 0.144482 | 0.1525367 | 0.15643987 | 28891.801503500457 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 501.664921 | 0.1405205 | 0.15305749999999999 | 0.15579241 | 28608.488825238182 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 510.965095 | 0.120676 | 0.1323425 | 0.13381543999999998 | 32760.00056347201 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 494.308734 | 0.12704300000000002 | 0.13572765 | 0.13945023 | 62475.99555108435 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 507.224628 | 0.15498 | 0.16560350000000001 | 0.16845795 | 52306.795044977305 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 509.265741 | 0.142496 | 0.15220495 | 0.15938820999999997 | 56499.78466519569 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 508.179628 | 0.1353975 | 0.14140005 | 0.14871938999999998 | 58975.99102149513 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.223738 | 0.120774 | 0.13279145 | 0.14094679 | 130295.70610500532 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 505.274547 | 0.173465 | 0.18389195 | 0.19215051999999996 | 95959.38998615785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 507.580695 | 0.15086850000000002 | 0.17083255 | 0.17642086999999998 | 104090.95675983588 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 508.136513 | 0.1396305 | 0.14944965 | 0.15231364 | 113683.92033315072 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 486.822969 | 0.132722 | 0.1439287 | 0.14964365 | 238931.5696997735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 507.855159 | 0.17465150000000002 | 0.2247328 | 0.23050054 | 169168.85017835684 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 501.747485 | 0.1694795 | 0.18388805 | 0.18837210999999998 | 195883.53205921705 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 504.584582 | 0.16963899999999998 | 0.1867805 | 0.18890145 | 195459.38085309224 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.198997 | 0.168699 | 0.194654 | 0.2033912 | 369900.67357756716 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 505.195342 | 0.203551 | 0.26251375 | 0.26603931000000003 | 288896.87167119706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 515.3689 | 0.178159 | 0.24153729999999998 | 0.24371766 | 327280.2509994065 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 510.480646 | 0.18313400000000002 | 0.2194455 | 0.22164557 | 338714.73423912714 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.112925 | 0.2094605 | 0.2277023 | 0.23416202 | 603264.9642466535 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.726789 | 0.27010999999999996 | 0.2816498 | 0.29113762 | 474340.7589896838 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 512.315374 | 0.214273 | 0.24907444999999998 | 0.2601408 | 582187.3944785347 | - |
| `full_mlp_capacity_search_hd512_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 510.443131 | 0.22350799999999998 | 0.26842789999999994 | 0.28799052999999997 | 563918.6353521474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.358071 | 0.0441985 | 0.20178194999999996 | 0.24449187999999994 | 15630.392485407467 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.147969 | 0.0382375 | 0.040867099999999996 | 0.04410795 | 26174.063806085785 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.422433 | 0.036696 | 0.04167675 | 0.04489687 | 26953.930880417804 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.215791 | 0.028842 | 0.038008 | 0.04152524999999999 | 31819.448810779915 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.478732 | 0.039043999999999995 | 0.0432013 | 0.04631777999999999 | 50862.086942634174 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.789061 | 0.039985 | 0.0501395 | 0.05500378999999998 | 48697.796570798564 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.472459 | 0.040400500000000006 | 0.04782629999999999 | 0.05326213 | 53359.714376120886 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.724826 | 0.0377715 | 0.04188015 | 0.04786665999999999 | 52707.81108677181 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.739829 | 0.0435525 | 0.0456098 | 0.04988580999999999 | 92096.04143585097 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.968889 | 0.0604445 | 0.06549265 | 0.07093522999999999 | 65422.41947808611 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.338064 | 0.047492 | 0.05161979999999999 | 0.054759039999999995 | 83797.88787423613 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.285978 | 0.042325 | 0.0484213 | 0.05183595999999999 | 94448.81794943118 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.437868 | 0.06454750000000001 | 0.0691309 | 0.08053215999999998 | 122991.54802082002 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.836543 | 0.070577 | 0.12108529999999999 | 0.12596171 | 96890.09441697474 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.188702 | 0.078623 | 0.0907316 | 0.09480522999999999 | 108104.48514698427 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.72878 | 0.0737235 | 0.0803364 | 0.08290157999999999 | 115948.95231425414 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.633912 | 0.0893105 | 0.09424144999999999 | 0.09654413999999999 | 178802.31504297402 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.120098 | 0.104624 | 0.10907575 | 0.12518415999999996 | 151576.7582809225 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.208951 | 0.105115 | 0.11601305 | 0.12168462 | 161439.06785082223 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.723731 | 0.08755450000000001 | 0.0948227 | 0.10001369 | 185635.48013804783 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.744786 | 0.149457 | 0.15857655 | 0.15957789 | 212548.57565221528 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.831511 | 0.158302 | 0.21097075 | 0.21239217999999999 | 189775.66973909523 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.622118 | 0.128158 | 0.1342225 | 0.13753385 | 251911.37757736826 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.20091 | 0.1089335 | 0.11892074999999999 | 0.12189525 | 292841.31112376024 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.065215 | 0.2644665 | 0.2730181 | 0.27715056 | 241015.21634504944 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.820888 | 0.253355 | 0.280507 | 0.2829272 | 248901.95347586914 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.091778 | 0.196515 | 0.22675085 | 0.22989956 | 317966.72994867025 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.470939 | 0.1705565 | 0.1800392 | 0.184292 | 376552.63240884955 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.84592 | 0.5031995 | 0.5175681 | 0.6532665599999995 | 251024.3657192811 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.634903 | 0.4505935 | 0.4577671 | 0.45938006 | 284197.2686066949 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.990544 | 0.3455435 | 0.36510845 | 0.37403288 | 368972.8430222335 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.904401 | 0.2914365 | 0.30989455 | 0.31674877999999995 | 440131.1893534466 | - |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 224.61693 | 0.2207835 | 0.28845109999999996 | 0.31576313 | 4492.635851018445 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 221.609741 | 0.228543 | 0.28757315 | 0.3347418299999999 | 4314.505722026924 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 222.625263 | 0.2117035 | 0.28408999999999995 | 0.30155824999999997 | 4653.332325731773 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 222.942254 | 0.23587550000000002 | 0.2987477499999999 | 0.33315874999999995 | 4253.5840486514735 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 219.745 | 0.263075 | 0.6406020499999999 | 1.6418382299999974 | 6091.932253083478 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 220.727249 | 0.27723949999999997 | 0.35992715 | 0.4087500199999999 | 7006.899764266871 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 219.735203 | 0.302863 | 0.77042135 | 0.8514445799999999 | 5058.783315060165 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 219.510532 | 0.2340545 | 0.9661689999999996 | 32.54803011999989 | 1283.788659578416 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 222.003391 | 0.236517 | 0.29205619999999993 | 0.4708325199999994 | 16728.103414473557 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 221.796911 | 0.230552 | 0.3195537 | 0.3990063499999998 | 16685.011837181646 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 223.816464 | 0.28657699999999997 | 0.33226825 | 0.3549287 | 14096.863638148172 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 225.243793 | 0.5548150000000001 | 2.5231018499999984 | 3.09177965 | 4920.0721971394205 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 222.095281 | 0.310174 | 0.48521604999999995 | 0.7004710899999993 | 24106.258943045406 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 220.756897 | 0.29574449999999997 | 0.40962719999999997 | 0.44483391 | 26109.779874973316 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 221.164188 | 0.29971749999999997 | 0.35905204999999996 | 0.3949245399999999 | 26760.46745719764 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 219.672265 | 0.305021 | 0.5504259999999996 | 0.8548194099999995 | 23458.586153358396 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 219.519415 | 0.3039595 | 0.3797102 | 0.4476935799999998 | 52295.05339782533 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 221.156578 | 0.286018 | 0.37364605 | 0.4654969399999998 | 54928.72345852664 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 218.971263 | 0.4430845 | 4.504278649999996 | 5.317727489999999 | 15358.083704320643 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 220.27869 | 0.2582135 | 0.34913914999999995 | 0.39929489999999995 | 59727.781675845086 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 220.089359 | 0.6796525 | 1.2677925999999997 | 1.6852892399999997 | 46080.90937909805 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 223.181833 | 0.25439199999999995 | 0.45408664999999976 | 0.52269614 | 113626.37290840481 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 222.220925 | 0.29693650000000005 | 0.41955404999999985 | 0.5056821799999998 | 107935.20757424526 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 220.540458 | 0.3356875 | 0.41879569999999994 | 0.5299176199999998 | 93273.79899927706 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 222.486141 | 0.2534655 | 0.40582894999999997 | 0.44460291999999996 | 232907.90763105292 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 220.277587 | 0.3613095 | 0.44065855 | 0.45189656 | 174003.13738531902 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 224.70945 | 0.311668 | 0.5911162499999999 | 0.7412880399999999 | 185497.10586835747 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 223.059043 | 0.338234 | 0.3902083 | 0.40456838999999994 | 187513.75222928947 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.335667 | 0.42877200000000004 | 0.51247365 | 0.5500740499999999 | 294680.70930751786 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 219.512194 | 0.41743600000000003 | 0.5019889 | 0.5950087199999997 | 304355.9229469719 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.069851 | 0.43904 | 1.10785905 | 1.1843082399999998 | 245588.08676749901 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 221.93812 | 0.3098695 | 0.47637909999999994 | 0.50811821 | 387335.27432687936 | - |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd512_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
