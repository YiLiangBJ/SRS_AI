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

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`69426.893` samples/s, p50=`1.836` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.689` ms, throughput=`1447.115` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`103168.777` samples/s, p50=`1.222` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.250` ms, throughput=`3941.825` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`81741.202` samples/s, p50=`1.542` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.586` ms, throughput=`1681.763` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`121765.928` samples/s, p50=`1.037` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.138` ms, throughput=`7100.420` samples/s

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`39899.434` samples/s, p50=`3.205` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.849` ms, throughput=`1176.827` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `510,528`
- MACs / sample: `503,808`
- FLOPs / sample estimate: `1,014,552`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.7429950000000001 | 0.7686482 | 0.77104635 | 1341.353554506438 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.731762 | 0.7449944000000001 | 0.75021422 | 1365.2112713590034 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.688595 | 0.71685265 | 0.72429852 | 1447.1153853111132 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.7004095 | 0.7102525 | 0.71266586 | 1427.7355913637987 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.822723 | 0.8346876 | 0.84111728 | 2428.932944011007 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.7985495 | 0.81207445 | 0.81796315 | 2502.342881080984 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.7969999999999999 | 0.8055056 | 0.80820488 | 2511.744288356282 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.809369 | 0.82198215 | 0.82510072 | 2466.376809842718 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.837531 | 0.8513916 | 0.8548096399999999 | 4767.642847101861 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.866837 | 0.8802069499999999 | 0.9037624799999999 | 4612.065347431495 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.8354999999999999 | 0.8478277 | 0.85053231 | 4783.431913885788 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.8600289999999999 | 0.8741186 | 1.0425040299999995 | 4608.351738898164 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.9304135 | 0.9507134 | 0.9558329 | 8574.374910384384 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.926306 | 0.9478493499999999 | 0.95003078 | 8604.864678499775 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.8304765000000001 | 0.8843899 | 0.93545285 | 9509.823743024783 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.923827 | 0.9470384 | 0.94931142 | 8632.693005487825 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 1.001633 | 1.02907165 | 1.04540016 | 15907.653842373482 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 1.027868 | 1.0718314999999998 | 1.08082981 | 15509.739612410056 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.9158745 | 0.9857171 | 0.99439755 | 17244.60040844267 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.941685 | 1.01549265 | 1.03831169 | 16748.334686912807 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 1.151533 | 1.2083442 | 1.24148357 | 27517.866748926746 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 1.1819965 | 1.27352705 | 1.28232349 | 26667.126230142036 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 1.106317 | 1.186877 | 1.2028065099999998 | 28543.59397993482 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 1.0310199999999998 | 1.1199386 | 1.12815976 | 30574.47606049099 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 1.356754 | 1.4265025 | 1.43697812 | 46802.18033222324 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.5414385 | 1.6117272 | 1.61971022 | 41221.25225037443 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.4309695 | 1.48743605 | 1.50985358 | 44518.20988160132 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.2744775000000002 | 1.44538015 | 1.47246043 | 49387.31562422519 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.836349 | 1.8923191499999998 | 1.90247878 | 69426.89336196239 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 2.2250835 | 2.2839711 | 2.30108467 | 57437.81158778772 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 2.1546 | 2.1926811 | 2.22729291 | 59331.600961831995 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 2.0027865 | 2.0470036499999997 | 2.06395161 | 63782.4120417367 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 1.018593 | 1.0646356 | 1.07486509 | 975.3436062374003 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 1.0463885 | 1.0621764500000002 | 1.06794594 | 956.0799254517711 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 1.0799495000000001 | 1.1007729499999999 | 1.1093176599999999 | 925.4629887730974 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 1.083358 | 1.1032795 | 1.10769033 | 921.64571787455 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 1.4384495 | 1.5120999499999999 | 1.52118786 | 1382.7952558949737 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 1.5383415 | 1.5736554 | 1.57585557 | 1297.8810210404035 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 1.5186675 | 1.5531514499999999 | 1.56471055 | 1314.5259029563872 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 1.635802 | 1.67654415 | 1.68915887 | 1221.366005297382 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.776596 | 1.84995505 | 1.8960730799999999 | 2238.4749431382593 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.9395 | 2.02347815 | 2.03635915 | 2049.5906029878133 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.9234955 | 1.9723988 | 1.991821 | 2077.5558668796107 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.9104705 | 1.9513715 | 1.97188097 | 2091.4474960406287 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.866886 | 1.9476654 | 1.95209597 | 4264.2025716254375 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.9276305 | 2.0471432 | 2.10389261 | 4094.208473942953 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.9150895000000001 | 2.001382 | 2.01954108 | 4161.004060266152 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.942907 | 2.0229639500000003 | 2.04906362 | 4106.122067864425 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.7867264999999999 | 1.86762645 | 1.87440619 | 8924.628854248786 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.9950865000000002 | 2.09498195 | 2.1403465500000003 | 7956.290764117888 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 2.0568345 | 2.14262875 | 2.1566816199999996 | 7759.246412677304 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 2.0835065 | 2.14407605 | 2.20332225 | 7678.172103636319 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.8481269999999999 | 1.9300933 | 1.9626680499999998 | 17264.867902665035 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 2.1718615 | 2.290317 | 2.329087 | 14644.325442418572 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 2.218146 | 2.32594595 | 2.3835062 | 14375.891170478588 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 2.2282219999999997 | 2.34396035 | 2.37225692 | 14337.747764570382 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.9861045000000002 | 2.1852643 | 2.1926855499999998 | 31526.61060217255 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.5559365 | 2.62154505 | 2.6441668799999998 | 25054.97119402558 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.5553495 | 2.6585689 | 2.66961295 | 25011.935773913614 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.6678485 | 2.9988004 | 3.01700224 | 23405.978615449472 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.3878695 | 2.52225695 | 2.53564846 | 53460.06085308608 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 3.021896 | 3.10228645 | 3.1381485899999997 | 42292.03563852043 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 3.3610165 | 3.4134202 | 3.46401817 | 38076.20429932582 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 3.3519305 | 3.96429315 | 4.017828 | 36051.32273063556 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 320.624338 | 0.269787 | 0.28985445 | 0.2951064 | 3659.997906481198 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 320.016419 | 0.273917 | 0.2950305 | 0.33128118999999995 | 3598.8298045007687 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 316.787767 | 0.24975350000000002 | 0.26813015 | 0.3177823799999999 | 3941.824970357476 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 318.974756 | 0.2528955 | 0.2610438 | 0.26473605 | 3940.5584102274197 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 321.104888 | 0.305286 | 0.32019075 | 0.32874925 | 6525.685751687216 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 319.797039 | 0.322098 | 0.33305225 | 0.3431323 | 6220.56003328746 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 314.219448 | 0.294434 | 0.30202295 | 0.31885540999999995 | 6757.947160016633 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 314.566932 | 0.30448149999999996 | 0.31276345 | 0.33011041999999996 | 6531.36551128117 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 321.445929 | 0.3420515 | 0.34986075 | 0.3716305899999999 | 11648.682944575043 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 320.929278 | 0.358287 | 0.3688673 | 0.37244994 | 11142.179220838549 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 322.793028 | 0.350976 | 0.36080799999999996 | 0.36647579999999996 | 11334.269657349727 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 317.115667 | 0.3581755 | 0.36836515 | 0.3705864 | 11121.389128230452 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 320.026336 | 0.367079 | 0.37699794999999997 | 0.39175806999999996 | 21708.89254309853 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 322.379246 | 0.3655575 | 0.37567605 | 0.39451834999999996 | 21787.580567068628 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 322.086983 | 0.36433150000000003 | 0.3885035 | 0.40161897999999996 | 21792.022278420216 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 321.21214 | 0.366467 | 0.37688325 | 0.37770018 | 21756.528563112915 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 324.411336 | 0.4373855 | 0.44866 | 0.45221437 | 36520.67693448542 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 320.165101 | 0.5134345 | 0.5401060999999999 | 0.55248787 | 31001.048261695658 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 319.534582 | 0.439183 | 0.4515771 | 0.45237265 | 36344.87910193984 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 323.992721 | 0.4377405 | 0.45422045 | 0.46165492999999996 | 36400.81814478862 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 326.607976 | 0.544857 | 0.5542791 | 0.5575394899999999 | 58680.44190040279 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 320.467035 | 0.6835355000000001 | 0.70355965 | 0.7333422099999999 | 46699.888188792705 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 324.155099 | 0.5813385 | 0.6304387499999999 | 0.63931252 | 54397.9871656789 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 319.597313 | 0.5808305 | 0.62213465 | 0.62854088 | 54567.960728802864 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 325.575706 | 0.7756685 | 0.7844743 | 0.7875846599999999 | 82520.52762593857 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 330.360693 | 1.1078875 | 1.1922557999999999 | 1.20806164 | 57110.974941096276 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 322.027593 | 1.004221 | 1.0904867 | 1.09694255 | 62772.34246667647 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 325.673201 | 0.855035 | 0.89029525 | 0.89238268 | 74566.3674458153 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 328.305485 | 1.2224345 | 1.3290072 | 1.34808826 | 103168.77741387858 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 331.414729 | 1.8024565 | 1.8644333499999999 | 2.0031972999999996 | 70574.09867943951 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 333.494492 | 1.646595 | 1.68011395 | 1.69292903 | 77852.53134586202 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 331.316763 | 1.5046015000000001 | 1.5392629500000001 | 1.54480375 | 84975.50488129839 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 317.603278 | 0.4820465 | 0.49183659999999996 | 0.5105468 | 2070.8999949138697 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 320.152626 | 0.5211945 | 0.54358455 | 0.54894579 | 1911.3577272274626 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 323.766028 | 0.5222905 | 0.5303317 | 0.53160716 | 1915.1308243102264 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 324.686132 | 0.5464105 | 0.5755623999999999 | 0.5818543899999999 | 1821.3608545329723 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 325.508838 | 0.725186 | 0.85807715 | 0.8725902299999999 | 2672.166769714325 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 329.118104 | 0.8666240000000001 | 0.9190082 | 0.9206754500000001 | 2289.824350321876 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 325.970002 | 0.833393 | 0.868871 | 0.88856815 | 2389.20674895477 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 330.885266 | 0.9171525 | 0.9527448999999999 | 0.9713372100000001 | 2174.616721084638 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 328.385777 | 0.93771 | 1.0048464 | 1.00892801 | 4221.418820554156 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 334.153597 | 1.139704 | 1.27855225 | 1.32545742 | 3447.2031875322655 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 332.125765 | 1.0862175 | 1.1707298 | 1.2549327 | 3649.9253654136473 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 336.478168 | 1.209052 | 1.29309295 | 1.35265247 | 3288.993209906408 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 326.644001 | 1.0096595000000002 | 1.0746580000000001 | 1.07838217 | 7820.924143468519 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 331.631305 | 1.2128355 | 1.2958831 | 1.31122497 | 6580.726112712769 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 328.880898 | 1.1581515 | 1.353143 | 1.36147378 | 6734.3229297416 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 338.997542 | 1.190524 | 1.3514678 | 1.36426082 | 6570.3247897155225 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 329.661463 | 1.097328 | 1.30485685 | 1.31411678 | 14273.94108812644 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 336.462149 | 1.29135 | 1.31839495 | 1.34109632 | 12407.03139968405 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 331.70622 | 1.3028629999999999 | 1.3572398 | 1.36913013 | 12254.972359292813 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 336.384173 | 1.3031350000000002 | 1.3502625 | 1.3956726899999998 | 12308.640830491076 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 332.934483 | 1.1857165 | 1.352543 | 1.36918987 | 26133.760524494122 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 330.827065 | 1.486889 | 1.5942740999999998 | 1.61949619 | 21368.654008917754 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 332.404501 | 1.4677254999999998 | 1.55404345 | 1.58315959 | 21638.925158883132 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 350.531701 | 2.051368 | 2.1298778499999997 | 2.17778154 | 15559.17889934174 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 330.613146 | 1.300606 | 1.43300925 | 1.45572109 | 47966.59845916297 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 341.9085 | 1.9347504999999998 | 2.0198096 | 2.04975742 | 33000.0054346884 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 339.923866 | 1.910011 | 1.9697917 | 1.98391334 | 33482.6698775431 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 335.156994 | 1.8962815000000002 | 1.9524317 | 1.9686040200000001 | 33705.22023185758 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 332.032762 | 1.671947 | 1.8048364499999998 | 1.81450107 | 75183.53593338362 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 338.663565 | 2.5605835 | 2.6724705 | 2.68524137 | 49905.53233627018 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 343.344622 | 2.6966185 | 2.73877805 | 2.7520483 | 47671.54697425005 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 342.847092 | 2.6369285 | 3.2097972 | 3.30173022 | 47361.10473565684 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 499.499592 | 0.6315230000000001 | 0.6446476999999999 | 0.71960642 | 1576.311095965914 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 521.867844 | 0.5952265000000001 | 0.6673638 | 0.6882625299999999 | 1656.0625119878223 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 511.569871 | 0.586032 | 0.6554517 | 0.67880878 | 1681.7628426474066 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 547.617014 | 0.5879075 | 0.6847328500000001 | 0.70105294 | 1671.8986505738173 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 499.977754 | 0.644328 | 0.6913671499999999 | 0.71657809 | 3084.2596290277197 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 503.148509 | 0.639198 | 0.6915511999999999 | 0.71182752 | 3104.771251010448 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 508.546493 | 0.639537 | 0.6872952999999999 | 0.70969254 | 3103.463611252948 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 551.771965 | 0.6495785000000001 | 0.7210967 | 0.72575865 | 3038.413661242582 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.536341 | 0.6779205 | 0.6891995 | 0.74042166 | 5879.277387423155 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.967507 | 0.6917795 | 0.7452949499999999 | 0.7623805300000001 | 5730.19502776079 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 511.798354 | 0.6917095 | 0.7029605000000001 | 0.7416814399999999 | 5764.978986363346 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 533.392469 | 0.6754145 | 0.7533115 | 0.7572167 | 5846.53517516731 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 497.896912 | 0.7445200000000001 | 0.904486 | 0.9074573899999999 | 10523.673516604686 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.980529 | 0.738294 | 0.87917485 | 0.90278516 | 10569.573437647274 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 513.382075 | 0.6670575000000001 | 0.7400839499999998 | 0.77753727 | 11848.921043174269 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 540.829351 | 0.747498 | 0.8462027 | 0.8582300599999999 | 10446.344895739603 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.856892 | 0.8042885 | 0.9716083 | 0.98566754 | 19481.966237898625 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 516.192883 | 0.7960875000000001 | 0.90280185 | 0.9525288 | 19673.93792459432 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 517.084404 | 0.7532995 | 0.7907415 | 0.80013476 | 21070.519340893083 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 540.896931 | 0.7433255 | 0.77346895 | 0.78808717 | 21391.8940711911 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 496.224232 | 0.9171469999999999 | 1.0545972999999997 | 1.07263403 | 34339.992322865466 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 511.98229 | 0.95965 | 1.0470727999999998 | 1.08484596 | 33020.24568259771 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 510.104614 | 0.8722745000000001 | 0.88986855 | 0.8981497399999999 | 36663.43409751158 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 545.926223 | 0.862431 | 0.88847915 | 0.89282973 | 37004.15815725213 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 499.673875 | 1.1294870000000001 | 1.24753745 | 1.27381913 | 55892.958835604346 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 514.188314 | 1.4580525 | 1.5432241999999998 | 1.57066735 | 43500.69978352285 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.41081 | 1.224937 | 1.24857085 | 1.26339406 | 52203.78438936531 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 543.235351 | 1.1066924999999999 | 1.1387994 | 1.1481172 | 57710.91807610556 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.841748 | 1.541767 | 1.6529595000000001 | 1.66963288 | 81741.20204115956 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 509.480691 | 1.84911 | 1.90322845 | 1.9239356 | 69185.56910103299 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 517.64694 | 1.8251475 | 1.8713293 | 1.8838767 | 70017.96573478298 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 550.780123 | 1.6711365 | 1.6994861 | 1.71239464 | 76514.93595460751 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.568949 | 0.893993 | 0.9469734 | 0.97057918 | 1111.17978202164 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.455307 | 0.944982 | 0.9657441 | 0.9894484299999999 | 1055.938712303437 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 498.527054 | 0.94577 | 0.990297 | 1.00041882 | 1049.254019214737 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.014622 | 0.9347675 | 0.94533025 | 0.95272506 | 1069.3475752180154 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 491.4639 | 1.2595165000000001 | 1.28274765 | 1.2862570500000001 | 1587.011063514357 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.022327 | 1.276567 | 1.3138208 | 1.4740845699999996 | 1555.860850029017 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 511.010833 | 1.295299 | 1.31280665 | 1.31534399 | 1543.178012774983 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 502.531631 | 1.349421 | 1.3818546999999999 | 1.3915141500000001 | 1479.282566298449 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 495.026081 | 1.6239620000000001 | 1.7357856999999999 | 1.74910042 | 2440.049538373738 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 497.178591 | 1.755101 | 1.8362681 | 2.00264841 | 2269.225471025684 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 502.56578 | 1.7065155 | 1.7443377999999998 | 1.77069747 | 2341.564067622263 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 498.296735 | 1.8117105 | 1.84372065 | 1.86663734 | 2205.0439741445807 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.25853 | 1.5791650000000002 | 1.7078548 | 1.7241731 | 5009.3919210606 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 508.146231 | 1.801862 | 1.87606905 | 1.9001814 | 4433.195408127229 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 511.314305 | 1.815407 | 1.89255945 | 1.90835025 | 4411.135324445565 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 505.109414 | 1.847825 | 1.8929987 | 1.9805342499999996 | 4314.588979040428 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 499.571865 | 1.633375 | 1.7386485499999997 | 1.7613759900000001 | 9738.161292970732 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 499.779602 | 1.8807775000000002 | 1.9842756 | 2.09443018 | 8492.233459735275 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 512.772823 | 1.921492 | 2.0048694 | 2.0340283899999996 | 8320.645735361197 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 507.096259 | 1.9723545 | 2.05347855 | 2.08171278 | 8090.687791290676 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.399969 | 1.6516125 | 1.699551 | 1.74821254 | 19252.260260477065 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 504.363555 | 2.1192085 | 2.2539346 | 2.29921843 | 15002.939310231799 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 513.086532 | 1.9943075000000001 | 2.0692724 | 2.1196783 | 16051.25696038932 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 505.896752 | 2.1269055 | 2.2290098 | 2.2529228199999998 | 15015.131263919788 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 495.594223 | 1.7572455 | 1.7950948 | 1.79649692 | 36385.98149104248 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 503.43731 | 2.2464775 | 2.3686698 | 2.4411707499999995 | 28316.535757847036 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.214052 | 2.3399805000000002 | 2.44698525 | 2.46853639 | 27396.551435116126 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.000546 | 2.4112295 | 2.49585465 | 2.5032482799999998 | 26540.64130665814 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 491.940527 | 1.9635525 | 2.01509845 | 2.16012307 | 64961.5986107475 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 504.595034 | 2.6383975 | 2.7143202 | 2.72612691 | 48517.52908381385 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 505.833362 | 2.5399950000000002 | 2.6094103 | 2.61881922 | 50405.937523305365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 511.007354 | 2.761765 | 2.81798165 | 2.83678863 | 46407.04544802935 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 15.462898 | 0.1378765 | 0.16045004999999998 | 0.17346268999999997 | 7100.419961439041 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 17.495906 | 0.1406155 | 0.16427784999999998 | 0.17081686 | 6951.756340384129 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 17.800271 | 0.1396305 | 0.15979595 | 0.16416545 | 7093.909169586992 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 18.11155 | 0.138327 | 0.15751469999999998 | 0.16424534999999998 | 7131.406142137481 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 17.221662 | 0.140122 | 0.15888945 | 0.16532316 | 14122.279166390104 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 17.41827 | 0.147403 | 0.16525959999999998 | 0.17237111 | 13492.805703462942 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 17.767646 | 0.1424105 | 0.16076915 | 0.16844016999999997 | 13862.665346940925 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 18.273529 | 0.142089 | 0.16229785 | 0.16810518 | 13920.75194333697 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 16.983674 | 0.150739 | 0.17399945 | 0.17792856999999998 | 26189.56610305411 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 17.203864 | 0.21187450000000002 | 0.22368739999999998 | 0.22930961 | 18717.86376764393 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 18.033243 | 0.2122695 | 0.2220237 | 0.22677217 | 18786.755112204835 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 18.538673 | 0.2172245 | 0.2306765 | 0.25743707 | 18209.91276632339 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 17.011158 | 0.1829065 | 0.20129729999999998 | 0.20930838999999998 | 43102.64010136017 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 17.031495 | 0.261412 | 0.2920507 | 0.3173515 | 30066.425754419237 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 18.156043 | 0.2842615 | 0.2939869 | 0.29704933 | 28121.449227637364 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 18.375826 | 0.2825165 | 0.2952108 | 0.29719627 | 28180.345758752286 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 16.901116 | 0.2375505 | 0.24802259999999998 | 0.25150903999999996 | 67017.03732254203 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 17.529316 | 0.33535 | 0.35482274999999996 | 0.37378261999999995 | 47346.82031183325 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 17.949079 | 0.3439235 | 0.3527496 | 0.35906801 | 46499.22991462858 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 18.492667 | 0.3538195 | 0.36466985 | 0.36824941 | 45139.32931081724 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 17.185446 | 0.39735750000000003 | 0.40980595 | 0.4372649199999999 | 80129.50931943754 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 17.458962 | 0.6086325 | 0.65594485 | 0.6733819 | 52403.686691065166 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 18.131308 | 0.595378 | 0.6253168 | 0.6518863199999999 | 53586.55840127719 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 18.360792 | 0.610854 | 0.6389897 | 0.6439429200000001 | 52372.368456240314 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 17.071468 | 0.605446 | 0.6133223999999999 | 0.61643958 | 105787.21523156062 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 17.409714 | 1.1952445 | 1.23256865 | 1.25272576 | 53419.93416927962 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 18.008183 | 1.1553725 | 1.2246446 | 1.2275664 | 54785.041047264 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 19.025308 | 1.1856925 | 1.2375314 | 1.2528205000000001 | 53780.77775172585 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 17.062194 | 1.0371665 | 1.12924295 | 1.14065528 | 121765.92787282782 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 17.538489 | 1.72687 | 1.7457509999999998 | 1.74908974 | 74204.20196366348 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 17.731015 | 1.633336 | 1.7184082 | 1.72801432 | 77842.88416059318 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 17.865017 | 1.562691 | 1.62535275 | 1.65309594 | 81470.55782610888 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 318.082084 | 0.8778315000000001 | 0.91756235 | 0.9235559799999999 | 1142.9372398989149 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 361.654957 | 0.854661 | 0.90321025 | 0.92042412 | 1177.8259749401507 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 351.003396 | 0.8783559999999999 | 0.9436386999999999 | 0.9699683299999999 | 1137.8757439232486 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 334.110215 | 0.848608 | 0.9136961 | 0.92552738 | 1176.8272638114154 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 347.721309 | 1.197386 | 1.2752329 | 1.30071627 | 1679.302494320053 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 371.955491 | 1.1749415 | 1.3204666999999999 | 1.3616762999999998 | 1690.5188450841836 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 334.985156 | 1.1945145 | 1.2954823 | 1.31565497 | 1669.1640420870365 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 312.293718 | 1.207753 | 1.3084921 | 1.3306096699999999 | 1654.0143938287001 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 306.60511 | 1.2702545 | 1.43344225 | 1.4789219299999998 | 3113.4370353827644 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 322.03437 | 1.3023254999999998 | 1.42928165 | 1.4659969899999998 | 3062.222430589446 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 337.609626 | 1.283521 | 1.38124445 | 1.45078701 | 3120.5415262943075 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 365.644742 | 1.3305635 | 1.4211406 | 1.45225115 | 2986.211303684744 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 314.706563 | 1.4494175 | 1.6024204 | 1.6541751399999998 | 5476.050002469699 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 310.969235 | 1.4576655 | 1.5918492999999998 | 1.65439525 | 5491.628952103975 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 330.321025 | 1.457224 | 1.62110945 | 1.6817845299999998 | 5479.622373468019 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 306.838853 | 1.5044559999999998 | 1.63842095 | 1.67437627 | 5304.308370104457 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 311.540297 | 2.6332190000000004 | 2.9546246 | 3.04956315 | 6027.600200490037 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 310.522505 | 2.561654 | 2.83471125 | 3.02886206 | 6249.725451513954 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 364.982829 | 2.6787845 | 2.92282155 | 2.9829481099999997 | 5961.070054696469 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 312.550622 | 2.7528075000000003 | 3.10736595 | 3.1902511299999996 | 5754.643232800316 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 333.612695 | 2.7884485000000003 | 3.0015446 | 3.07863734 | 11443.338239187799 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 308.288627 | 3.0069784999999998 | 3.2999240999999997 | 3.41849713 | 10605.237103373827 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 360.889552 | 3.093886 | 3.35427385 | 3.44450608 | 10329.15531912119 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 311.109873 | 2.746398 | 3.13638875 | 3.16510716 | 11563.701958735724 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 364.288069 | 3.4255455 | 3.7279151 | 3.9518856099999997 | 18618.677738511633 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 384.945227 | 3.894932 | 4.25487055 | 4.41275311 | 16925.963076455853 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 330.323326 | 3.5595559999999997 | 3.8697901 | 3.99794408 | 17881.154393693276 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 372.197059 | 3.5347565 | 3.74338705 | 3.7731007599999997 | 18138.51248326439 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 329.207386 | 3.529001 | 3.89767595 | 3.9606235599999997 | 35901.4323482272 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 404.946858 | 3.2052199999999997 | 3.3873083499999996 | 3.4554761299999996 | 39899.434224152414 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 330.764601 | 3.581456 | 3.8363479 | 3.9345239999999997 | 35653.67307733265 | - |
| `separator1_grid_search_6ports_learned_dense_hd128_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 311.761654 | 3.6552035 | 4.03528355 | 4.13285799 | 34609.081207683674 | - |
