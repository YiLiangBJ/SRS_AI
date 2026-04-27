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

### full_mlp_capacity_search_hd32_depth5::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1693836.420` samples/s, p50=`0.072` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.046` ms, throughput=`21034.591` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2943962.141` samples/s, p50=`0.044` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.023` ms, throughput=`43638.242` samples/s

### full_mlp_capacity_search_hd32_depth5::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1937055.385` samples/s, p50=`0.065` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.043` ms, throughput=`23051.481` samples/s

### full_mlp_capacity_search_hd32_depth5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3831013.974` samples/s, p50=`0.034` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.011` ms, throughput=`92358.788` samples/s

### full_mlp_capacity_search_hd32_depth5::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`424743.879` samples/s, p50=`0.290` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.138` ms, throughput=`7119.836` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `7,664`
- MACs / sample: `7,424`
- FLOPs / sample estimate: `15,160`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.047452 | 0.0536122 | 0.05673501999999999 | 20530.845542342817 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0465955 | 0.05550785 | 0.05849450999999999 | 20447.413954214968 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.046066499999999996 | 0.0525975 | 0.05358594 | 21034.590542763955 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.0479245 | 0.0532109 | 0.05369157 | 20327.012851957144 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.051373 | 0.0582787 | 0.061591309999999996 | 38136.54178601815 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.050155 | 0.05782755 | 0.059783789999999996 | 38255.181186101894 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.049728999999999995 | 0.05793659999999999 | 0.09267769999999986 | 37912.11895003145 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.050707 | 0.058717399999999996 | 0.060490789999999996 | 38373.66277378649 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.052412 | 0.05875385 | 0.06013105 | 74946.33842169007 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.050736500000000004 | 0.058115549999999995 | 0.059856179999999995 | 76311.76101598426 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0508385 | 0.05871095 | 0.05964353 | 76462.15715802397 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.049275 | 0.056983349999999995 | 0.10777325999999982 | 75302.19712985677 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0519235 | 0.0576984 | 0.06390441 | 150491.7506065758 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0519205 | 0.0607761 | 0.06191658 | 149662.0630616069 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.0531885 | 0.0608769 | 0.06250793 | 148808.1951649241 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.050416 | 0.0580896 | 0.06199784999999999 | 152600.63916777715 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.0520625 | 0.0594772 | 0.060635579999999994 | 295929.1249745686 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.053055000000000005 | 0.06102075 | 0.08065841999999993 | 289021.3088185459 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0523765 | 0.05952655 | 0.09632217999999987 | 287670.2378529444 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.052227499999999996 | 0.06054015 | 0.06177224 | 295792.1350350255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.056565000000000004 | 0.0647739 | 0.07218465999999997 | 549004.2949292248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.056037000000000003 | 0.06401124999999999 | 0.06738167999999999 | 558440.8331937231 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.063809 | 0.07286775 | 0.10677148999999989 | 476378.6173884745 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.0569185 | 0.0649414 | 0.11409371999999982 | 534719.6832320596 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0621785 | 0.07018925 | 0.07415047 | 999597.0374442802 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.077614 | 0.10908169999999992 | 0.14373076999999995 | 780444.5948968191 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.071606 | 0.08056955 | 0.0818348 | 881960.1122514734 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.071352 | 0.08479904999999999 | 0.11642780999999994 | 856376.323870821 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.07151199999999999 | 0.0838997 | 0.12486994999999984 | 1693836.4203943198 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.0952005 | 0.1065183 | 0.10807676000000001 | 1326614.858210782 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.123361 | 0.13122255 | 0.13275614000000002 | 1033943.3922454893 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.086726 | 0.09637855 | 0.09856041 | 1459372.6612983036 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0607535 | 0.0686253 | 0.06983667 | 16119.656857640543 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.063644 | 0.07182369999999999 | 0.08290284999999996 | 15198.617047437621 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.06348200000000001 | 0.07248455 | 0.07418124999999999 | 15268.55557021337 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0632925 | 0.0726628 | 0.07387484999999999 | 15288.934878005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.06820899999999999 | 0.07944369999999999 | 0.08684472999999998 | 28366.44453849497 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.0679755 | 0.0791547 | 0.08396620999999999 | 28488.09681850632 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.072366 | 0.07891555 | 0.08169678000000001 | 27394.881047317256 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.0725055 | 0.08859189999999999 | 0.09919271999999998 | 26562.844901939276 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0694855 | 0.0904023 | 0.12680745999999987 | 52782.592723180634 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0755685 | 0.09325495 | 0.09657342 | 50551.5681601959 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.07489599999999999 | 0.09029324999999999 | 0.14125148999999987 | 50736.68396703028 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.073527 | 0.08998475 | 0.09118894 | 51808.2369915993 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.10214200000000001 | 0.12139549999999998 | 0.1991172299999998 | 73804.78229157576 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.103743 | 0.12102565 | 0.12190933999999999 | 73989.69841428979 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1150545 | 0.1340492 | 0.18237351999999982 | 65943.36684740095 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.106983 | 0.1119312 | 0.11496998 | 74584.6057000912 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.1089395 | 0.12118155 | 0.12475073999999998 | 145665.99951638887 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1138275 | 0.13730769999999998 | 0.2058170299999998 | 132647.7654654859 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.106903 | 0.11703049999999998 | 0.12640085999999998 | 147859.84882439408 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.119783 | 0.139804 | 0.19138652999999978 | 127297.82519621764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1168 | 0.13744425 | 0.21556917999999975 | 261338.283951559 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.11698549999999999 | 0.13660275 | 0.14152063 | 267402.2009875163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.11834449999999999 | 0.14247295000000001 | 0.14762348999999997 | 263002.6453134819 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.11722250000000001 | 0.12576959999999998 | 0.12879419 | 271345.9829177528 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.119813 | 0.12850175 | 0.13147293000000002 | 529568.2793896329 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1500445 | 0.17269734999999997 | 0.2355019699999998 | 413195.3949373234 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.14937699999999998 | 0.15905 | 0.16401961 | 427103.8501676984 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.1585695 | 0.17586185 | 0.19608445 | 397323.38137593836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.14096599999999998 | 0.1612429 | 0.17329660999999996 | 888622.9191124045 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1903695 | 0.2223599 | 0.27962259999999994 | 662004.4999755886 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.187822 | 0.20760089999999998 | 0.2176432 | 676670.9649370252 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1905735 | 0.2051391 | 0.21603510999999997 | 668611.4204889911 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 52.992674 | 0.023823999999999998 | 0.0256106 | 0.02854847999999999 | 41477.94203042822 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 53.141493 | 0.023084 | 0.02400805 | 0.025186579999999997 | 43489.87077419798 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 52.852568 | 0.023164 | 0.024911549999999998 | 0.026957859999999993 | 42691.657793917126 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 53.262501 | 0.0230715 | 0.023817349999999998 | 0.02441759 | 43638.242461057234 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 53.336696 | 0.024933 | 0.0264812 | 0.028361839999999992 | 78893.09827397681 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 52.91441 | 0.0246135 | 0.026553499999999997 | 0.027607129999999997 | 80186.03159329644 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 52.775157 | 0.02437 | 0.0249175 | 0.026793329999999997 | 81687.9341203149 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 52.843486 | 0.0251095 | 0.02575465 | 0.026941169999999993 | 79308.17889387297 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 51.760894 | 0.024768 | 0.02615915 | 0.027936419999999997 | 160655.9905405753 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 51.968042 | 0.02422 | 0.02546165 | 0.028720839999999987 | 162863.00127196006 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 52.850189 | 0.025257500000000002 | 0.026924550000000002 | 0.029983819999999987 | 155457.6517810783 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 51.572509 | 0.0245095 | 0.025523 | 0.026310359999999998 | 161565.3746014384 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 51.514723 | 0.0254185 | 0.0259892 | 0.03077187 | 312484.37578121095 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 51.819779 | 0.0255675 | 0.02750015 | 0.028956129999999997 | 307938.1844888457 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 51.118595 | 0.025147 | 0.02559 | 0.028079719999999996 | 317251.3344384255 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 53.03366 | 0.0259525 | 0.02808385 | 0.02871311 | 303392.7654478115 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 53.439381 | 0.0276155 | 0.028173249999999997 | 0.02842242 | 577971.6230382379 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 53.267027 | 0.026976 | 0.0343171 | 0.03554881 | 574882.3826575232 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 50.950239 | 0.0268595 | 0.02800875 | 0.028626049999999997 | 592709.3782923154 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 51.439991 | 0.0276495 | 0.0280293 | 0.031516169999999996 | 576565.3569263878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 53.319122 | 0.0297725 | 0.03252059999999999 | 0.03666 | 1066771.9214962544 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 53.259688 | 0.0299095 | 0.04015295 | 0.04256007999999999 | 1011397.1819946016 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 52.570523 | 0.0350675 | 0.0399177 | 0.042508649999999995 | 899485.8314116194 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 51.798022 | 0.0298345 | 0.031110499999999996 | 0.03417363999999999 | 1066103.7638793385 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 53.29965 | 0.0345695 | 0.035661849999999995 | 0.03999729999999998 | 1841066.4837873958 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 52.827586 | 0.047354499999999994 | 0.0500901 | 0.05339508 | 1343074.5410567396 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 52.347494 | 0.043005 | 0.0449532 | 0.04652493 | 1483246.9573738 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 52.99994 | 0.041482000000000005 | 0.045830749999999996 | 0.04740453 | 1529569.4453207604 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 52.745362 | 0.043576500000000004 | 0.0447517 | 0.04574131 | 2943962.140646871 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 55.141881 | 0.0684915 | 0.07085205 | 0.07139898 | 1874828.2628017084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 55.04745 | 0.0799715 | 0.08473544999999999 | 0.08522188 | 1600208.8272519566 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 54.955271 | 0.0581995 | 0.06294815 | 0.06893137999999999 | 2186483.8401762308 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 53.631544 | 0.033182500000000004 | 0.0352344 | 0.03931688999999999 | 29853.223640648386 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 52.606137 | 0.033640500000000004 | 0.0361181 | 0.038542429999999996 | 29462.665793786084 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 54.481913 | 0.034822000000000006 | 0.036794349999999997 | 0.04129833999999999 | 28468.17855469335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 53.339609 | 0.035936499999999996 | 0.037094800000000004 | 0.04087763 | 27675.593337045557 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 51.921549 | 0.038043 | 0.04042125 | 0.041667869999999996 | 52393.27228469247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 54.049811 | 0.041174 | 0.04315595 | 0.04917515999999999 | 48477.468157575044 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 52.350797 | 0.0418915 | 0.04442525 | 0.046552859999999995 | 47392.196969553355 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 53.376605 | 0.0421575 | 0.04405065 | 0.04708537999999999 | 47340.16893812687 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 52.376996 | 0.042407 | 0.045088249999999996 | 0.048662489999999996 | 93503.12253677711 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 53.673543 | 0.04485 | 0.0467924 | 0.049777449999999994 | 88752.52556405558 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 53.476298 | 0.048061 | 0.05012735 | 0.050438369999999996 | 83245.26677818756 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 54.203289 | 0.0483035 | 0.050212099999999996 | 0.05291982 | 82604.41817991078 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 54.782707 | 0.0670485 | 0.07293125 | 0.07459853 | 117663.774000836 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 54.008694 | 0.0717955 | 0.0788725 | 0.08276819 | 109908.07563324226 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 54.490833 | 0.06710150000000001 | 0.0726401 | 0.07865835999999998 | 118097.91498131101 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 55.496039 | 0.071766 | 0.0774214 | 0.081104 | 110427.38157353496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 54.709174 | 0.06903799999999999 | 0.08047295 | 0.08142385 | 224277.22459177335 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 55.518829 | 0.072617 | 0.0849096 | 0.08539917999999999 | 218024.93932774736 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 53.675828 | 0.077003 | 0.0888093 | 0.09105590999999999 | 204452.4112734037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 54.321037 | 0.076781 | 0.0883171 | 0.08946583000000001 | 205779.73542899414 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 53.363614 | 0.07685700000000001 | 0.09024765 | 0.09152049 | 404880.4286873979 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 54.068048 | 0.08283199999999999 | 0.0958633 | 0.09952544999999999 | 380155.77358017163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 56.019527 | 0.085894 | 0.09807384999999999 | 0.09889682 | 367545.7979292929 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 55.915668 | 0.0867705 | 0.09606239999999999 | 0.09905837999999999 | 367191.19163409946 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 55.969124 | 0.083199 | 0.08717130000000001 | 0.09617401999999999 | 763247.8964768716 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 55.087844 | 0.1066405 | 0.11504039999999999 | 0.12137497 | 594339.2897274025 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 55.115974 | 0.1122265 | 0.12166129999999999 | 0.12471181999999999 | 565141.8726858986 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 57.178889 | 0.1178875 | 0.12298945 | 0.12521260999999997 | 544429.347872319 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 55.001921 | 0.106412 | 0.11923675 | 0.2037421199999997 | 1150021.5718890163 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 55.812498 | 0.1313315 | 0.13956515 | 0.14454462999999998 | 969663.4722315653 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 57.422298 | 0.150005 | 0.1580947 | 0.16187115 | 851090.8790851199 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 56.100105 | 0.1496205 | 0.1580504 | 0.16358911999999998 | 849352.8130631525 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 493.960136 | 0.0456975 | 0.0481955 | 0.04849944 | 21811.611367688365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 510.409754 | 0.0458685 | 0.048965800000000004 | 0.04978990999999999 | 21708.46486854005 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 513.807306 | 0.044548500000000005 | 0.0485055 | 0.051229779999999996 | 22273.24279705601 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 543.347859 | 0.0431815 | 0.04523345 | 0.05108328 | 23051.481334293505 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.229641 | 0.045307 | 0.0510905 | 0.052832849999999994 | 43032.296599243666 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 504.698229 | 0.0460005 | 0.05186595 | 0.055868779999999986 | 42501.43017312532 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 518.871666 | 0.0445015 | 0.050234350000000004 | 0.05137115 | 44420.85203636291 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.923998 | 0.046943 | 0.0502858 | 0.05337557 | 42090.90794297523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 495.70932 | 0.044681 | 0.04925135 | 0.05190797 | 88753.78590367996 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 508.759429 | 0.0446845 | 0.05116844999999999 | 0.05530617999999999 | 88175.56813722939 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 517.528896 | 0.045315499999999995 | 0.0506975 | 0.05101793 | 86724.92847361523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.442498 | 0.046937 | 0.0536094 | 0.05809876999999999 | 83606.51762968834 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.409474 | 0.045318 | 0.05184835 | 0.052239110000000005 | 173313.0788115229 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 498.550401 | 0.0460845 | 0.05264965 | 0.05541276999999999 | 168829.51342490086 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 506.08204 | 0.048956 | 0.054503249999999996 | 0.05740551 | 161791.0921060508 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 542.66852 | 0.046268000000000004 | 0.0520781 | 0.05542397999999999 | 168010.11756928003 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.337111 | 0.0511505 | 0.053894399999999995 | 0.05650124 | 312005.42265424575 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 520.422364 | 0.050189 | 0.05597005 | 0.05997434999999999 | 315468.2357005149 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 519.9207 | 0.047279 | 0.0495587 | 0.054837569999999995 | 337147.8806462788 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 536.908374 | 0.046939999999999996 | 0.0544233 | 0.05522563 | 331394.5372096 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.653316 | 0.048959 | 0.0571436 | 0.06003140999999999 | 642636.3513678515 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 507.505513 | 0.053032499999999996 | 0.05596075 | 0.05875399999999999 | 602841.5690911071 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 506.672688 | 0.0588165 | 0.0631464 | 0.06647891 | 538707.6605239403 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 543.244693 | 0.0516415 | 0.05419255 | 0.05474886 | 617911.8749531739 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.645162 | 0.0545975 | 0.05708925 | 0.05728349 | 1170079.8872042987 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 502.736722 | 0.0828995 | 0.08910295 | 0.09005512 | 766054.9558249622 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 519.279129 | 0.0669845 | 0.07518129999999999 | 0.07924409999999998 | 939855.1448258037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 549.701394 | 0.06386649999999999 | 0.06825679999999999 | 0.07071010999999999 | 995515.8234129145 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 493.648263 | 0.065469 | 0.07004735 | 0.07466360999999999 | 1937055.385256103 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 508.524886 | 0.114531 | 0.12153934999999999 | 0.12560908 | 1116852.4306373564 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 513.591429 | 0.1390225 | 0.1481916 | 0.15022943 | 924263.268304673 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 552.016133 | 0.0813815 | 0.0848854 | 0.09214128999999999 | 1566083.7164732676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 497.813036 | 0.065191 | 0.07004555 | 0.07126745 | 15188.044700238119 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 501.533826 | 0.069812 | 0.0736078 | 0.07632779999999999 | 14242.547159922156 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.596373 | 0.071851 | 0.08035595 | 0.08212966999999999 | 13844.02271859504 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.653985 | 0.061783500000000005 | 0.06795625 | 0.07435601999999998 | 15925.215189470247 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.64358 | 0.0704125 | 0.07386655 | 0.07500885 | 28305.374228183206 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 502.418021 | 0.0719515 | 0.0776732 | 0.08722723 | 27454.305054667013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 502.628519 | 0.0674975 | 0.0755839 | 0.08070213999999999 | 29155.87049078951 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 497.543374 | 0.06700400000000001 | 0.07320584999999999 | 0.07611883 | 29474.35436426765 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 491.813824 | 0.066674 | 0.07053785 | 0.07224815 | 59969.13388678846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 496.77215 | 0.06925899999999999 | 0.07467755 | 0.07598275 | 57343.6089257621 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 496.761111 | 0.072549 | 0.08024285 | 0.08188514000000001 | 54392.80361451059 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 508.729647 | 0.069666 | 0.07653579999999999 | 0.08095428 | 56635.804872661465 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.254403 | 0.096102 | 0.1011917 | 0.10296726 | 83068.98359906755 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 506.539087 | 0.096789 | 0.10310815 | 0.10793699999999999 | 81876.64546470933 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 502.799503 | 0.1070685 | 0.11087035 | 0.12049736999999998 | 74191.88804444094 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 499.958643 | 0.10134750000000001 | 0.108272 | 0.11504268999999999 | 78284.35146699003 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.052476 | 0.095616 | 0.1009864 | 0.1020325 | 166783.34551546685 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.224918 | 0.105144 | 0.1132486 | 0.11792952 | 150567.68723327172 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.740282 | 0.11222599999999999 | 0.1198317 | 0.12406857999999998 | 141894.691435804 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 498.285603 | 0.10354 | 0.1154248 | 0.11726734999999999 | 152936.1835540075 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 488.917138 | 0.102493 | 0.10955039999999999 | 0.1127145 | 308505.3379136093 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 511.301836 | 0.10012750000000001 | 0.1061556 | 0.11329594999999999 | 315641.8695783577 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 503.031886 | 0.114751 | 0.12013025 | 0.12778357999999998 | 277529.9762729216 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.226254 | 0.105229 | 0.12136594999999999 | 0.1241679 | 299609.42166767846 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.722307 | 0.0973785 | 0.10396014999999999 | 0.10656940999999999 | 651763.9482067631 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.939422 | 0.1289705 | 0.1363575 | 0.14068401 | 496192.4978795524 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 505.412872 | 0.14307150000000002 | 0.1543904 | 0.1584351 | 444555.95389621344 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 497.03818 | 0.1272395 | 0.1357702 | 0.14059918999999999 | 499633.237976248 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.811563 | 0.107661 | 0.11316644999999999 | 0.12269689999999998 | 1178968.9621683597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 501.541582 | 0.14997549999999998 | 0.17232845 | 0.17457402 | 829987.3388025176 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 499.989092 | 0.166211 | 0.17479165 | 0.1825268 | 781252.0980891306 | - |
| `full_mlp_capacity_search_hd32_depth5` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 509.15185 | 0.157314 | 0.17211100000000001 | 0.17784198 | 805869.8552922166 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.287242 | 0.010512 | 0.0110298 | 0.01862006999999997 | 92358.78803103995 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.641341 | 0.010703500000000001 | 0.01124735 | 0.017969269999999992 | 91357.23969581693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.763926 | 0.010687499999999999 | 0.0110703 | 0.015493749999999983 | 91954.69943686943 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.074604 | 0.010679000000000001 | 0.011870949999999996 | 0.015059989999999995 | 92244.96573099523 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.37438 | 0.010815499999999999 | 0.01110275 | 0.01115406 | 185297.39305097717 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.58599 | 0.0111145 | 0.011634199999999999 | 0.015536499999999984 | 177323.51433926597 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.703685 | 0.0111745 | 0.0115028 | 0.015569909999999987 | 176231.3282907676 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.471188 | 0.011327 | 0.012013999999999999 | 0.020497529999999993 | 170740.8273759013 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.452422 | 0.011834500000000001 | 0.01221445 | 0.015352339999999989 | 334432.5014882246 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.570373 | 0.0121005 | 0.01252795 | 0.01504637999999999 | 327883.9028676726 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.64827 | 0.012257 | 0.013590699999999997 | 0.018765309999999997 | 320281.8480262631 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.329175 | 0.0119615 | 0.012536199999999999 | 0.019789719999999997 | 327230.48479196324 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.483079 | 0.012659 | 0.01293275 | 0.013000659999999999 | 637642.2540016037 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.51339 | 0.013101999999999999 | 0.01364945 | 0.01714915999999999 | 603115.39256027 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.713931 | 0.013037 | 0.013485099999999998 | 0.023041569999999987 | 595904.9412437727 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.257597 | 0.013268 | 0.014354849999999999 | 0.020164929999999994 | 590721.5368211501 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.455894 | 0.014218 | 0.015606749999999997 | 0.022257089999999997 | 1111748.8225885124 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.469669 | 0.017008 | 0.018335349999999997 | 0.023372719999999996 | 928471.7007628556 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.799704 | 0.0173245 | 0.0239245 | 0.030696109999999974 | 886114.366350693 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.637312 | 0.017183499999999997 | 0.0241767 | 0.031072299999999976 | 891849.8303255697 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.440378 | 0.0181925 | 0.018538950000000002 | 0.022360379999999996 | 1751474.5226137256 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.535269 | 0.028824500000000003 | 0.0315789 | 0.03254272 | 1184174.6894501878 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.69214 | 0.02894 | 0.0307573 | 0.03286699999999999 | 1172671.4775514035 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.321155 | 0.028025 | 0.0297213 | 0.03257176999999999 | 1253177.9810362842 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.401414 | 0.023198 | 0.024119899999999996 | 0.027552969999999996 | 2739217.114628532 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.673533 | 0.045801499999999995 | 0.0520707 | 0.05510935 | 1381192.8153802727 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.57847 | 0.0426915 | 0.05008774999999999 | 0.05360635999999999 | 1469100.231199649 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.092034 | 0.0425575 | 0.049334300000000005 | 0.05453818999999998 | 1484075.8659582678 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.453135 | 0.033987500000000004 | 0.0345879 | 0.034999909999999995 | 3831013.973623469 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.576283 | 0.081541 | 0.1033531 | 0.10623187999999999 | 1551438.4258330015 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.719317 | 0.0822795 | 0.09609949999999999 | 0.09945034 | 1565814.3580283462 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.332584 | 0.0778735 | 0.09724255 | 0.0983687 | 1605335.7346480365 | - |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 229.59445 | 0.167965 | 0.23019685 | 0.3599633999999995 | 5829.97371381452 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 224.455839 | 0.138005 | 0.18035754999999995 | 0.20453682999999998 | 7119.835662801166 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 223.152407 | 0.229269 | 0.35398029999999997 | 0.36722306 | 4191.099730319496 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 221.228387 | 0.20821299999999998 | 0.29727289999999995 | 8.30799291999997 | 1893.0281590210286 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 220.206741 | 0.186154 | 0.27297984999999997 | 0.29186249999999997 | 10404.91561509387 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 218.094662 | 0.229821 | 0.3205238 | 0.38395428999999986 | 8316.616933297406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 223.222728 | 0.20416800000000002 | 0.2654929 | 0.3259662799999999 | 9730.623263935177 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 220.716874 | 0.2201925 | 0.27185085 | 0.3535699699999999 | 8995.38626638397 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 217.170501 | 0.29639899999999997 | 0.9399447499999999 | 0.9935197199999999 | 10050.336608386271 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 218.580343 | 0.209002 | 0.25940525 | 0.27797156999999995 | 19337.55717148778 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 222.148571 | 0.2220695 | 0.28117739999999997 | 0.29669145999999996 | 17642.33638519289 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 219.968875 | 0.47653049999999997 | 1.1151212499999998 | 1.4548481899999999 | 7303.550113290843 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 219.29165 | 0.1612075 | 0.5251393 | 0.7220693399999998 | 36549.283465003646 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 221.71329 | 0.26620350000000004 | 0.6338414499999997 | 0.7414160599999998 | 24863.26911337396 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 222.872618 | 0.3524715 | 0.7260340499999997 | 3.2606268399999907 | 16329.315926990159 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 218.201509 | 0.1930165 | 0.2675661999999999 | 0.29069556999999996 | 41107.98770336764 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 218.37666 | 0.24671349999999997 | 0.31365469999999995 | 0.39661750999999995 | 64055.13866336142 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 221.816911 | 0.8081875000000001 | 1.9850511999999998 | 2.2077821699999998 | 18046.32147843406 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 220.236146 | 0.25938150000000004 | 0.36863189999999996 | 0.49915016999999956 | 58889.91913530754 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 218.151492 | 0.2561095 | 0.40143505 | 0.43889141 | 60410.83293097202 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 218.082193 | 0.34622600000000003 | 0.434223 | 0.4825885899999998 | 90824.2579558786 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 220.609342 | 0.3364475 | 0.5104665999999999 | 0.7638771299999992 | 87311.84774298055 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 223.072349 | 0.4155095 | 0.8300836999999996 | 1.3545749599999992 | 66005.44610935848 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 221.379196 | 0.3162965 | 0.5543362 | 0.8964946599999991 | 92987.45564353073 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 219.515261 | 0.298302 | 0.4127382999999999 | 0.44775493 | 213267.12100116652 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 219.879478 | 0.33313550000000003 | 0.43383505 | 0.6015658199999994 | 183828.71437837547 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 219.185414 | 0.27671500000000004 | 0.39000959999999996 | 0.42932424999999996 | 226414.25062614153 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 222.842665 | 0.448647 | 1.9882444999999995 | 8.860790309999974 | 71854.8196721341 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 218.862923 | 0.2895935 | 0.3804223 | 0.43310853999999993 | 424743.8794406973 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 223.255262 | 0.31475549999999997 | 0.40488545 | 0.4342745699999999 | 391614.840929723 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 220.584301 | 0.28800800000000004 | 0.41230349999999993 | 0.43446301 | 421012.88463885157 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 219.435423 | 0.37323300000000004 | 0.45169515 | 0.5480507299999998 | 338754.30198114633 | - |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth5` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
