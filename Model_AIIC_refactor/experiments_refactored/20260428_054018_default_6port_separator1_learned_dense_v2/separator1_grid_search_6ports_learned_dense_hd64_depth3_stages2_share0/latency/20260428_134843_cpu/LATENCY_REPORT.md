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

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`114790.810` samples/s, p50=`1.112` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.570` ms, throughput=`1717.309` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`217797.307` samples/s, p50=`0.587` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.157` ms, throughput=`6343.227` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`148819.631` samples/s, p50=`0.837` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.482` ms, throughput=`2064.350` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`293347.784` samples/s, p50=`0.436` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.055` ms, throughput=`17938.964` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`38560.469` samples/s, p50=`3.352` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.712` ms, throughput=`1404.675` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0/MODEL_COMPLEXITY.md`
- Trainable parameters: `157,248`
- MACs / sample: `153,600`
- FLOPs / sample estimate: `311,064`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.592365 | 0.6103581499999999 | 0.61404051 | 1682.3651523765225 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5997265 | 0.60911015 | 0.61123218 | 1665.9636855234985 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5804445 | 0.5858213 | 0.5882390200000001 | 1723.7190078769822 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.569814 | 0.69047935 | 0.69637911 | 1717.308668201368 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.6276305 | 0.6423926 | 0.6465751 | 3180.4954715787426 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.6029815000000001 | 0.62625195 | 0.71860297 | 3279.9474867287586 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.615309 | 0.6809176499999998 | 0.7327536 | 3207.821850149104 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.6211395 | 0.6368241 | 0.64473005 | 3207.995556797834 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.656704 | 0.6675508 | 0.6733344 | 6080.787272231191 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.635582 | 0.6506 | 0.65794132 | 6279.519037821952 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.6614694999999999 | 0.67849185 | 0.6826371 | 6034.659220379962 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.6759660000000001 | 0.6882366 | 0.69629224 | 5920.447012694889 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.6668795 | 0.67862595 | 0.68046213 | 11991.68089129847 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.6704405 | 0.67784705 | 0.68255628 | 11935.253992200729 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.6692975 | 0.6898373 | 0.6939052 | 11905.111263086472 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.6776935 | 0.6893325 | 0.6922353499999999 | 11814.426200049602 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6862815 | 0.69977565 | 0.70842803 | 23279.01311417384 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.684393 | 0.71107365 | 0.71801354 | 23240.911394664432 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.726933 | 0.75030285 | 0.75376761 | 21959.30167334545 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6912290000000001 | 0.7059676 | 0.7125321099999999 | 23096.89056089846 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.786114 | 0.8084123 | 0.81426394 | 40551.349324605864 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.7572099999999999 | 0.79275585 | 0.79592366 | 42060.69210658558 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.877609 | 0.9447645 | 0.96350505 | 36058.765240152614 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.77786 | 0.80051285 | 0.80311906 | 41010.21382443543 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.886321 | 0.9067465499999999 | 0.91465741 | 72139.9437358035 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 1.0302175 | 1.15776665 | 1.18846762 | 60883.793329735236 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 1.037318 | 1.1216165999999999 | 1.1546261999999998 | 60886.99296094989 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.017323 | 1.1268177 | 1.1600768000000001 | 61760.05066176956 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.112151 | 1.14408405 | 1.15613072 | 114790.81024949926 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.4152680000000002 | 1.47265945 | 1.4917424499999998 | 89788.14963094123 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.374951 | 1.42658145 | 1.44165111 | 92536.33312605678 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.276592 | 1.34483665 | 1.36614541 | 99349.590827934 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5948495 | 0.6052711 | 0.60647375 | 1678.775957443701 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.589955 | 0.59899815 | 0.60491644 | 1692.514981127104 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.6002995 | 0.60486835 | 0.60712327 | 1665.461927024055 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5975090000000001 | 0.6048979 | 0.60587293 | 1672.2748074609676 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.9435264999999999 | 0.96365925 | 0.9672737 | 2115.3758569149613 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.9615505 | 0.97557075 | 0.9826975 | 2075.625239553098 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.9520575 | 0.96591805 | 0.9702282600000001 | 2100.2998177989907 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.9282395 | 0.9501995000000001 | 0.95372333 | 2150.367177345899 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.32065 | 1.39140565 | 1.3974111599999999 | 2995.8610681413093 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.3797345 | 1.42957005 | 1.44118651 | 2897.685156684574 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.360223 | 1.3986688 | 1.4123237499999999 | 2929.0509748892164 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.4119565 | 1.4514289999999999 | 1.5603439799999996 | 2816.9102559293956 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.6434549999999999 | 1.7029524999999999 | 1.70917693 | 4835.043778420387 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.6138705 | 1.6595256500000002 | 1.67001487 | 4943.5299950721665 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.802968 | 1.87183435 | 1.87438488 | 4418.482570030712 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.718043 | 1.74735925 | 1.76541799 | 4660.278946588683 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.6045595000000001 | 1.66821345 | 1.6869641499999999 | 9899.931124941673 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.725814 | 1.81351505 | 1.82551158 | 9227.47562530594 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.718189 | 1.8119670499999998 | 1.8742087299999999 | 9235.802992056137 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.8250535 | 1.91300865 | 1.9214650400000002 | 8717.130995182326 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.69833 | 1.7425965 | 1.8588531299999997 | 18747.40150762619 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.8713605 | 1.94086875 | 1.96150327 | 17018.68776365604 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.8421405000000002 | 1.8943455 | 1.9048810999999999 | 17348.223993459025 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.926151 | 1.98669535 | 2.04611702 | 16585.711635406416 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.7361435 | 1.77002285 | 1.7827491199999999 | 36848.44380155169 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.209906 | 2.3431702 | 2.39083823 | 28738.43041308341 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.2755270000000003 | 2.3452407500000003 | 2.35462967 | 28052.39760072752 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.285399 | 2.5197489 | 2.5937447899999997 | 27709.203512366686 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 2.05251 | 2.0900408 | 2.09597327 | 62600.01085327688 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.614449 | 2.709784 | 2.7460732 | 49286.53732688181 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.654012 | 2.72452205 | 2.75191825 | 48341.60498479293 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.8060045 | 3.1990250999999996 | 3.21876353 | 44371.53615596251 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 319.498878 | 0.1570865 | 0.16166224999999998 | 0.16440335 | 6343.227203107472 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 317.578827 | 0.166001 | 0.1705463 | 0.18866462999999994 | 5992.296782639981 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 313.893221 | 0.161057 | 0.16355735000000002 | 0.16392794 | 6203.029882352095 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 317.179688 | 0.1584885 | 0.1625598 | 0.2237367299999998 | 6201.524235432681 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 317.878584 | 0.180981 | 0.18490320000000002 | 0.18648432 | 11010.73116870433 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 313.513405 | 0.1728 | 0.17594265 | 0.1772347 | 11546.030849839828 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 315.268831 | 0.173299 | 0.17765889999999998 | 0.17870407 | 11501.56231471702 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 330.436525 | 0.1851265 | 0.19192055 | 0.21035306999999995 | 10718.65586340488 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 318.811771 | 0.18831599999999998 | 0.19258025 | 0.19553898 | 21182.293974654116 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 315.06402 | 0.18198399999999998 | 0.18499259999999998 | 0.18898905 | 21931.635364191803 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 316.495332 | 0.18666349999999998 | 0.1910525 | 0.21572457999999992 | 21265.19398109949 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 315.47025 | 0.18392399999999998 | 0.1909447 | 0.19113383 | 21669.803551312416 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 315.888072 | 0.221086 | 0.2275432 | 0.23040103999999997 | 36054.17355902285 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 318.998595 | 0.2225205 | 0.2267503 | 0.22791527 | 35927.205732832364 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 316.230941 | 0.217537 | 0.22223935 | 0.24099908999999994 | 36547.1562474988 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 316.737226 | 0.22070800000000002 | 0.22439085 | 0.23066194999999998 | 36169.07632588374 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 315.65306 | 0.2490685 | 0.25315425 | 0.2545589 | 64255.61397283305 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 316.992195 | 0.247899 | 0.25337499999999996 | 0.25446248 | 64361.79567479078 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 316.229191 | 0.24262699999999998 | 0.24669539999999998 | 0.24730017 | 65880.18354548537 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 316.106963 | 0.242867 | 0.25058655 | 0.25450993 | 65680.1184475256 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 317.881291 | 0.29952100000000004 | 0.30538025 | 0.31347979 | 106597.7547714654 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 316.946577 | 0.293217 | 0.30074219999999996 | 0.30692355 | 108771.82200687818 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 321.63586 | 0.41383000000000003 | 0.45027995 | 0.45996228 | 76488.13888869235 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 319.942652 | 0.29656150000000003 | 0.30180029999999997 | 0.30636321 | 107855.18042250515 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 321.016627 | 0.409415 | 0.4155562 | 0.43712950999999994 | 156118.51131827495 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 323.873833 | 0.551546 | 0.5854765 | 0.5972202 | 115172.04129594316 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 318.972928 | 0.5771865 | 0.64168325 | 0.6736532199999998 | 108305.43255649794 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 322.84063 | 0.5810715 | 0.61911025 | 0.6312002499999999 | 109211.40898826277 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 316.822538 | 0.5871185000000001 | 0.5966530999999999 | 0.60947116 | 217797.30693629975 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 321.26726 | 0.982159 | 1.0409008499999999 | 1.06016202 | 128480.3004768787 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 322.280927 | 0.9452484999999999 | 1.02415905 | 1.06841426 | 133585.31977486867 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 323.628966 | 0.7900115000000001 | 0.83741545 | 0.84250911 | 160445.67395969972 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 312.676255 | 0.2275245 | 0.2301684 | 0.23160264 | 4388.809168608568 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 314.993849 | 0.222357 | 0.2271853 | 0.22845462 | 4483.994471414176 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 314.697061 | 0.23474050000000002 | 0.24110045000000002 | 0.24265818 | 4238.805843617736 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 314.879629 | 0.225925 | 0.22790405 | 0.22827756000000002 | 4424.274080593816 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 317.812074 | 0.416648 | 0.42475635 | 0.43804497 | 4786.486332284008 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 321.006644 | 0.4344895 | 0.43972525 | 0.44372497 | 4601.956531023077 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 318.463049 | 0.43361150000000004 | 0.44362769999999996 | 0.4820181699999999 | 4592.810387063229 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 315.196333 | 0.43301 | 0.4381507 | 0.44044066 | 4616.431337923975 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 325.676635 | 0.6881079999999999 | 0.8014281 | 0.8238837999999999 | 5692.701754260126 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 327.152399 | 0.7406465 | 0.8334292499999999 | 0.8864531299999999 | 5334.059761152005 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 330.149512 | 0.754505 | 0.7654506 | 0.77983066 | 5292.131759051146 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 326.525991 | 0.7346619999999999 | 0.7506073 | 0.75406562 | 5434.201385417038 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 328.032602 | 0.8107059999999999 | 0.8231454 | 0.82439935 | 9855.985327986002 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 332.870819 | 0.9463145 | 1.0211166499999997 | 1.12158335 | 8368.921999784876 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 328.757406 | 0.9548025 | 0.99744085 | 1.00688665 | 8302.647784951127 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 334.593352 | 1.0053325 | 1.023771 | 1.04741061 | 7952.2219372896925 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 332.425533 | 0.9291175 | 1.0195836999999996 | 1.13550084 | 17002.66410493194 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 328.622215 | 1.007778 | 1.0712077499999997 | 1.19810632 | 15732.69614340683 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 329.838562 | 1.0286680000000001 | 1.1036148499999998 | 1.20430796 | 15421.646613546161 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 329.901027 | 1.077897 | 1.1357331 | 1.23165049 | 14761.552672513604 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 325.590822 | 0.9737705 | 1.0097963 | 1.2190808899999999 | 32481.554638664427 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 334.214803 | 1.0997875000000001 | 1.2838399 | 1.29903607 | 28624.04467698141 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 328.979282 | 1.138124 | 1.25092285 | 1.2638482 | 27843.71469123313 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 331.634476 | 1.126426 | 1.1837943 | 1.3486129199999999 | 28094.89697636397 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 333.075483 | 1.1231944999999999 | 1.2916615999999999 | 1.32885126 | 55181.62436733621 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 331.540364 | 1.546663 | 1.63220635 | 1.64904947 | 41104.12038492571 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 333.993868 | 1.5260470000000002 | 1.57574725 | 1.60625127 | 41805.31497358864 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 336.963036 | 1.6336205000000001 | 1.6945299 | 1.70735164 | 39008.903233600366 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 332.893566 | 1.3385470000000002 | 1.4595729999999998 | 1.53012028 | 93457.21788493115 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 334.708204 | 1.8939145 | 1.9426800999999998 | 1.98296814 | 67423.6097612489 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 337.44816 | 1.964391 | 2.0232283 | 2.04087855 | 65055.01606142276 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 342.13904 | 2.048387 | 2.09842145 | 2.1205415199999997 | 62628.025383764965 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 498.71004 | 0.4988365 | 0.5428165999999999 | 0.60433573 | 1983.1803303097897 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 503.765597 | 0.498884 | 0.5694603500000001 | 0.61329331 | 1971.720713963225 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 505.073484 | 0.501738 | 0.5607493499999999 | 0.5986863499999999 | 1966.4858115885559 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 531.304856 | 0.4815475 | 0.4934768 | 0.5361152899999999 | 2064.3499155680884 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 497.636148 | 0.5057475 | 0.54527255 | 0.5785441699999999 | 3914.5329069728473 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 507.926893 | 0.5073045 | 0.5600268999999999 | 0.5759218500000001 | 3897.8718594115576 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 512.919795 | 0.505811 | 0.5146308 | 0.51703553 | 3948.5786084411375 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 540.544584 | 0.5007955 | 0.56797185 | 0.6174419999999999 | 3927.9682890406466 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 494.27008 | 0.5217635 | 0.55767755 | 0.59487459 | 7597.654360973827 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 499.514762 | 0.518555 | 0.55094965 | 0.5884638099999999 | 7654.7657767975925 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 518.223427 | 0.517191 | 0.59041145 | 0.5994944099999999 | 7600.859125106911 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 546.896275 | 0.511762 | 0.520029 | 0.52459955 | 7803.325605699985 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 494.96263 | 0.5316485 | 0.61606645 | 0.6463000099999999 | 14797.004154147939 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 512.876414 | 0.535032 | 0.6014448499999999 | 0.61920999 | 14750.840963881903 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 516.741141 | 0.5233099999999999 | 0.5529298999999999 | 0.6553681999999998 | 15108.937136509663 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 530.10783 | 0.5243709999999999 | 0.5856915999999999 | 0.65532896 | 15054.919593932686 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 496.015063 | 0.5481659999999999 | 0.64384745 | 0.6773231699999999 | 28618.268500315444 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 507.282997 | 0.5536289999999999 | 0.6387389 | 0.6752087 | 28385.382663343687 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 515.257034 | 0.559518 | 0.6269665 | 0.6460115799999999 | 28209.050076811476 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 545.492353 | 0.5540975 | 0.64760525 | 0.6629006999999999 | 28371.426228759377 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.121508 | 0.6081555000000001 | 0.73746765 | 0.74650863 | 51256.209609706384 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 509.233004 | 0.6051335 | 0.7244132 | 0.73968073 | 51789.6529581829 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 512.522437 | 0.711733 | 0.74572575 | 0.74927587 | 44700.9097054196 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 542.160913 | 0.6035005 | 0.6755859 | 0.73335164 | 52269.30110913497 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 490.915026 | 0.6906595 | 0.8196955999999999 | 0.84901262 | 90157.91158213612 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 509.775267 | 0.8083184999999999 | 0.8332137 | 0.83479755 | 78663.83575187754 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 517.628359 | 0.8317625 | 0.8595021 | 0.8619552500000001 | 76594.03190792477 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 545.599896 | 0.833073 | 0.86368065 | 0.8668043 | 76522.52015920031 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 496.243708 | 0.83748 | 0.9905240999999999 | 1.03208996 | 148819.63128122612 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 514.845604 | 1.2780575 | 1.3278365 | 1.33531413 | 99856.40025540772 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 509.187767 | 1.1786485 | 1.200024 | 1.2362735299999998 | 108221.0565236211 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 540.914338 | 1.025763 | 1.0516188 | 1.1686032999999998 | 123765.7269496003 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 495.118041 | 0.5540215 | 0.5884533 | 0.61293616 | 1791.697595846415 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 511.699655 | 0.554859 | 0.56369815 | 0.56618504 | 1799.7708171841396 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.229918 | 0.5655285 | 0.57213605 | 0.57380899 | 1767.5075689976627 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.925122 | 0.5484845 | 0.572678 | 0.59340516 | 1815.4921830171525 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.38201 | 0.857081 | 0.86730445 | 0.8773356699999999 | 2331.261975984178 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 517.141166 | 0.84771 | 0.85863725 | 0.87179824 | 2356.5239800822824 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 501.956001 | 0.8369825 | 0.84821945 | 0.86892982 | 2385.4538459527344 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.603226 | 0.8385055 | 0.8912055499999999 | 1.0693436199999995 | 2355.3754153969526 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 497.693579 | 1.1551195 | 1.1743864499999999 | 1.19604854 | 3459.246863496545 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 507.018936 | 1.1840435 | 1.2061333 | 1.2247599999999998 | 3376.9518443965426 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 503.684277 | 1.143529 | 1.1540366 | 1.17735458 | 3493.927396957355 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 503.157629 | 1.190097 | 1.2035197 | 1.2098219 | 3359.7161765529513 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.824815 | 1.4526345 | 1.59406755 | 1.59919035 | 5426.793056857854 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.97783 | 1.6431335 | 1.7006907500000001 | 1.73992576 | 4850.063088408138 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 506.631593 | 1.582669 | 1.62422655 | 1.64801151 | 5045.091641693252 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.213198 | 1.5767765 | 1.59938305 | 1.60264038 | 5071.261493285181 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 489.911024 | 1.494015 | 1.64024315 | 1.65051323 | 10484.268217932971 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 507.636982 | 1.5861355 | 1.6121245 | 1.62763109 | 10071.574018446114 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 508.24407 | 1.634638 | 1.66580265 | 1.70969861 | 9766.91692275116 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 503.589081 | 1.8281684999999999 | 1.88153255 | 1.91328334 | 8720.557327330567 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 494.479969 | 1.5351569999999999 | 1.6211269 | 1.6578077599999999 | 20648.98441648248 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 501.353676 | 1.645352 | 1.68015135 | 1.7026578399999999 | 19411.936494383586 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 511.079925 | 1.7242385 | 1.7752092 | 1.78334199 | 18530.47947117625 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 502.316556 | 1.7452675 | 1.79328055 | 1.8841826899999998 | 18246.868891468617 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.467625 | 1.5355755 | 1.66652315 | 1.69866009 | 41052.72635132682 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 502.165467 | 1.9578695000000002 | 2.0643769 | 2.14940244 | 32606.378539247897 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 502.332016 | 1.952737 | 2.0754824999999997 | 2.14358762 | 32541.902402665262 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 509.131919 | 2.035184 | 2.0973001499999997 | 2.10974591 | 31360.996796738782 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 497.336592 | 1.6962545 | 1.7438911 | 1.75590349 | 75186.71475186017 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 499.464902 | 2.214863 | 2.305219 | 2.32808019 | 57697.69934120136 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 500.534914 | 2.2721945 | 2.3500847499999997 | 2.38776339 | 56383.23004865308 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 512.82124 | 2.409751 | 2.45146525 | 2.4570016100000003 | 53216.02924007943 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 13.706376 | 0.0553675 | 0.0576127 | 0.058510339999999994 | 17938.963752171065 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 14.454256 | 0.059319 | 0.06316664999999999 | 0.06788973 | 16696.59810153001 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 14.999447 | 0.0582405 | 0.06373704999999999 | 0.06574272 | 17006.490356979837 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.465732 | 0.060771 | 0.0642585 | 0.06803994 | 16295.76952045787 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 14.182403 | 0.063806 | 0.06779134999999999 | 0.0704967 | 31079.60605977727 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 14.664423 | 0.061319 | 0.0641636 | 0.07106160999999998 | 32174.007327308427 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 15.078317 | 0.061512 | 0.06553694999999998 | 0.07601063999999999 | 32186.693119836855 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.345982 | 0.061393 | 0.06689755 | 0.06928772 | 32198.010034187846 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 14.257712 | 0.065174 | 0.07020879999999999 | 0.0722248 | 60624.06411601022 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 14.438192 | 0.06616749999999999 | 0.06994629999999999 | 0.07395865 | 59754.61171154536 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 14.809982 | 0.0671245 | 0.07376909999999999 | 0.07537521999999999 | 58873.935890404995 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.291646 | 0.0673385 | 0.07228125 | 0.07324174 | 58931.929496196826 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 14.373499 | 0.07919899999999999 | 0.08225715 | 0.08392126 | 100298.43800227577 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 14.587416 | 0.0841925 | 0.0904837 | 0.0918194 | 93772.52005676988 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 15.060945 | 0.08292849999999999 | 0.08652639999999999 | 0.09128604 | 95749.03030169562 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 16.095278 | 0.08237249999999999 | 0.0878372 | 0.08972274000000001 | 96177.8443755918 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 14.209373 | 0.1069725 | 0.11159 | 0.11529727999999999 | 148905.52577238227 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 14.709469 | 0.15123550000000002 | 0.1697884 | 0.17147215999999998 | 104735.7582279757 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 15.040015 | 0.1517415 | 0.16700249999999997 | 0.17381087 | 104517.98591228199 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 16.214364 | 0.1461785 | 0.1563916 | 0.15919485 | 108684.60651009923 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 14.253768 | 0.153507 | 0.15653535 | 0.15798159 | 208858.38918228415 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 14.502171 | 0.27759 | 0.29919165000000003 | 0.30012643 | 115206.97509109991 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 15.062235 | 0.30954950000000003 | 0.3227646 | 0.32546778 | 103033.34017217129 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 15.253142 | 0.30955900000000003 | 0.3179672 | 0.31981673 | 103116.83524120961 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 14.703959 | 0.2487315 | 0.2542913 | 0.25676128 | 257317.50680564597 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 14.429122 | 0.546813 | 0.6035234999999999 | 0.63331024 | 116258.20074448847 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 15.068592 | 0.476122 | 0.49703965 | 0.50103261 | 133959.79071294446 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 15.796381 | 0.49373100000000003 | 0.5564784499999998 | 0.5945633699999999 | 128097.9949661491 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 14.406775 | 0.436267 | 0.44043775 | 0.44195985 | 293347.7843785613 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 14.57107 | 1.0504495 | 1.08496595 | 1.0919775600000001 | 121533.07210179274 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 14.876845 | 1.0069955 | 1.05843805 | 1.06760138 | 126733.79254002914 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 15.618976 | 0.9788079999999999 | 1.0468590500000001 | 1.0783368 | 129770.38205556244 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 316.470849 | 0.7209235 | 0.7802929 | 0.84728562 | 1386.8571816415947 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 309.127933 | 0.71201 | 0.772268 | 0.8200657299999999 | 1404.6751747654712 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 302.260556 | 0.7383045 | 0.80063615 | 0.8244891099999999 | 1360.4415645045342 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 303.759112 | 0.7398640000000001 | 0.8055542 | 0.81107518 | 1352.0235262909764 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 305.507738 | 1.084947 | 3.644958849999992 | 10.957837689999977 | 1234.0973140279214 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 304.579157 | 1.0597155 | 1.3905109999999998 | 1.5482322599999996 | 1856.813601649831 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 309.547805 | 1.1057044999999999 | 1.23366015 | 1.25551914 | 1820.7446321170905 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 303.494048 | 1.020917 | 1.164189 | 1.20555011 | 1941.308120969432 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 306.733857 | 1.167779 | 1.25867675 | 1.27097576 | 3481.087876164615 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 307.316527 | 1.139578 | 1.28121725 | 1.3418339 | 3450.354641251446 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 306.386156 | 1.1655175 | 1.27963575 | 1.33065193 | 3408.2887025028344 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 304.414581 | 1.2048234999999998 | 7.115789799999996 | 16.060786269999973 | 1892.614581654722 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 302.738764 | 1.3351245 | 1.49985285 | 1.6295143699999999 | 5906.531065510887 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 304.803653 | 1.3211565 | 1.4834398 | 1.51301778 | 6083.680631649092 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 307.255749 | 1.3035065000000001 | 1.43682265 | 1.4555265 | 6155.744490743337 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 301.990009 | 1.3582655 | 1.5044515 | 1.53971813 | 5885.55260453948 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 308.531668 | 2.5087075 | 2.76978585 | 2.8411690000000003 | 6363.053009059532 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 307.699851 | 2.3998749999999998 | 2.7323765499999997 | 28.992953209999918 | 4636.2047499389655 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 305.107342 | 2.5052589999999997 | 13.66700485 | 19.754735869999976 | 4383.736705058118 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 302.85537 | 2.800886 | 3.03266085 | 3.12617163 | 5683.973328125678 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 304.251227 | 2.5905785000000003 | 2.81233245 | 2.98880796 | 12290.823341745352 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 302.170277 | 2.632976 | 2.89843705 | 2.97702956 | 12051.128600244541 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 307.731877 | 2.742121 | 3.0198639999999997 | 3.08045195 | 11665.98293069834 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 305.363029 | 2.737668 | 3.0960897000000003 | 3.2630287 | 11625.348048388883 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 311.682632 | 3.448094 | 3.77720135 | 3.83484465 | 18553.013057732343 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 304.979203 | 3.266966 | 3.5932939499999996 | 3.76086577 | 19451.554663075003 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 304.112681 | 3.149522 | 3.4461979 | 3.597628 | 20248.96212837246 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 307.424302 | 3.2037120000000003 | 3.4695108 | 3.5525322799999994 | 19829.25806655922 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 304.773217 | 3.5374220000000003 | 3.7739972 | 3.86546888 | 36452.22625051598 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 302.148356 | 3.4330755 | 3.8331865499999997 | 3.89043013 | 37173.97019593203 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 312.337577 | 3.3523875 | 3.56474215 | 3.8819581 | 38560.46877383481 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share0` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 303.652394 | 3.492931 | 3.7569369999999997 | 3.9885756299999993 | 36580.23813792184 | - |
