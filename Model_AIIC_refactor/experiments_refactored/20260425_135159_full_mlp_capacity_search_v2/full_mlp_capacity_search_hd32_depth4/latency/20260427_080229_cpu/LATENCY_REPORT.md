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

### full_mlp_capacity_search_hd32_depth4::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2009318.844` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.040` ms, throughput=`23601.538` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`3179560.197` samples/s, p50=`0.040` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.022` ms, throughput=`46607.227` samples/s

### full_mlp_capacity_search_hd32_depth4::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2104194.449` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25382.334` samples/s

### full_mlp_capacity_search_hd32_depth4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`4021363.494` samples/s, p50=`0.031` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.010` ms, throughput=`96052.252` samples/s

### full_mlp_capacity_search_hd32_depth4::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`398808.013` samples/s, p50=`0.319` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.182` ms, throughput=`5450.270` samples/s

## Run References

### full_mlp_capacity_search_hd32_depth4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260425_135159_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth4/MODEL_COMPLEXITY.md`
- Trainable parameters: `6,608`
- MACs / sample: `6,400`
- FLOPs / sample estimate: `13,080`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.0403505 | 0.04810349999999999 | 0.051800879999999994 | 23601.538065032622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.040651 | 0.046631099999999995 | 0.04774796 | 23684.884922249632 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.040397 | 0.0475236 | 0.08873500999999985 | 22922.877811433656 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.040837 | 0.048325349999999996 | 0.04989789 | 23548.7938072324 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.043659500000000004 | 0.049371450000000004 | 0.05065893999999999 | 44215.74098065207 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.0429975 | 0.04941684999999999 | 0.05152158 | 44944.931223019 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0471655 | 0.0559128 | 0.09700519999999985 | 40995.636424458986 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.043206 | 0.049389249999999996 | 0.05028008 | 44909.48717407501 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.0440015 | 0.053009850000000004 | 0.05430589999999999 | 86835.29212043533 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0447365 | 0.0524978 | 0.05342798 | 87534.86075829699 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0456495 | 0.0505095 | 0.050941349999999996 | 86566.44580680794 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.044393 | 0.05070735 | 0.054526399999999996 | 87868.55917969428 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.044445 | 0.05214815 | 0.06688443999999996 | 172450.14143067223 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.043819 | 0.051094799999999996 | 0.05269561999999999 | 175539.3006163185 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.046313 | 0.05376235 | 0.07571171999999991 | 167242.32869890903 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0438925 | 0.05240765 | 0.05672947999999999 | 176528.73887868944 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.046915 | 0.0537566 | 0.055476029999999996 | 331728.3211395531 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.045483 | 0.05222175 | 0.053783809999999994 | 337449.22970886144 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.0450105 | 0.050985499999999996 | 0.05350937 | 343617.22415697813 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.045947 | 0.05405315 | 0.05742909999999999 | 336310.6636124053 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.047683500000000004 | 0.0533126 | 0.054745129999999996 | 654564.0501150601 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0498165 | 0.05688335 | 0.07541092999999995 | 625870.5957779553 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.055363499999999996 | 0.06300214999999999 | 0.06595051999999998 | 566810.6838145793 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.047714 | 0.0542654 | 0.056820529999999994 | 645842.0286543962 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.0524645 | 0.0588488 | 0.06134894999999999 | 1175835.1185205053 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.06933249999999999 | 0.07807785 | 0.07960708 | 900423.8182359459 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.061846 | 0.06901725 | 0.07091788 | 1008623.0970906898 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.059913999999999995 | 0.0724089 | 0.11901160999999982 | 995419.8245323704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0609935 | 0.07152795 | 0.07387582999999999 | 2009318.8440514274 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.084579 | 0.09630364999999999 | 0.10032673 | 1492418.2819374015 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.1055825 | 0.11886424999999998 | 0.16224281999999984 | 1187115.566998855 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.0799745 | 0.08904815 | 0.13533976999999983 | 1550231.372032276 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.0526055 | 0.060461999999999995 | 0.06820438999999999 | 18463.067771644346 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.056509500000000004 | 0.06473695 | 0.06855858 | 17086.73843552456 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.0581735 | 0.06943829999999998 | 0.07538845 | 16625.535092846963 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.0571425 | 0.0612274 | 0.06543608999999999 | 17306.104139827785 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.058399 | 0.071244 | 0.11764075999999984 | 32002.478271917378 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.062141 | 0.07263389999999999 | 0.07453829 | 31539.125073131345 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.0624865 | 0.08263939999999996 | 0.1310878399999999 | 29412.61248118328 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.060922000000000004 | 0.0662987 | 0.07192235 | 32401.477636986157 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.060899499999999995 | 0.0710574 | 0.07305284 | 62701.33005196373 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.070035 | 0.09913389999999993 | 0.15306923999999988 | 53441.01341262554 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.073872 | 0.08149864999999999 | 0.08618044 | 53427.27986224311 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.06440199999999999 | 0.07628839999999999 | 0.0791148 | 60015.48999796848 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.08094950000000001 | 0.0975433 | 0.13821485999999986 | 92851.69611032639 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0887925 | 0.11101269999999999 | 0.2053522599999997 | 82191.03209539255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.08590300000000001 | 0.1011106 | 0.1045758 | 89572.4460824259 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.0858925 | 0.1013661 | 0.13272481999999988 | 89519.74447484137 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0878635 | 0.0941778 | 0.09887314 | 181667.56289331027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.0913725 | 0.10023245 | 0.10572699999999999 | 173140.83535258833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.0913765 | 0.10820674999999999 | 0.11697479999999998 | 168204.47608931322 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.098603 | 0.1207096 | 0.12562947 | 156095.21574017327 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.088924 | 0.10232275 | 0.11588316999999995 | 343663.20490424574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0992045 | 0.10792385 | 0.11143180999999999 | 319136.0985811408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.0960925 | 0.11313014999999998 | 0.12020344 | 327119.49409334647 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.09859000000000001 | 0.10368405 | 0.10660182 | 323596.12372293294 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.10220299999999999 | 0.11880839999999998 | 0.19500770999999983 | 593027.3695103503 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.1303295 | 0.15222839999999999 | 0.2151814999999998 | 475188.97300228704 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.127369 | 0.1494526 | 0.18673926999999985 | 483426.5515688475 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.129693 | 0.15766144999999998 | 0.16906873999999997 | 470528.4505027597 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1140455 | 0.1387569 | 0.14205843 | 1062707.19676937 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.1504945 | 0.17236914999999997 | 0.23968932999999987 | 830878.3474660547 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1645015 | 0.17955944999999998 | 0.18473479999999998 | 774479.8248610686 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.1578125 | 0.1809414 | 0.18840858 | 793847.4835406888 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 48.102084 | 0.021997000000000003 | 0.0225152 | 0.023826739999999996 | 45382.38745848646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 48.771113 | 0.021580000000000002 | 0.02202845 | 0.02469629999999999 | 46607.226916605694 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 48.418223 | 0.021811 | 0.02234065 | 0.029303299999999984 | 45286.98361516932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 47.7993 | 0.021645499999999998 | 0.023469749999999998 | 0.02461989 | 45292.6038083833 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 49.851911 | 0.023516500000000003 | 0.0241733 | 0.02701460999999999 | 85371.72556746585 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 49.621061 | 0.023309999999999997 | 0.0241047 | 0.02635431999999999 | 86149.93362147614 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 48.628112 | 0.023282 | 0.02362695 | 0.025585649999999995 | 86050.37388887454 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 49.535381 | 0.023954999999999997 | 0.027576049999999998 | 0.029023169999999997 | 82171.42932271022 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 48.553807 | 0.023573499999999997 | 0.026643 | 0.028588109999999996 | 165938.61101085652 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 49.657806 | 0.0235065 | 0.0239399 | 0.029361749999999985 | 170042.86780697416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 48.719753 | 0.02348 | 0.0240847 | 0.02686194999999999 | 170616.72828774172 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 48.380294 | 0.023122 | 0.0238907 | 0.02958722 | 173343.27168090967 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 50.100402 | 0.023747 | 0.0245983 | 0.02801351999999999 | 336741.4372966397 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 48.459111 | 0.023642999999999997 | 0.02558275 | 0.026977939999999995 | 331692.28574958723 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 49.409908 | 0.023547 | 0.025408999999999998 | 0.0264404 | 333559.3197724792 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 47.543319 | 0.023847 | 0.024398 | 0.02571693 | 338759.4122185439 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 50.035579 | 0.025395 | 0.02602175 | 0.029767019999999998 | 626075.5782786208 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 48.462553 | 0.025017 | 0.0289554 | 0.029721789999999998 | 624283.0499348405 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 48.586072 | 0.024359 | 0.02606805 | 0.03202102999999998 | 641936.5943177377 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 48.525309 | 0.0251045 | 0.0289838 | 0.02979576 | 618925.3444512956 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 49.728274 | 0.0273325 | 0.0280301 | 0.03251838 | 1167946.297829226 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 48.856455 | 0.027262 | 0.02804745 | 0.034428129999999994 | 1161411.521492646 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 50.698394 | 0.032935000000000006 | 0.0378367 | 0.04036409999999999 | 953320.0861563028 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 49.661473 | 0.0268205 | 0.028110299999999998 | 0.02830612 | 1185297.5689546862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 48.99043 | 0.030858999999999998 | 0.032347600000000004 | 0.03504197 | 2049869.4809666416 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 49.861194 | 0.043426 | 0.04478695 | 0.046503539999999996 | 1468120.9015913971 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 51.252855 | 0.0405115 | 0.04341094999999999 | 0.04885755 | 1571404.3689952097 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 49.450676 | 0.0383895 | 0.039881900000000005 | 0.043474109999999996 | 1660225.510506374 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 49.563977 | 0.040172 | 0.04324244999999999 | 0.046111009999999994 | 3179560.1972718383 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 50.479904 | 0.063588 | 0.06940815 | 0.07171817999999999 | 2007106.411136932 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 51.751811 | 0.0711335 | 0.07545295 | 0.07685368 | 1805753.9220128737 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 51.522188 | 0.054291 | 0.060030499999999994 | 0.061830119999999995 | 2325050.824157859 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 49.31299 | 0.028812499999999998 | 0.03082985 | 0.0329207 | 34256.007476031074 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 49.446187 | 0.0325755 | 0.0339637 | 0.03706017999999999 | 30484.087306426045 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 49.810733 | 0.0332065 | 0.038798349999999995 | 0.042878379999999994 | 29445.17632066033 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 50.896029 | 0.0336195 | 0.03465235 | 0.037208740000000004 | 29637.409998595183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 50.841048 | 0.0346195 | 0.0373603 | 0.038522339999999995 | 57455.11749858804 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 51.110561 | 0.038843 | 0.0405517 | 0.04159732 | 51413.30020945778 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 51.392564 | 0.039213 | 0.04292194999999999 | 0.046257809999999996 | 50809.083851247255 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 50.8961 | 0.0404655 | 0.0452524 | 0.049814719999999986 | 48836.065629811885 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 49.126938 | 0.0386205 | 0.04084815 | 0.042964409999999995 | 102745.7253925272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 49.959089 | 0.040537 | 0.04336955 | 0.048556419999999996 | 97571.35149006089 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 49.767245 | 0.043644 | 0.0451691 | 0.04606278 | 91550.94627058067 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 50.289025 | 0.043701000000000004 | 0.047103099999999995 | 0.05078405999999999 | 90713.27393730535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 50.079158 | 0.0580435 | 0.0599674 | 0.06101213 | 138904.3227025225 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 50.306936 | 0.0616425 | 0.06710980000000001 | 0.07441379999999999 | 127904.71610269212 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 51.429418 | 0.060907 | 0.0663859 | 0.06836376999999999 | 130158.50702986096 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 52.375732 | 0.0599995 | 0.06454305 | 0.07157514999999998 | 131577.216089262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 50.110529 | 0.061147 | 0.06708515 | 0.06983212999999999 | 260371.06782731408 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 52.291595 | 0.0653705 | 0.07544495000000001 | 0.07821565999999999 | 239552.94629163056 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 51.309394 | 0.0676355 | 0.07549515 | 0.08041952999999999 | 235216.49620331175 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 50.965157 | 0.06757450000000001 | 0.07679719999999998 | 0.08003135 | 234413.20660564696 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 51.936504 | 0.0657095 | 0.07242905 | 0.07691630999999999 | 481790.5746707638 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 51.012485 | 0.073353 | 0.0802859 | 0.08502564999999998 | 430876.8505487486 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 50.675897 | 0.0749155 | 0.08411505 | 0.08541324 | 419673.8924019091 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 53.362962 | 0.07587150000000001 | 0.08552585 | 0.08867955 | 417924.46694387245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 50.914812 | 0.075183 | 0.08356065 | 0.08648533 | 837305.9576150491 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 51.8669 | 0.09231500000000001 | 0.10069719999999999 | 0.10371095999999999 | 686884.6881597178 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 51.794814 | 0.098167 | 0.104589 | 0.10939251 | 649682.082133621 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 52.881392 | 0.09790850000000001 | 0.1060359 | 0.10799055 | 651699.969553392 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 53.59304 | 0.09075749999999999 | 0.09954765 | 0.10473345999999999 | 1380548.6731848698 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 54.14875 | 0.1173115 | 0.12613895 | 0.13086379 | 1084858.6615624067 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 53.812703 | 0.1324605 | 0.1413927 | 0.14339375 | 961405.4305286182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 58.195964 | 0.128083 | 0.13629825 | 0.1398937 | 1003939.3639448042 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 495.647111 | 0.041837 | 0.04493965 | 0.04858333 | 23693.908817309064 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 505.148392 | 0.038824 | 0.042129 | 0.04580406 | 25382.334098526102 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 504.584252 | 0.038945 | 0.04206705 | 0.04286263 | 25424.50016703897 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 545.560638 | 0.039367 | 0.042198099999999995 | 0.04258019 | 25205.373382319136 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 496.528817 | 0.0402985 | 0.045207899999999995 | 0.047655789999999996 | 48459.14457979622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 503.841132 | 0.039952 | 0.0454843 | 0.04656539 | 49181.30343241235 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 517.645006 | 0.0400045 | 0.04458155 | 0.045990689999999994 | 48926.766904564916 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 538.977849 | 0.039409 | 0.044067049999999997 | 0.05032703999999998 | 50017.93142841709 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.605833 | 0.0399245 | 0.04469375 | 0.047312440000000004 | 99393.55025313052 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 503.35416 | 0.0402585 | 0.0453564 | 0.047891539999999996 | 97432.0800969644 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 520.702359 | 0.0409705 | 0.048140249999999996 | 0.052985289999999984 | 95019.68570338559 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 539.529771 | 0.0402525 | 0.045859899999999995 | 0.04759098999999999 | 97399.05561875673 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 490.918548 | 0.041263499999999995 | 0.047720399999999996 | 0.050065029999999996 | 189942.2717950447 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.684435 | 0.0401285 | 0.047261 | 0.04799344 | 191963.63440909755 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 504.199394 | 0.043605 | 0.049485499999999995 | 0.05785291999999999 | 178594.47044730323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 539.536207 | 0.042275 | 0.048044949999999996 | 0.04896532 | 186542.54125971245 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 495.899974 | 0.0416025 | 0.0503088 | 0.052449159999999995 | 372084.3701309272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 514.017805 | 0.042473 | 0.045080049999999997 | 0.04580615 | 374540.9532441801 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 506.300064 | 0.0420165 | 0.04974385 | 0.05265517999999999 | 370871.90594317595 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 533.6512 | 0.0418025 | 0.04592379999999999 | 0.049498 | 379417.90653324436 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 499.3518 | 0.0454525 | 0.0474831 | 0.048143729999999996 | 705598.6607737419 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 508.640636 | 0.047495499999999996 | 0.05589635 | 0.05721405 | 656319.2468080323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 507.193592 | 0.0524595 | 0.057945149999999994 | 0.05943893 | 601862.8407248986 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 540.401713 | 0.0476165 | 0.0552977 | 0.056138959999999995 | 656875.6612377261 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 493.487884 | 0.0477705 | 0.051066549999999995 | 0.057530929999999994 | 1322479.6327804679 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 505.245658 | 0.07717850000000001 | 0.0877491 | 0.08863491 | 819089.4949980507 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 504.97358 | 0.061324500000000004 | 0.06581065 | 0.06978987999999998 | 1031353.800119766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 540.405646 | 0.058517 | 0.06385245 | 0.0685404 | 1076616.034311753 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 492.677095 | 0.060646 | 0.0642213 | 0.0683593 | 2104194.448608996 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 502.48988 | 0.10552049999999999 | 0.11202654999999999 | 0.11347284 | 1207857.1105038272 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 504.458315 | 0.121811 | 0.12961345 | 0.13014578999999998 | 1046394.6879773664 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 536.451587 | 0.077262 | 0.08360269999999999 | 0.08696559 | 1648246.651870219 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 490.936519 | 0.0513365 | 0.05700905 | 0.06102524999999999 | 19116.094719484692 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 512.910931 | 0.06343 | 0.0693395 | 0.07423872000000001 | 15587.682114787069 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 505.41682 | 0.057750499999999996 | 0.06400795 | 0.06655694999999999 | 17123.944480062248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.20277 | 0.0576575 | 0.06368639999999999 | 0.06423330000000001 | 17049.83906656905 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 493.541481 | 0.0581415 | 0.07642164999999997 | 0.17980732999999977 | 31358.23490767351 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 505.652468 | 0.06346299999999999 | 0.069865 | 0.07260278 | 31215.45814459687 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 508.923941 | 0.0610525 | 0.06698275 | 0.068552 | 32513.917582421156 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 505.638107 | 0.060903 | 0.0673129 | 0.07393921999999999 | 32417.453816474423 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 480.971894 | 0.060974 | 0.066284 | 0.06898828 | 65373.192676371975 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 511.208734 | 0.069629 | 0.07712554999999999 | 0.07908356999999999 | 56984.700177849256 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 496.103788 | 0.063227 | 0.0693501 | 0.07205588999999998 | 62612.74206868743 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 502.204457 | 0.0719785 | 0.08284529999999998 | 0.0880937 | 54703.90689832677 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.552736 | 0.0892745 | 0.09638155 | 0.10093546999999999 | 89033.16218191788 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 504.319841 | 0.08076900000000001 | 0.08717745 | 0.08986883999999999 | 98410.52244670209 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 507.663855 | 0.0948 | 0.10061149999999999 | 0.10206045 | 83985.69219747724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 500.142089 | 0.0866195 | 0.09740684999999999 | 0.10022069 | 91604.95546167066 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.785716 | 0.077071 | 0.08433009999999999 | 0.08780629999999999 | 203925.51518595073 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 506.010502 | 0.084726 | 0.095349 | 0.09944956999999999 | 184407.38726773026 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 506.230868 | 0.08421500000000001 | 0.08964219999999999 | 0.09531411999999999 | 189478.40621186007 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.341127 | 0.09016099999999999 | 0.1027937 | 0.10755945 | 174195.27769666808 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.457523 | 0.08534 | 0.08955965 | 0.09257102999999998 | 373651.7593626622 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 499.108962 | 0.0845765 | 0.0907664 | 0.09583704 | 374929.58353759185 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 509.031074 | 0.09043899999999999 | 0.10081915 | 0.10508804999999999 | 348482.977368862 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 509.27681 | 0.098826 | 0.10812585 | 0.11319205999999998 | 321460.5237435766 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 489.834685 | 0.0828115 | 0.08889994999999999 | 0.09119864 | 772088.6106796262 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 498.07691 | 0.108133 | 0.1167033 | 0.12085873 | 590954.2267170036 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 490.905792 | 0.105184 | 0.11258515 | 0.11481055999999999 | 604585.2120346465 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 500.093758 | 0.1177605 | 0.1275631 | 0.12912897 | 541916.6610075891 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 493.100373 | 0.0987645 | 0.1040186 | 0.10717106999999998 | 1287224.677223384 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 508.232713 | 0.1366525 | 0.14743474999999998 | 0.15539118 | 939528.7030422819 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 503.068324 | 0.135241 | 0.14382305 | 0.1498998 | 957324.7082365974 | - |
| `full_mlp_capacity_search_hd32_depth4` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.738532 | 0.1393025 | 0.1518909 | 0.15606848 | 915441.1317713143 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.29074 | 0.0103775 | 0.011035999999999999 | 0.014418549999999988 | 94685.49266755546 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 3.486064 | 0.010465 | 0.011123049999999999 | 0.016184609999999995 | 93618.40762578101 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 3.763499 | 0.010273500000000001 | 0.01077675 | 0.013920599999999991 | 96052.25242531937 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.047628 | 0.010322000000000001 | 0.0108116 | 0.014546719999999987 | 95567.39316998953 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.390879 | 0.010548 | 0.012014599999999995 | 0.016066589999999995 | 185276.45099252596 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.582508 | 0.010696 | 0.01130965 | 0.014826569999999987 | 183927.32663470006 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 3.663524 | 0.01066 | 0.011178849999999999 | 0.011943029999999999 | 186547.326123901 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.026537 | 0.010526 | 0.0108802 | 0.014869569999999985 | 186856.86203797304 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.431295 | 0.011051 | 0.01153275 | 0.014746049999999993 | 358790.0880650271 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 3.508786 | 0.011498000000000001 | 0.012495099999999999 | 0.019340989999999992 | 339332.16037516564 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 3.661649 | 0.011428 | 0.01179435 | 0.012025409999999999 | 349701.7044461075 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.356004 | 0.0116475 | 0.01197785 | 0.015654809999999984 | 340100.7718587017 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.287688 | 0.0121345 | 0.01240855 | 0.01245861 | 662741.0306285768 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.493299 | 0.0127045 | 0.013715199999999999 | 0.019420269999999996 | 614087.4736901899 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 3.626948 | 0.012584 | 0.013158749999999999 | 0.017256669999999988 | 628507.859490783 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.477865 | 0.0129385 | 0.01342145 | 0.01669665999999999 | 609925.3146452217 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.426756 | 0.0137385 | 0.015430149999999998 | 0.018889369999999996 | 1146843.2422978724 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.550511 | 0.0160195 | 0.0218531 | 0.03324523999999999 | 947355.4572411113 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 3.599769 | 0.016420999999999998 | 0.021221249999999997 | 0.022568139999999997 | 952705.3258609478 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.373411 | 0.016082 | 0.02250145 | 0.03176208999999997 | 942214.0144912514 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.484098 | 0.0168465 | 0.0173511 | 0.02021280999999999 | 1889131.5898223037 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.397773 | 0.029278 | 0.0312451 | 0.03543615 | 1150653.643185182 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 3.865501 | 0.0280575 | 0.02932785 | 0.02952842 | 1254329.3963069406 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.434673 | 0.0280165 | 0.029834 | 0.03461107999999998 | 1226248.148173695 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.392747 | 0.021894999999999998 | 0.023024999999999997 | 0.030182259999999975 | 2878663.4365664027 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.598789 | 0.0442575 | 0.049770849999999985 | 0.053061399999999995 | 1438574.0135091092 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 3.677888 | 0.0407015 | 0.045802499999999996 | 0.05208282999999998 | 1560856.8323581233 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.001496 | 0.0400875 | 0.04474324999999999 | 0.05011811 | 1600278.4484500303 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.376253 | 0.03128 | 0.03517595 | 0.03826095999999999 | 4021363.493559535 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.492579 | 0.0851155 | 0.09231365000000001 | 0.09524674999999999 | 1610363.8994659882 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 3.702668 | 0.06943550000000001 | 0.08253415 | 0.08769779999999999 | 1825608.7264097123 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.184325 | 0.0784695 | 0.08723055 | 0.08935936 | 1668692.563732323 | - |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `onnxruntime` | `onnxruntime` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | ONNX Runtime benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 225.594685 | 0.18169 | 0.23093934999999996 | 0.28848630999999997 | 5450.2696684426555 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 219.253971 | 0.187081 | 0.2518664 | 0.2951278799999999 | 5210.617863602906 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 218.086992 | 0.195693 | 0.3017182 | 0.4089632599999999 | 4716.8827895192 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 210.393116 | 0.21004299999999998 | 0.4598661999999997 | 0.7299873899999992 | 4076.691370329253 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 216.861957 | 0.228961 | 0.32839265 | 0.3650046199999999 | 8417.147783819699 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 226.634553 | 0.193639 | 0.5070430999999999 | 1.1073310499999998 | 8195.369075577939 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 218.358345 | 0.24349500000000002 | 0.35057999999999995 | 0.36720327999999997 | 8079.879629185237 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 218.450477 | 0.210107 | 0.24857610000000002 | 0.27554301999999997 | 9482.997412374494 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 216.856255 | 0.227629 | 0.2844059 | 0.3107414 | 17605.075895482183 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 227.163596 | 0.24191849999999998 | 0.33168274999999997 | 0.873066039999998 | 14918.572937052275 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 212.587294 | 0.2441835 | 0.3228797999999999 | 0.4456227199999999 | 15995.642786904846 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 216.058151 | 0.2338445 | 0.4375517 | 0.48367047999999985 | 15651.019820451502 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 222.450347 | 0.21848800000000002 | 0.31862315 | 0.3466543899999999 | 35505.36832292701 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 219.126329 | 0.2449865 | 0.4796955999999998 | 0.6099228699999997 | 29600.369116602884 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 212.059037 | 0.468681 | 0.8056601999999998 | 1.0229261299999999 | 16966.916760433625 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 217.879824 | 0.2434045 | 0.3009243 | 0.34436744999999985 | 32542.525555441927 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 219.822401 | 0.26133300000000004 | 0.4993213499999999 | 0.7276089099999999 | 54594.66294858814 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 220.690146 | 0.221823 | 0.32378609999999997 | 0.5891686999999999 | 65898.74312150979 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 218.9517 | 0.3306095 | 1.047487949999998 | 15.014668529999998 | 16529.993580370367 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 219.210309 | 0.258401 | 0.6887976999999998 | 0.79787055 | 48765.2991982741 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 219.279109 | 0.344997 | 0.5871898 | 1.9840475299999967 | 75460.70647822619 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 217.123217 | 0.3063015 | 0.4342535999999999 | 0.47593294999999997 | 101999.09297306574 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 217.483501 | 0.28550450000000005 | 0.3690568 | 0.46576979999999996 | 109810.71583752599 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 219.221383 | 0.3134415 | 0.41209134999999997 | 0.4836724 | 99199.53415898759 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 213.598627 | 0.295379 | 0.39212389999999997 | 0.44152686999999996 | 214110.33975630897 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 221.048362 | 0.317836 | 0.43381909999999985 | 0.5017720299999999 | 196088.12764209602 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 217.428892 | 0.285795 | 0.37098369999999997 | 0.5186471299999997 | 215162.00959703248 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 222.08073 | 0.25418050000000003 | 0.38814339999999997 | 0.46372708999999995 | 240598.5188905551 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 220.925311 | 0.6164835 | 1.4366878999999997 | 1.59155723 | 180751.6393114594 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 217.965108 | 0.318879 | 0.39000314999999997 | 0.46034407999999993 | 398808.0127010382 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 217.099159 | 0.3244095 | 0.41348935000000003 | 0.42918875 | 385390.4966196135 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 217.579032 | 0.349622 | 0.49189979999999994 | 0.7202884699999996 | 341308.5406809383 | - |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 1 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 2 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 4 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 8 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 16 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 32 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 64 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 1 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 2 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 4 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
| `full_mlp_capacity_search_hd32_depth4` | `openvino` | `openvino` | `bf16` | 128 | 8 | skipped | 0.0 | - | - | - | - | OpenVINO benchmark currently supports precision profile fp32 only |
