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

### full_mlp_capacity_search_hd64_depth3::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2008283.542` samples/s, p50=`0.062` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.035` ms, throughput=`27233.753` samples/s

### full_mlp_capacity_search_hd64_depth3::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2969592.303` samples/s, p50=`0.043` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.020` ms, throughput=`50081.784` samples/s

### full_mlp_capacity_search_hd64_depth3::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`2117659.126` samples/s, p50=`0.061` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.033` ms, throughput=`29950.898` samples/s

## Run References

### full_mlp_capacity_search_hd64_depth3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd64_depth3/MODEL_COMPLEXITY.md`
- Trainable parameters: `10,960`
- MACs / sample: `10,752`
- FLOPs / sample estimate: `21,784`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.035035 | 0.0404524 | 0.044531709999999995 | 27556.969901726334 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.0397655 | 0.0447146 | 0.04840417999999999 | 24483.208436326026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.035206 | 0.0398056 | 0.043479130000000005 | 27435.33902122782 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.035004 | 0.041437299999999996 | 0.04563623 | 27233.753295964998 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0371135 | 0.03954465 | 0.04007942 | 26765.950499051145 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.037362 | 0.04501895 | 0.06196248999999994 | 50803.25018873407 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.037004999999999996 | 0.0445741 | 0.04511321 | 51723.5047639934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0370925 | 0.04388804999999999 | 0.06298760999999994 | 51009.68571912435 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.036866 | 0.0414547 | 0.04530434999999999 | 52349.771859694236 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.036377 | 0.03867125 | 0.045478699999999976 | 54284.36648244363 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.038014 | 0.0454257 | 0.05203951999999998 | 100422.32609238151 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.037504499999999996 | 0.043316749999999994 | 0.04595518 | 102598.67042382999 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.041688 | 0.047265499999999995 | 0.05076081999999999 | 95887.57373754421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.039998000000000006 | 0.0437853 | 0.08021506999999989 | 96926.74373635149 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.037313 | 0.03924529999999999 | 0.04450546 | 106289.29714607923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0391025 | 0.044690099999999996 | 0.047521669999999995 | 198785.5198663764 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.0390505 | 0.045480150000000004 | 0.05035714999999999 | 195354.85232149938 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.039858 | 0.045526849999999994 | 0.05134475 | 194518.9424977691 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0389315 | 0.0451813 | 0.051808389999999996 | 196180.17579705553 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.0383205 | 0.03990315 | 0.04531284999999999 | 206880.85731427273 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.039945999999999995 | 0.04632764999999999 | 0.04862631 | 383279.25103401556 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.039941000000000004 | 0.0458234 | 0.049259779999999996 | 390093.5736959903 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.047479 | 0.05579185 | 0.08958338999999987 | 318539.05248966144 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.039303500000000005 | 0.040836599999999994 | 0.049704969999999994 | 401677.4048426228 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.0419205 | 0.04701585 | 0.047899989999999996 | 380436.07485079777 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0449585 | 0.0505552 | 0.05144554 | 696435.8156046891 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.0539085 | 0.06339284999999999 | 0.08202974999999996 | 569531.4215824645 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.0516855 | 0.05941345 | 0.06123195 | 608866.465689994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.048999 | 0.058427099999999996 | 0.10043402999999992 | 614254.6231298347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.1232245 | 0.14524905 | 0.15541096 | 256927.4051616716 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.050563 | 0.057574549999999995 | 0.06236772999999999 | 1236576.8615601966 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.0657585 | 0.0745204 | 0.07758783999999999 | 971768.0134662752 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.062783 | 0.073736 | 0.1018240099999999 | 968026.0979836015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.058610999999999996 | 0.0665003 | 0.06828457 | 1072089.2943173237 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.153272 | 0.16973985 | 0.17605673 | 410236.9502985115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.0621835 | 0.07136275 | 0.07358421 | 2008283.5420222348 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.111563 | 0.1207818 | 0.12287358999999999 | 1174433.6798218533 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.09388350000000001 | 0.11321454999999998 | 0.16978421 | 1313828.9944815077 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.077488 | 0.08675949999999999 | 0.11977000999999989 | 1609134.2505733797 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.2874545 | 0.3259162 | 0.33865172 | 440438.68518867326 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.066708 | 0.0788884 | 0.09690235999999994 | 14162.881636549298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.0665275 | 0.07537214999999998 | 0.11700924999999986 | 14492.421767733433 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.065079 | 0.0786513 | 0.1250454299999999 | 14402.170234228253 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.065198 | 0.07109644999999999 | 0.07292474 | 15216.72110702255 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.14439950000000001 | 0.15804435 | 0.16067632999999998 | 6852.465325154962 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.065413 | 0.07898695 | 0.08227901 | 29081.28160044762 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.067452 | 0.08239455 | 0.08566295 | 27882.219926586116 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.07617 | 0.09155545 | 0.0933909 | 25139.537000119162 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.06995950000000001 | 0.0840853 | 0.11526999999999993 | 26931.311689535843 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.141689 | 6.0092669 | 7.213320309999997 | 2239.6660353346733 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.0742495 | 0.08767965 | 0.09126422 | 52097.99946286964 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.0892085 | 0.10264039999999999 | 0.10838774999999999 | 43670.51556755621 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.085187 | 0.10475425000000001 | 0.12688084999999993 | 44790.60615575217 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.08307700000000001 | 0.11039894999999998 | 0.1433567499999999 | 44891.78610500458 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.2204305 | 0.24534925 | 0.24871837 | 18197.174943181595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.0861445 | 0.10080019999999999 | 0.17785965999999986 | 88114.46068442908 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.0796055 | 0.09181465 | 0.09640562999999998 | 97144.84028538241 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.0851195 | 0.10144289999999999 | 0.1336786199999999 | 90503.9622634679 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.086426 | 0.10205615 | 0.14452908999999983 | 88014.28119726708 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.219622 | 0.24312069999999997 | 0.25028924 | 36513.00109301669 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.0786485 | 0.09151615 | 0.09823276 | 195679.92536767648 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.080917 | 0.09867125 | 0.14331346999999983 | 186565.2497968771 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.08503949999999999 | 0.0990973 | 0.1515045699999998 | 179728.7533654209 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.0914585 | 0.10772549999999999 | 0.13745583999999988 | 169842.90592719897 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 5.9957484999999995 | 6.0058941 | 6.527140789999998 | 4299.285148185019 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.07704949999999999 | 0.09205885 | 0.16826839999999987 | 386865.90260650904 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.0861945 | 0.10076414999999998 | 0.10747477999999999 | 362201.71556842583 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.09203549999999999 | 0.1006854 | 0.10722629 | 343566.38658132765 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.087083 | 0.1020978 | 0.10912672999999998 | 348586.99173565605 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.20578649999999998 | 0.3156713 | 0.34294863 | 144824.25712171022 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.086163 | 0.10848574999999998 | 0.1793108299999999 | 707356.932086882 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.108035 | 0.1277828 | 0.12935908000000002 | 574623.2715960485 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.168719 | 0.1817555 | 0.18955579 | 412304.7720798546 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.110391 | 0.13012215 | 0.13222982 | 561158.4695735493 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.253045 | 0.26633445 | 0.27100667 | 252643.6393568893 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.1004525 | 0.11962529999999999 | 0.12418341999999999 | 1229110.1684092183 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.116867 | 0.13251924999999998 | 0.20430336 | 1057051.7241834938 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.1273435 | 0.1418111 | 0.1717199599999999 | 983937.527346545 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.12699549999999998 | 0.1361746 | 0.14669477 | 1008470.2042750942 | - |
| `full_mlp_capacity_search_hd64_depth3` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.3246875 | 0.33813145 | 0.34182784 | 395068.9713262028 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 1 | ok | 46.696532 | 0.0203025 | 0.0211773 | 0.024930929999999997 | 48865.68094815058 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 2 | ok | 46.867483 | 0.0201005 | 0.021718749999999995 | 0.0245584 | 49261.42347779738 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 4 | ok | 46.14145 | 0.0198955 | 0.0205768 | 0.026476269999999993 | 50081.783552541296 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 8 | ok | 44.38291 | 0.020143 | 0.021590099999999997 | 0.025358829999999995 | 49119.773656082994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 1 | 128 | ok | 48.956393 | 0.020081500000000002 | 0.02186045 | 0.022535219999999998 | 48795.72159113089 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 1 | ok | 46.586214 | 0.021455000000000002 | 0.02207385 | 0.02230152 | 93398.23888280762 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 2 | ok | 46.784617 | 0.021481 | 0.02187805 | 0.02379278999999999 | 93011.31575667496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 4 | ok | 45.111529 | 0.021564 | 0.0224029 | 0.030349649999999992 | 91370.01108318235 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 8 | ok | 45.877907 | 0.021388 | 0.022012749999999998 | 0.02265567 | 93425.20144809064 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 2 | 128 | ok | 48.964023 | 0.0212865 | 0.02163165 | 0.024156719999999993 | 93840.49742970878 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 1 | ok | 46.801166 | 0.0219375 | 0.025655 | 0.02719094 | 176348.02635697598 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 2 | ok | 45.33787 | 0.0208195 | 0.02243285 | 0.025442179999999995 | 188239.90045874062 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 4 | ok | 45.01604 | 0.0217885 | 0.02227695 | 0.023701569999999995 | 183574.82601695863 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 8 | ok | 45.333254 | 0.0219735 | 0.022599849999999998 | 0.025779769999999987 | 182498.23661078874 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 4 | 128 | ok | 51.948598 | 0.0214745 | 0.0226309 | 0.024687639999999993 | 186996.62845078905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 1 | ok | 46.708465 | 0.0225755 | 0.02298025 | 0.024445249999999995 | 357776.9529031364 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 2 | ok | 45.416733 | 0.0223165 | 0.028353749999999997 | 0.03173538999999999 | 331663.68307545094 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 4 | ok | 45.257192 | 0.022183 | 0.023877799999999998 | 0.025998899999999995 | 360873.7113425314 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 8 | ok | 46.882442 | 0.0223955 | 0.022780250000000002 | 0.027143339999999985 | 355171.65446060075 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 8 | 128 | ok | 49.302245 | 0.0217045 | 0.02269555 | 0.02345299 | 367268.6505907057 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 1 | ok | 46.886796 | 0.0240305 | 0.02473015 | 0.02758312999999999 | 663447.8389015958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 2 | ok | 45.813913 | 0.0240435 | 0.02442115 | 0.031350159999999974 | 658884.146753266 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 4 | ok | 45.925686 | 0.028727500000000003 | 0.032911149999999986 | 0.03551923 | 549927.5814116228 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 8 | ok | 46.09033 | 0.023743 | 0.02456665 | 0.02476291 | 678241.4217296682 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 16 | 128 | ok | 53.399021 | 0.024159 | 0.024779 | 0.027814839999999993 | 661284.3630219704 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 1 | ok | 45.9974 | 0.026742 | 0.033154499999999996 | 0.035754209999999995 | 1126833.569498869 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 2 | ok | 46.609271 | 0.034012 | 0.036118149999999995 | 0.03815232999999999 | 938134.706762544 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 4 | ok | 47.222066 | 0.0343095 | 0.0353376 | 0.03926846 | 927152.4713828595 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 8 | ok | 46.123384 | 0.033094 | 0.03725955 | 0.03834723 | 954856.1897060536 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 32 | 128 | ok | 74.826246 | 0.11752599999999999 | 0.14240249999999996 | 0.14964833 | 270190.8104390246 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 1 | ok | 48.960384 | 0.0311525 | 0.035287449999999984 | 0.03803035 | 2012708.9993880107 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 2 | ok | 48.250256 | 0.046815999999999997 | 0.05021335 | 0.05346027 | 1361168.6653868847 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 4 | ok | 47.77065 | 0.043183 | 0.04495645 | 0.04646822999999999 | 1479461.6054285143 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 8 | ok | 47.84061 | 0.042435 | 0.04493164999999999 | 0.05201493999999999 | 1498233.9567235124 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 64 | 128 | ok | 92.501717 | 0.135054 | 0.15767165 | 0.16442115 | 470413.55967085756 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 1 | ok | 46.990454 | 0.0432965 | 0.044666599999999994 | 0.04689499999999999 | 2969592.302816751 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 2 | ok | 48.642612 | 0.07142699999999999 | 0.07605355 | 0.07795592 | 1794617.7730531974 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 4 | ok | 49.206687 | 0.07078699999999999 | 0.07411645 | 0.07442999 | 1808297.3157239081 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 8 | ok | 48.114698 | 0.0548745 | 0.05893805 | 0.06394146999999999 | 2338278.792980487 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `fp32` | 128 | 128 | ok | 156.172239 | 0.187664 | 0.20081249999999998 | 0.20686948 | 678621.503415958 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 1 | ok | 46.774563 | 0.039091 | 0.041487649999999994 | 0.046849719999999984 | 25500.27718801303 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 2 | ok | 47.07123 | 0.0409285 | 0.044323549999999996 | 0.049082859999999985 | 24057.89291350706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 4 | ok | 45.9914 | 0.042994000000000004 | 0.0452076 | 0.053655029999999986 | 23030.118789352717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 8 | ok | 45.61317 | 0.042583 | 0.04731985 | 0.051284359999999994 | 23170.718143969865 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 1 | 128 | ok | 48.724492 | 0.113255 | 0.1302693 | 0.14610406999999995 | 8715.531163689006 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 1 | ok | 45.313626 | 0.04262 | 0.04544735 | 0.04965857999999999 | 46446.537944731404 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 2 | ok | 46.979716 | 0.046047500000000005 | 0.05046825 | 0.055920779999999996 | 42504.68189071026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 4 | ok | 45.530021 | 0.044851 | 0.05184755 | 0.05514376999999999 | 43921.119426355035 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 8 | ok | 50.928382 | 0.0518315 | 0.05509345 | 0.05765670999999999 | 38376.03370643793 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 2 | 128 | ok | 49.753058 | 0.1408025 | 0.20374509999999976 | 6.559534019999997 | 5006.814023545344 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 1 | ok | 45.413947 | 0.0529945 | 0.05544885 | 0.06150643999999998 | 75275.95223138624 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 2 | ok | 46.474182 | 0.0529975 | 0.059049349999999994 | 0.06204778 | 74291.25216791162 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 4 | ok | 47.14254 | 0.055354 | 0.06041695 | 0.06730324999999998 | 71670.04829844554 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 8 | ok | 45.550358 | 0.0539605 | 0.06213584999999999 | 0.06686199999999999 | 72462.56040660192 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 4 | 128 | ok | 47.255493 | 0.140922 | 0.1673396 | 0.16938299999999998 | 27761.595081311632 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 1 | ok | 46.318949 | 0.0525855 | 0.05883505 | 0.06396963999999998 | 148724.48306158272 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 2 | ok | 45.299142 | 0.0591785 | 0.0635241 | 0.07074625 | 133821.73807004298 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 4 | ok | 46.790301 | 0.060263 | 0.06376949999999999 | 0.07457104999999999 | 132958.25808726915 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 8 | ok | 45.071359 | 0.0592645 | 0.06586239999999999 | 0.07319317 | 133366.8084022423 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 8 | 128 | ok | 47.915661 | 0.159686 | 2.508217249999988 | 7.020952309999995 | 15427.221216131706 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 1 | ok | 45.493566 | 0.056416499999999994 | 0.059404849999999995 | 0.06468367999999998 | 286857.6182211959 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 2 | ok | 45.164312 | 0.062619 | 0.06989205 | 0.07477335999999998 | 252819.3305660151 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 4 | ok | 47.378831 | 0.06503 | 0.07039305 | 0.07953261999999998 | 243917.75824945106 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 8 | ok | 46.634568 | 0.07251099999999999 | 0.07854405 | 0.08256207 | 220452.41796345505 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 16 | 128 | ok | 48.275017 | 0.1283665 | 0.1409762 | 0.18972281999999985 | 123683.71538459994 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 1 | ok | 46.912847 | 0.057766 | 0.06200184999999999 | 0.06665529999999999 | 559699.2735803053 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 2 | ok | 46.135252 | 0.0651805 | 0.0706395 | 0.07237001 | 485741.21833005204 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 4 | ok | 46.013387 | 0.0696965 | 0.076419 | 0.08240962999999998 | 454287.4657261402 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 8 | ok | 47.753124 | 0.068599 | 0.07601895 | 0.07739427 | 462305.4885774427 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 32 | 128 | ok | 98.281567 | 0.18700850000000002 | 0.20676725 | 0.22451875 | 169438.14522839946 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 1 | ok | 47.864484 | 0.060025499999999996 | 0.0704109 | 0.07461721999999998 | 1053524.3021718403 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 2 | ok | 47.533111 | 0.0786365 | 0.08405209999999999 | 0.08821232999999998 | 820559.529286026 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 4 | ok | 46.441617 | 0.0812605 | 0.08767475 | 0.0901294 | 788468.8403274214 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 8 | ok | 46.903117 | 0.08562349999999999 | 0.09402139999999999 | 0.09900008999999999 | 742526.5859323698 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 64 | 128 | ok | 79.657146 | 0.20159749999999999 | 0.22356865 | 0.22871639 | 314306.7110965395 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 1 | ok | 47.823963 | 0.065762 | 0.0726616 | 0.07566534999999999 | 1913289.138347247 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 2 | ok | 48.787498 | 0.08743200000000001 | 0.09817379999999999 | 0.10060842999999998 | 1440034.8128416005 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 4 | ok | 48.364496 | 0.095142 | 0.1053641 | 0.1066059 | 1333715.1092833655 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 8 | ok | 47.751404 | 0.09742200000000001 | 0.1051958 | 0.10894613 | 1312267.8798036438 | - |
| `full_mlp_capacity_search_hd64_depth3` | `jit` | `bf16` | 128 | 128 | ok | 95.352822 | 0.3728165 | 0.4079074 | 0.41644505 | 341147.31252675876 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 1 | ok | 509.290867 | 0.033242 | 0.03538965 | 0.03635809 | 30040.855563566452 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 2 | ok | 507.292502 | 0.038163 | 0.0422153 | 0.04695639999999999 | 25718.153724605894 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 4 | ok | 509.972927 | 0.0329405 | 0.03716729999999999 | 0.04093165999999999 | 29950.898497004015 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 8 | ok | 540.340404 | 0.0338655 | 0.03677 | 0.039304769999999996 | 29216.410039459683 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 1 | 128 | ok | 711.50949 | 0.035414 | 0.03826265 | 0.040509899999999995 | 27946.79377613724 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 1 | ok | 493.292822 | 0.036949 | 0.041937949999999995 | 0.04583186999999999 | 53622.954485372466 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 2 | ok | 506.934358 | 0.0337895 | 0.04122675 | 0.044192169999999996 | 56765.99580612824 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 4 | ok | 517.129758 | 0.0337335 | 0.0399811 | 0.04302457999999999 | 56212.966195208515 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 8 | ok | 537.198725 | 0.0338955 | 0.03973275 | 0.03986444 | 57624.79802508293 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 2 | 128 | ok | 835.886814 | 0.03352 | 0.035537549999999994 | 0.041842889999999994 | 59190.09015834534 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 1 | ok | 501.154814 | 0.0370585 | 0.0415431 | 0.04333541 | 106423.73675024479 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 2 | ok | 511.391895 | 0.033973 | 0.03711079999999999 | 0.04138454 | 116560.50918292832 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 4 | ok | 516.908333 | 0.037336499999999995 | 0.04145565 | 0.043143089999999995 | 104927.61044155638 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 8 | ok | 546.233301 | 0.035873 | 0.0395664 | 0.043686809999999986 | 109967.57056344084 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 4 | 128 | ok | 836.954929 | 0.035568 | 0.03969175 | 0.04313376999999999 | 109649.00257784805 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 1 | ok | 494.126149 | 0.038075 | 0.04359024999999999 | 0.04804183999999999 | 203957.5931372349 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 2 | ok | 518.803605 | 0.0378985 | 0.04085095 | 0.047233769999999994 | 209223.95174877223 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 4 | ok | 509.673913 | 0.036282499999999995 | 0.0423268 | 0.046344969999999985 | 213201.2054396156 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 8 | ok | 538.295323 | 0.035931000000000005 | 0.0416121 | 0.042547209999999995 | 214254.3413285912 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 8 | 128 | ok | 698.267883 | 0.0345545 | 0.03701195 | 0.04509460999999997 | 228256.825449723 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 1 | ok | 494.355003 | 0.036768499999999996 | 0.04184789999999999 | 0.04610280999999999 | 429382.885549602 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 2 | ok | 507.771823 | 0.0388525 | 0.04377585 | 0.04497792 | 406000.2781101905 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 4 | ok | 512.036457 | 0.045357999999999996 | 0.0498893 | 0.053100249999999995 | 350934.4507086244 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 8 | ok | 535.871196 | 0.036906499999999995 | 0.04173745 | 0.0434949 | 426093.31550132803 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 16 | 128 | ok | 822.662392 | 0.0373305 | 0.0429999 | 0.04509314 | 414185.4370329412 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 1 | ok | 494.273857 | 0.042216000000000004 | 0.0449026 | 0.046898419999999996 | 753598.3142005712 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 2 | ok | 510.869998 | 0.051877999999999994 | 0.0584806 | 0.06184741 | 605396.2753750903 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 4 | ok | 517.36057 | 0.049876000000000004 | 0.0541645 | 0.056734889999999996 | 637099.6674737923 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 8 | ok | 544.889878 | 0.0477205 | 0.05195615 | 0.05661052 | 661967.2508251836 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 32 | 128 | ok | 727.313266 | 0.17029650000000002 | 8.8455514 | 9.09457037 | 33372.857921398245 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 1 | ok | 503.545649 | 0.045563 | 0.048410449999999994 | 0.050526129999999995 | 1393301.0087499302 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 2 | ok | 505.783561 | 0.0810395 | 0.08560904999999999 | 0.08849536999999999 | 785443.9562419354 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 4 | ok | 511.751642 | 0.060468 | 0.0638435 | 0.0668513 | 1053024.3847540496 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 8 | ok | 545.655752 | 0.0587455 | 0.06571645 | 0.06662809 | 1083902.5232912023 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 64 | 128 | ok | 744.286477 | 0.17696099999999998 | 0.19695935 | 0.20686453 | 357002.2078355291 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 1 | ok | 493.624834 | 0.06081 | 0.06315545 | 0.06355330000000001 | 2117659.126366345 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 2 | ok | 509.797479 | 0.1302835 | 0.13982345000000002 | 0.14385119 | 987477.3979538542 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 4 | ok | 516.438696 | 0.12563249999999998 | 0.1378303 | 0.14136828 | 1016871.1636946487 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 8 | ok | 547.288051 | 0.0716665 | 0.0757427 | 0.07857513999999999 | 1786656.9666082188 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `fp32` | 128 | 128 | ok | 808.81587 | 0.166596 | 6.00076045 | 6.00437534 | 96902.72837505086 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 1 | ok | 538.65582 | 0.056882 | 0.0629414 | 0.06520777 | 17284.46531025097 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 2 | ok | 511.18146 | 0.07792199999999999 | 0.08172575 | 0.08451991 | 12821.63717946548 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 4 | ok | 516.132576 | 0.06479750000000001 | 0.07275095 | 0.07510054999999999 | 15198.104249268363 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 8 | ok | 537.853309 | 0.064578 | 0.06827445 | 0.07122002999999999 | 15450.701600908997 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 1 | 128 | ok | 848.211022 | 0.15174949999999998 | 8.43306605 | 8.64271772 | 378.75909022763443 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 1 | ok | 503.685775 | 0.06934 | 0.0756428 | 0.07761448 | 28552.706154107524 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 2 | ok | 506.077966 | 0.070313 | 0.07710365 | 0.07822019999999999 | 28080.524836241402 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 4 | ok | 511.131819 | 0.0775695 | 0.0832001 | 0.08850546999999999 | 25699.07259756717 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 8 | ok | 538.626258 | 0.0762495 | 0.08423205 | 0.08598416 | 26026.958723846157 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 2 | 128 | ok | 829.402639 | 0.1911335 | 5.9977778 | 6.002809399999999 | 1917.2487449833513 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 1 | ok | 499.236751 | 0.074956 | 0.0791509 | 0.08316636999999999 | 53383.24310675527 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 2 | ok | 517.224904 | 0.09256249999999999 | 0.0983285 | 0.1048355 | 42991.217109300655 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 4 | ok | 514.961167 | 0.07702200000000001 | 0.08647664999999999 | 0.09201862 | 51127.998689078115 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 8 | ok | 549.571522 | 0.0893895 | 0.09795774999999998 | 0.10273518999999999 | 44302.901596388605 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 4 | 128 | ok | 764.181365 | 0.1921575 | 0.2226358 | 0.23966631999999996 | 20456.672716045734 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 1 | ok | 499.193783 | 0.07260849999999999 | 0.0817706 | 0.08260026000000001 | 108820.11518065093 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 2 | ok | 514.363324 | 0.0887175 | 0.09666255 | 0.09869618 | 88854.61075460668 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 4 | ok | 512.221013 | 0.0884655 | 0.09976715 | 0.10123732 | 89765.70030150053 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 8 | ok | 548.743612 | 0.079112 | 0.0888426 | 0.09213595999999999 | 100047.62266839013 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 8 | 128 | ok | 703.5829 | 0.229795 | 0.25289805 | 0.26682343 | 34749.64789919266 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 1 | ok | 498.350315 | 0.07717650000000001 | 0.0863868 | 0.08865445 | 204995.74633826347 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 2 | ok | 515.201247 | 0.080766 | 0.08921075 | 0.10005953 | 194174.99293081666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 4 | ok | 508.966885 | 0.08146400000000001 | 0.0891064 | 0.09407146999999999 | 193849.7766002481 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 8 | ok | 543.624394 | 0.08566199999999999 | 0.09494324999999999 | 0.10296680999999999 | 184179.0220093931 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 16 | 128 | ok | 786.701616 | 0.16656949999999998 | 0.20061584999999998 | 0.22066135999999997 | 94018.06533620918 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 1 | ok | 499.037691 | 0.072542 | 0.08274179999999999 | 0.08717635 | 432650.04371117475 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 2 | ok | 505.935975 | 0.0953125 | 0.10151725 | 0.10657887999999999 | 332831.79741859814 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 4 | ok | 518.006226 | 0.0857295 | 0.09237035 | 0.0981634 | 371102.8690194743 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 8 | ok | 535.69899 | 0.092118 | 0.09913375 | 0.10492871 | 343392.73741530004 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 32 | 128 | ok | 845.838466 | 0.1583855 | 0.1719413 | 0.19325709999999996 | 200433.26154899586 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 1 | ok | 497.295065 | 0.08813499999999999 | 0.09737185 | 0.10330384999999999 | 712790.2866285666 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 2 | ok | 510.793428 | 0.12055850000000001 | 0.13168789999999997 | 0.13812094 | 525975.9813068135 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 4 | ok | 510.650732 | 0.1063665 | 0.11579869999999999 | 0.11829046 | 605031.822783154 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 8 | ok | 549.114767 | 0.1209375 | 0.13084315000000002 | 0.13265390000000002 | 523832.15304673056 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 64 | 128 | ok | 714.273669 | 0.24866349999999998 | 0.29997205 | 0.31809266999999997 | 248546.54639913526 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 1 | ok | 496.579692 | 0.09172050000000001 | 0.10097375 | 0.11054359 | 1370475.7970762183 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 2 | ok | 507.852742 | 0.12000849999999999 | 0.12780885 | 0.13088535 | 1067934.482886934 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 4 | ok | 507.633088 | 0.1512235 | 0.16081394999999998 | 0.16305445999999998 | 859370.9297372957 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 8 | ok | 528.517861 | 0.129358 | 0.14345009999999997 | 0.14936286 | 990898.9032297421 | - |
| `full_mlp_capacity_search_hd64_depth3` | `compile` | `bf16` | 128 | 128 | ok | 704.424014 | 0.257401 | 0.2799016 | 0.28793804999999995 | 496188.4970481436 | - |
