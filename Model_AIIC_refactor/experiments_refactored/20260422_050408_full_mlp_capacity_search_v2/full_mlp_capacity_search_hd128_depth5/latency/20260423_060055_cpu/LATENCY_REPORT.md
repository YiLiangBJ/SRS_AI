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

### full_mlp_capacity_search_hd128_depth5::eager

- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`787596.827` samples/s, p50=`0.151` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.050` ms, throughput=`19441.704` samples/s

### full_mlp_capacity_search_hd128_depth5::jit

- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1181975.902` samples/s, p50=`0.107` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.025` ms, throughput=`37536.579` samples/s

### full_mlp_capacity_search_hd128_depth5::compile

- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`bf16`, throughput=`918144.545` samples/s, p50=`0.138` ms
- Lowest batch-1 p50 latency: threads=`128`, precision=`fp32`, p50=`0.047` ms, throughput=`21103.404` samples/s

## Run References

### full_mlp_capacity_search_hd128_depth5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd128_depth5/MODEL_COMPLEXITY.md`
- Trainable parameters: `54,800`
- MACs / sample: `54,272`
- FLOPs / sample estimate: `109,144`

## Results

| Run | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.05036 | 0.05706055 | 0.05784004 | 19441.704244240686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.061142 | 0.06830235 | 0.0711347 | 16066.017192566316 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.056179 | 0.06409365 | 0.06613920999999999 | 17409.494999296658 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.059718 | 0.0693604 | 0.1227663499999998 | 15705.091056547437 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 1 | 128 | ok | 0.0 | 0.0503805 | 0.05481199999999999 | 0.059198979999999984 | 19665.961838594372 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.0526505 | 0.0606047 | 0.08659082999999992 | 35614.26051974027 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.053530499999999995 | 0.06646985 | 0.1412279099999999 | 34265.996652212125 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.0552055 | 0.06503385 | 0.10360240999999987 | 34568.01048655166 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.053136 | 0.06547729999999999 | 0.14300784999999988 | 34006.23742406832 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 2 | 128 | ok | 0.0 | 0.0517195 | 0.06124469999999999 | 0.06689683999999999 | 37615.38047769276 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.055703 | 0.06332635 | 0.06787188999999999 | 69794.21524605775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.0576675 | 0.0665046 | 0.06835297 | 67678.7115597608 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.0560885 | 0.0647669 | 0.06944538 | 68308.46051514825 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.058171 | 0.06359485 | 0.07322561 | 67788.79722337087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 4 | 128 | ok | 0.0 | 0.0561725 | 0.0588592 | 0.06715067999999999 | 70591.83132505456 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.0622985 | 0.0727672 | 0.08959859999999994 | 122213.23275999032 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.062147 | 0.07146945 | 0.07500089999999998 | 123673.71534701146 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.072032 | 0.08186869999999999 | 0.08398074 | 108479.9009795464 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.0640085 | 0.07160224999999999 | 0.07225501999999999 | 123532.0913210985 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 8 | 128 | ok | 0.0 | 0.061538999999999996 | 0.07063544999999999 | 0.07413682 | 127581.57326579164 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.06883149999999999 | 0.0888408 | 0.16376815999999983 | 214507.80647534717 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.089367 | 0.0985144 | 0.10078738999999999 | 174744.40899802648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.08622250000000001 | 0.10098225 | 0.13150311999999992 | 179083.45535259182 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.0810425 | 0.0900874 | 0.09202518999999999 | 193122.42754891427 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 16 | 128 | ok | 0.0 | 0.4645985 | 0.48736254999999995 | 0.49868471000000003 | 34834.71385481509 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.0788705 | 0.08947975 | 0.09035974999999999 | 389227.9226273274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.1105385 | 0.1259104 | 0.12743344 | 285939.3608397467 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.104123 | 0.11425745 | 0.12095336999999998 | 302824.3480617633 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.098046 | 0.11187509999999999 | 0.14436917999999987 | 318972.7164693388 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 32 | 128 | ok | 0.0 | 0.22751749999999998 | 0.24449235 | 0.25134922 | 138765.57784545104 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.103053 | 0.1199335 | 0.20606328999999984 | 577441.6986891171 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.145584 | 0.1620738 | 0.2294775799999998 | 427166.50843707245 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.1431905 | 0.1614545 | 0.20205757999999985 | 437853.7721649789 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.1279545 | 0.1390576 | 0.14145396 | 500506.4499640573 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 64 | 128 | ok | 0.0 | 0.316226 | 0.32707035 | 0.33096446 | 202460.82272359537 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.15094649999999998 | 0.1893016 | 0.26763495999999987 | 787596.8267231603 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.214266 | 0.22514204999999998 | 0.22910241 | 628206.1827659879 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.184549 | 0.20360314999999998 | 0.20892180999999999 | 685122.4378026582 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.17479250000000002 | 0.19874914999999999 | 0.21661589999999997 | 721356.94454839 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `fp32` | 128 | 128 | ok | 0.0 | 0.6907945 | 0.72807525 | 0.73726483 | 185600.9075884381 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.11207800000000001 | 0.12408485 | 0.12919909000000002 | 8819.577203580184 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.13342199999999999 | 0.1545649 | 0.24510546999999966 | 7129.909227699641 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.1405885 | 0.1609543 | 0.2592880199999996 | 6758.340400304612 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.13436199999999998 | 0.1554392 | 0.15678534 | 7270.251685935016 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 1 | 128 | ok | 0.0 | 0.450468 | 0.48170275 | 0.48636023 | 2245.171870151832 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.135204 | 0.15330444999999998 | 0.24793079999999973 | 14095.692271867561 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.15335700000000002 | 0.1817123 | 0.22359593999999985 | 12379.528617261123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.1562065 | 0.18191895 | 0.18705465999999998 | 12432.839355904325 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.1641655 | 0.18698265 | 0.19366008999999998 | 11961.208842826007 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 2 | 128 | ok | 0.0 | 0.45755250000000003 | 0.47965755 | 0.48509913 | 4365.687633565484 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.14023950000000002 | 0.16495895 | 0.17097573 | 27532.865292978597 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.1599255 | 0.18870484999999998 | 0.2636508699999998 | 24254.23679071819 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.1600785 | 0.19757205 | 0.2746465499999998 | 23338.897715530344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.161306 | 0.18490109999999998 | 0.2177873999999999 | 24354.34500389487 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 4 | 128 | ok | 0.0 | 0.492396 | 0.51355425 | 0.51980956 | 8116.966131309459 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.130921 | 0.1538598 | 0.24125267999999983 | 58364.71409461155 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.1708595 | 0.1803843 | 0.18710552 | 47550.868730596274 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.1609905 | 0.17765245 | 0.18318606999999998 | 49456.577310583474 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.168741 | 0.19267494999999998 | 0.19815555999999998 | 46785.49715088019 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 8 | 128 | ok | 0.0 | 0.459242 | 0.4786682 | 0.48656214 | 17499.222815766694 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.13322450000000002 | 0.16803595 | 0.23936367999999975 | 115144.26280756833 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.1652245 | 0.20026394999999997 | 0.2780528299999999 | 93480.07992079432 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.16156700000000002 | 0.1796392 | 0.18085531 | 97604.57628816385 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.1733785 | 0.1985358 | 0.20243227 | 90289.59144975626 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 16 | 128 | ok | 0.0 | 0.43432950000000003 | 0.4563452 | 0.4860812199999999 | 36710.931790446295 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.1418995 | 0.15699665000000002 | 0.16675248999999998 | 222077.31677302785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.189321 | 0.22578425 | 0.2867539799999998 | 164450.11888715785 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.1899995 | 0.22008824999999996 | 0.23643983999999998 | 166805.54966233866 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.1891515 | 0.20329195 | 0.23742173999999988 | 167988.86149853835 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 32 | 128 | ok | 0.0 | 0.406267 | 0.4308005 | 0.44098611 | 78172.85415393203 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.14958549999999998 | 0.17978555 | 0.26274359999999974 | 412996.5896806607 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 0.20736 | 0.22630605 | 0.27541121999999985 | 305482.1735400315 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 0.229191 | 0.23598795 | 0.23827029 | 280368.01806971873 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 0.347425 | 0.36068259999999996 | 0.36907637 | 184764.2904208261 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 64 | 128 | ok | 0.0 | 0.7298175 | 0.8004190499999999 | 0.9530790099999995 | 86342.98677605891 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.184153 | 0.21976815 | 0.31254286999999975 | 666327.1868467847 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 0.263695 | 0.28450945 | 0.3010242699999999 | 482553.1767945907 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 0.2783585 | 0.29175275 | 0.31184001999999994 | 458676.47469861736 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 0.297709 | 0.3243437 | 0.33362316 | 425457.7143893056 | - |
| `full_mlp_capacity_search_hd128_depth5` | `eager` | `bf16` | 128 | 128 | ok | 0.0 | 0.890023 | 0.9567222 | 0.97364958 | 142264.77203481813 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 1 | ok | 52.892066 | 0.0251415 | 0.0275923 | 0.029673269999999995 | 39369.179758887396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 2 | ok | 52.461091 | 0.0315995 | 0.0356835 | 0.039344519999999994 | 31421.601219660875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 4 | ok | 54.178599 | 0.0325245 | 0.037817949999999996 | 0.03952924 | 29905.594020795153 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 8 | ok | 52.839915 | 0.0334645 | 0.03965655 | 0.04277378999999999 | 29020.331644350037 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 1 | 128 | ok | 57.509804 | 0.0251365 | 0.03053225 | 0.04969560999999993 | 37536.57939662201 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 1 | ok | 53.143387 | 0.0270245 | 0.02993344999999999 | 0.03363992 | 73045.69885011461 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 2 | ok | 53.80232 | 0.0266785 | 0.032233899999999996 | 0.03655974999999999 | 73945.24502496392 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 4 | ok | 52.07678 | 0.026959499999999997 | 0.0390987 | 0.041029239999999995 | 69335.96255303334 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 8 | ok | 52.986345 | 0.02725 | 0.0284475 | 0.034654359999999995 | 73370.1015368835 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 2 | 128 | ok | 57.212567 | 0.026449 | 0.0274323 | 0.028929009999999998 | 75949.29017793399 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 1 | ok | 53.470884 | 0.028271 | 0.02886155 | 0.03160113999999999 | 140936.18268711746 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 2 | ok | 52.445554 | 0.0279305 | 0.030745199999999997 | 0.03302495999999999 | 140665.27642485133 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 4 | ok | 52.12399 | 0.0287385 | 0.03997625 | 0.04414311999999999 | 132735.14030104328 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 8 | ok | 52.037172 | 0.0287075 | 0.02926955 | 0.03187421 | 139457.3713680071 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 4 | 128 | ok | 59.11093 | 0.027642 | 0.03025915 | 0.03346537999999999 | 142431.67908433522 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 1 | ok | 54.772191 | 0.0313165 | 0.032312049999999995 | 0.0338483 | 254620.4055339199 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 2 | ok | 53.736229 | 0.031096 | 0.0433136 | 0.04414714 | 245859.1178083135 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 4 | ok | 52.581731 | 0.0429605 | 0.0465181 | 0.05497327999999998 | 182334.93560157996 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 8 | ok | 53.334644 | 0.030392000000000002 | 0.03180325 | 0.03367790999999999 | 260178.50847466447 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 8 | 128 | ok | 53.972556 | 0.0311175 | 0.0439407 | 0.04742782999999999 | 246589.66493396327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 1 | ok | 52.324749 | 0.035339499999999996 | 0.0372552 | 0.03996872 | 447923.0647344011 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 2 | ok | 54.928882 | 0.052714 | 0.0571751 | 0.058312789999999996 | 300840.548492488 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 4 | ok | 52.859308 | 0.05007 | 0.05658534999999999 | 0.059922899999999994 | 315266.09049161413 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 8 | ok | 54.839645 | 0.050328 | 0.05426745 | 0.059086440000000004 | 313622.0564119834 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 16 | 128 | ok | 142.699914 | 0.267073 | 0.29092865 | 0.29886344000000004 | 59797.89656413993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 1 | ok | 53.90693 | 0.046249 | 0.04863225 | 0.04946514 | 687616.3307159503 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 2 | ok | 52.906817 | 0.0735665 | 0.0802835 | 0.08193169 | 429560.4147191023 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 4 | ok | 54.578974 | 0.0677285 | 0.07543905 | 0.07787403999999999 | 464323.9588260729 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 8 | ok | 54.78715 | 0.0667935 | 0.07373864999999999 | 0.07588225 | 473064.16967629106 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 32 | 128 | ok | 108.199872 | 0.3169285 | 0.34440375 | 0.34955042000000003 | 101067.64065551967 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 1 | ok | 54.489151 | 0.0664005 | 0.06973974999999999 | 0.07266164 | 956596.5252229094 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 2 | ok | 55.031097 | 0.115942 | 0.1238608 | 0.12462827 | 548078.5650069992 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 4 | ok | 52.520113 | 0.10284950000000001 | 0.11101604999999999 | 0.11510298999999999 | 617432.4726646243 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 8 | ok | 53.317282 | 0.09725400000000001 | 0.1058704 | 0.11057874 | 647976.4606351262 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 64 | 128 | ok | 160.047885 | 0.277108 | 0.29768215 | 0.30150262 | 228967.05092584973 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 1 | ok | 54.259786 | 0.106649 | 0.11654805 | 0.11770103 | 1181975.9017275686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 2 | ok | 56.264639 | 0.1588585 | 0.16620535 | 0.16897321999999998 | 803065.9048582726 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 4 | ok | 55.855873 | 0.160332 | 0.1679861 | 0.16927655 | 798677.6892507344 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 8 | ok | 54.650522 | 0.1566055 | 0.1706336 | 0.17625041 | 809757.1690079615 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `fp32` | 128 | 128 | ok | 103.475831 | 0.6571745 | 0.70804885 | 0.71666221 | 194266.98694817247 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 1 | ok | 55.393432 | 0.064638 | 0.07365205 | 0.07515841 | 15104.284511553116 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 2 | ok | 52.58209 | 0.07314799999999999 | 0.0796875 | 0.08422742999999999 | 13443.400728686094 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 4 | ok | 54.428632 | 0.0740825 | 0.08286115000000001 | 0.08836777999999999 | 13260.548832899314 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 8 | ok | 52.971397 | 0.0751085 | 0.0820535 | 0.08876415 | 13097.501468229913 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 1 | 128 | ok | 54.592273 | 0.35783149999999997 | 0.39157385 | 0.4030326 | 2768.4212828476648 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 1 | ok | 52.497305 | 0.0777015 | 0.08585265 | 0.09240131999999998 | 25466.4952601759 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 2 | ok | 52.16878 | 0.0975545 | 0.10724439999999999 | 0.11187480999999999 | 20206.363549659964 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 4 | ok | 52.198114 | 0.0996145 | 0.1057451 | 0.11013008999999999 | 19903.666255324228 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 8 | ok | 52.200463 | 0.1016115 | 0.11105475 | 0.11784197999999999 | 19374.09280810426 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 2 | 128 | ok | 56.476856 | 0.467077 | 0.4970082 | 0.5095772799999999 | 4288.899774116516 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 1 | ok | 53.742302 | 0.0823855 | 0.0983983 | 0.10126972000000001 | 47410.891229933346 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 2 | ok | 54.391568 | 0.09252099999999999 | 0.10592184999999998 | 0.12081689999999999 | 42362.08440941853 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 4 | ok | 53.67153 | 0.103125 | 0.1172684 | 0.12423317999999998 | 38366.507573548595 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 8 | ok | 53.766227 | 0.100853 | 0.11397499999999999 | 0.12070207 | 38858.6741109524 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 4 | 128 | ok | 60.404923 | 0.39087950000000005 | 0.41166135 | 0.42227243 | 10226.281533314335 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 1 | ok | 55.318999 | 0.08357200000000001 | 0.100555 | 0.10139001 | 92820.15236892112 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 2 | ok | 53.902794 | 0.1090345 | 0.11734104999999999 | 0.12836649 | 73005.26349698498 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 4 | ok | 54.625997 | 0.1026745 | 0.11842504999999999 | 0.11971673000000001 | 77622.19917274141 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 8 | ok | 52.982201 | 0.1075795 | 0.11896205 | 0.12135387 | 74755.3723882809 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 8 | 128 | ok | 56.311832 | 0.2700685 | 23.090069999999994 | 24.00367024 | 3483.1670466482624 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 1 | ok | 53.215935 | 0.0848655 | 0.10039565 | 0.10125479999999999 | 183133.58038053327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 2 | ok | 54.602953 | 0.10631299999999999 | 0.12182174999999999 | 0.12713750999999998 | 147446.33091785526 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 4 | ok | 53.874506 | 0.1054135 | 0.12548035 | 0.12751307 | 149081.14766394498 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 8 | ok | 52.839573 | 0.1082445 | 0.1242177 | 0.12534961 | 144296.42236861493 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 16 | 128 | ok | 158.002587 | 0.393565 | 0.43331644999999996 | 0.44616931 | 40164.31220121517 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 1 | ok | 52.316517 | 0.08831549999999999 | 0.09982559999999997 | 0.11039774999999998 | 354867.66430317407 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 2 | ok | 54.44913 | 0.142505 | 0.1536079 | 0.1544973 | 223748.93579412412 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 4 | ok | 52.724947 | 0.1332605 | 0.1424496 | 0.14898999999999998 | 239239.3146750592 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 8 | ok | 54.057623 | 0.1141745 | 0.1266312 | 0.13295917 | 275749.3920587622 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 32 | 128 | ok | 108.154024 | 0.454414 | 0.4969479 | 0.50359791 | 70111.09321817587 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 1 | ok | 53.245856 | 0.0982105 | 0.10673479999999999 | 0.11076567 | 645698.9388341502 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 2 | ok | 53.85254 | 0.14755200000000002 | 0.15553140000000001 | 0.15888551999999997 | 433132.5267774033 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 4 | ok | 53.508344 | 0.1602805 | 0.17424675 | 0.17608789 | 395497.60581816523 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 8 | ok | 55.001213 | 0.151347 | 0.16110705 | 0.16270755 | 419337.59598735854 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 64 | 128 | ok | 114.011208 | 0.47756350000000003 | 0.5051599 | 0.51576102 | 134158.64939469087 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 1 | ok | 55.561468 | 0.11672199999999999 | 0.12639395 | 0.13105948 | 1079798.6378002746 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 2 | ok | 55.095256 | 0.1865055 | 0.19967885000000002 | 0.20280534 | 687673.7383523918 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 4 | ok | 54.502588 | 0.2078935 | 0.21695945 | 0.2200279 | 618934.7417451271 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 8 | ok | 56.966231 | 0.215343 | 0.24018174999999997 | 0.24960121999999998 | 588452.285898072 | - |
| `full_mlp_capacity_search_hd128_depth5` | `jit` | `bf16` | 128 | 128 | ok | 114.162385 | 0.659872 | 0.6817674 | 0.68885447 | 194359.36329573227 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 1 | ok | 496.959062 | 0.0503795 | 0.05245965 | 0.05415722 | 19920.2631705808 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 2 | ok | 509.848277 | 0.060872499999999996 | 0.0672838 | 0.0686738 | 16275.453588753791 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 4 | ok | 508.347499 | 0.057930499999999996 | 0.0617138 | 0.06799799999999999 | 17117.607205006283 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 8 | ok | 538.710263 | 0.058313500000000004 | 0.06334024999999999 | 0.06676843 | 17035.670650060747 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 1 | 128 | ok | 710.503049 | 0.047085 | 0.0505795 | 0.05420989999999999 | 21103.40414791629 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 1 | ok | 498.082895 | 0.047324000000000005 | 0.0537673 | 0.05684362 | 41334.90240622868 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 2 | ok | 513.732378 | 0.050524 | 0.0567154 | 0.05928701 | 39124.22763884113 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 4 | ok | 510.336192 | 0.050461 | 0.0576944 | 0.06083074 | 39136.507747658754 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 8 | ok | 536.920317 | 0.0466915 | 0.049155649999999995 | 0.05102771 | 42699.368689833915 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 2 | 128 | ok | 728.058123 | 0.049541 | 0.0538162 | 0.05736259 | 40029.84625336651 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 1 | ok | 494.724153 | 0.052704 | 0.058335399999999996 | 0.06245379999999999 | 75282.55424672653 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 2 | ok | 505.259824 | 0.051515500000000006 | 0.058873949999999994 | 0.06065844 | 76637.45502339359 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 4 | ok | 508.93182 | 0.0513845 | 0.05929245 | 0.0608649 | 76201.47823247623 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 8 | ok | 541.294538 | 0.0509125 | 0.05359085 | 0.0548612 | 78409.60235354262 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 4 | 128 | ok | 816.105376 | 0.0511275 | 0.057814449999999996 | 0.06070789 | 76919.17179589343 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 1 | ok | 500.338178 | 0.058793 | 0.061143699999999995 | 0.06623163 | 135354.85813964898 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 2 | ok | 510.399684 | 0.0590185 | 0.06174065 | 0.06384572 | 135244.427253375 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 4 | ok | 513.021919 | 0.0795975 | 0.08706985 | 0.09232002 | 98914.31646250752 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 8 | ok | 549.367429 | 0.05952 | 0.061291450000000004 | 0.06180807 | 134949.35182139455 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 8 | 128 | ok | 787.077946 | 0.055870500000000003 | 0.06457785 | 0.06647709 | 140537.9652772849 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 1 | ok | 502.933928 | 0.064438 | 0.06664975000000001 | 0.06708649 | 248468.3479277429 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 2 | ok | 507.624554 | 0.09851399999999999 | 0.1050366 | 0.1058081 | 161717.21037003526 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 4 | ok | 518.088979 | 0.0877675 | 0.0936066 | 0.09419067 | 180878.3543325566 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 8 | ok | 533.719445 | 0.08413699999999999 | 0.09079085 | 0.09253138999999999 | 188716.81061041396 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 16 | 128 | ok | 847.471723 | 0.27734000000000003 | 0.2949846 | 1.0259302799999972 | 52151.13320175167 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 1 | ok | 500.519814 | 0.07737250000000001 | 0.07964155 | 0.07998666 | 414365.8570821081 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 2 | ok | 508.877391 | 0.123847 | 0.13038805 | 0.13313174 | 259452.8819782893 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 4 | ok | 513.495822 | 0.10477800000000001 | 0.11352614999999999 | 0.11619602 | 301886.8494126131 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 8 | ok | 543.705543 | 0.0980145 | 0.1052046 | 0.10898052 | 324085.02694767 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 32 | 128 | ok | 719.642471 | 0.3337585 | 0.3480415 | 0.3854271699999999 | 95748.44585311686 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 1 | ok | 497.044223 | 0.0975725 | 0.10071179999999999 | 0.10292976999999999 | 654571.9498327875 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 2 | ok | 511.90898 | 0.17251450000000002 | 0.2291087 | 0.23508305 | 341922.4246402976 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 4 | ok | 513.113222 | 0.1576815 | 0.1620859 | 0.16667356 | 409740.60986560636 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 8 | ok | 540.885858 | 0.1327825 | 0.14104709999999998 | 0.1437743 | 481763.5194511268 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 64 | 128 | ok | 827.32795 | 0.456457 | 0.48599955 | 0.49636771 | 141473.18145951515 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 1 | ok | 497.400362 | 0.1373765 | 0.15126555 | 0.16046653 | 908575.447395327 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 2 | ok | 503.77405 | 0.211598 | 0.3287789 | 0.33399485999999995 | 597763.3563216932 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 4 | ok | 515.353493 | 0.1867545 | 0.27765055 | 0.2802577 | 621232.6848375932 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 8 | ok | 546.905553 | 0.189008 | 0.2090701 | 0.21460694 | 685879.8767130922 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `fp32` | 128 | 128 | ok | 803.496065 | 0.4220195 | 0.5036744999999997 | 0.9014585899999992 | 288414.8790351946 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 1 | ok | 499.515993 | 0.0992845 | 0.10495114999999999 | 0.10688407999999999 | 10009.883759223856 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 2 | ok | 511.385922 | 0.13671 | 0.14764275 | 0.14975214 | 7375.512413724944 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 4 | ok | 518.513358 | 0.163061 | 0.1787068 | 0.18380792999999998 | 6255.2043300025625 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 8 | ok | 536.244909 | 0.132516 | 0.14343495 | 0.14652662 | 7461.029550452079 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 1 | 128 | ok | 691.183863 | 0.4758715 | 0.5357185 | 0.5538718300000001 | 2077.6510442959775 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 1 | ok | 499.566229 | 0.11430850000000001 | 0.11952634999999999 | 0.123428 | 17409.04643689047 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 2 | ok | 509.111879 | 0.1528135 | 0.16255495 | 0.18831785999999992 | 13163.542935988718 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 4 | ok | 510.227497 | 0.176462 | 0.18666555 | 0.19045435 | 11853.877723458041 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 8 | ok | 553.957258 | 0.148162 | 0.15639224999999998 | 0.15998938 | 13601.922332479404 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 2 | 128 | ok | 743.377272 | 0.423725 | 0.4487879 | 0.46428393999999995 | 4682.4697836713 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 1 | ok | 498.285594 | 0.1199675 | 0.13117715 | 0.15189271999999995 | 32758.60007432926 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 2 | ok | 510.501686 | 0.17263499999999998 | 0.18360475 | 0.1865083 | 23539.652898402153 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 4 | ok | 510.247554 | 0.1833975 | 0.19862285 | 0.21593612999999995 | 22103.88050200123 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 8 | ok | 534.606937 | 0.148941 | 0.1585375 | 0.16068738 | 27103.146580456978 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 4 | 128 | ok | 825.811866 | 0.416489 | 0.43600134999999995 | 0.5578007499999997 | 9514.522896390128 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 1 | ok | 496.463911 | 0.116591 | 0.12238919999999999 | 0.12716071999999998 | 68229.34612406143 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 2 | ok | 511.01315 | 0.1547865 | 0.16469 | 0.16962585 | 51513.551863264496 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 4 | ok | 511.871166 | 0.1870595 | 0.20136545 | 0.22265305999999993 | 43843.385726544635 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 8 | ok | 539.206518 | 0.15112799999999998 | 0.16202075 | 0.16837988999999998 | 52723.109039561576 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 8 | 128 | ok | 757.3624 | 0.397807 | 0.41745214999999997 | 0.4413684399999999 | 19986.320363027524 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 1 | ok | 502.693537 | 0.118371 | 0.12608275 | 0.13062155 | 133960.20008965288 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 2 | ok | 515.188512 | 0.1761125 | 0.19082615 | 0.21025639999999993 | 91678.75489249405 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 4 | ok | 505.733837 | 0.182641 | 0.19873349999999998 | 0.21475441999999995 | 89293.43776944993 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 8 | ok | 539.544985 | 0.16899750000000002 | 0.1804053 | 0.18781684999999998 | 93856.76940776144 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 16 | 128 | ok | 709.649693 | 0.476699 | 0.503051 | 0.5175777799999999 | 33576.53376347112 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 1 | ok | 497.54833 | 0.116992 | 0.12310755 | 0.12528859 | 270277.1641641535 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 2 | ok | 506.373406 | 0.19355699999999998 | 0.20155815 | 0.20353773 | 169730.19052426048 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 4 | ok | 512.773061 | 0.182926 | 0.20998585 | 0.23408426999999996 | 171150.28717413652 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 8 | ok | 541.061878 | 0.173118 | 0.19107425 | 0.21072199999999994 | 182906.6844390995 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 32 | 128 | ok | 833.677706 | 0.561518 | 0.5808296000000001 | 0.5841137200000001 | 57200.619554210534 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 1 | ok | 494.287028 | 0.126163 | 0.14809779999999997 | 0.1825866799999999 | 491964.45249352953 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 2 | ok | 516.572591 | 0.191699 | 0.2165767 | 0.23001429999999995 | 326078.0624767606 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 4 | ok | 519.103204 | 0.21555000000000002 | 0.2531753 | 0.27529773999999996 | 286752.30344989005 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 8 | ok | 545.8074 | 0.19721349999999999 | 0.21821724999999997 | 0.29091329999999976 | 320685.75441724586 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 64 | 128 | ok | 693.689789 | 0.487763 | 0.512606 | 0.51500396 | 130592.04843789668 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 1 | ok | 501.350235 | 0.137876 | 0.1477307 | 0.16723289999999996 | 918144.5446433439 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 2 | ok | 506.287026 | 0.2173115 | 0.2702093 | 0.2985180899999999 | 563950.4375682634 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 4 | ok | 520.152632 | 0.2372275 | 0.2930261 | 0.31809736999999993 | 516005.648649335 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 8 | ok | 536.984659 | 0.239029 | 0.28073739999999997 | 0.32082666999999987 | 517259.1182881892 | - |
| `full_mlp_capacity_search_hd128_depth5` | `compile` | `bf16` | 128 | 128 | ok | 760.072007 | 0.722402 | 0.7554242 | 0.76165236 | 177615.48163143705 | - |
