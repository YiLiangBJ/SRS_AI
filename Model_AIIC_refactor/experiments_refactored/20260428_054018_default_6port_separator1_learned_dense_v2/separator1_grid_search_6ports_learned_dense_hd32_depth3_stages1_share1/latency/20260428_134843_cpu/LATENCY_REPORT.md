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

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`2`, batch=`128`, precision=`fp32`, throughput=`296670.609` samples/s, p50=`0.429` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`bf16`, p50=`0.272` ms, throughput=`3669.551` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`618494.632` samples/s, p50=`0.206` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.086` ms, throughput=`11600.853` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`378538.007` samples/s, p50=`0.336` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.254` ms, throughput=`3922.661` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1001694.115` samples/s, p50=`0.128` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`32055.536` samples/s

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`71138.485` samples/s, p50=`1.817` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.377` ms, throughput=`2615.820` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,168`
- MACs / sample: `26,112`
- FLOPs / sample estimate: `53,496`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.29370799999999997 | 0.3015213 | 0.30701424 | 3395.1411998483322 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.27576100000000003 | 0.28579024999999997 | 0.29645149 | 3605.062257262145 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.27988250000000003 | 0.28585485 | 0.29085517 | 3565.7076482004372 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.27916450000000004 | 0.2863757 | 0.29611246999999996 | 3565.40176908105 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.3071975 | 0.3408605999999999 | 0.35965035 | 6424.262950735925 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.29910749999999997 | 0.3174878 | 0.35533776999999994 | 6616.940359862302 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.302147 | 0.31003245 | 0.31701587 | 6597.915072033066 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.3025555 | 0.3119667 | 0.39517403999999967 | 6516.317837312305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.30420749999999996 | 0.3173581 | 0.33188313999999997 | 13058.33609245093 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.30318449999999997 | 0.34925845 | 0.35520169 | 12989.698649383094 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.29999050000000005 | 0.30485070000000003 | 0.30832814999999997 | 13323.508578107836 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.308581 | 0.354719 | 0.35946947 | 12780.764642474724 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.3045715 | 0.3535578 | 0.35878213 | 25727.71981072374 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.3051525 | 0.325793 | 0.35301028999999995 | 25948.658632745715 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.3138585 | 0.36112725 | 0.36606296 | 25068.36456371081 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.3067435 | 0.3139968 | 0.32805398999999996 | 26002.444619830934 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.32845599999999997 | 0.34337334999999997 | 0.35297294999999995 | 48418.35980627329 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.3237085 | 0.37556449999999997 | 0.3777857 | 48562.60752064822 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.320757 | 0.33532245 | 0.37375484 | 49405.232026422906 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.321747 | 0.37546464999999996 | 0.38854457 | 48910.598945132886 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.341914 | 0.40333859999999994 | 0.41233483 | 91814.10673252186 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.3295765 | 0.3698014999999999 | 0.39809859000000003 | 95668.80416685477 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.32823650000000004 | 0.3958873 | 0.40091204 | 93500.93687938755 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.3276795 | 0.332026 | 0.33492096 | 97610.02464714128 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.3578255 | 0.44216289999999997 | 0.44445879 | 172846.81348527072 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.3556975 | 0.36866965 | 0.38668888999999995 | 178897.60722773167 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.3558805 | 0.38654944999999996 | 0.44323419999999997 | 177036.95679667563 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.3630185 | 0.37298875 | 0.37687163999999995 | 176084.8394364889 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.443406 | 0.45621075 | 0.45930772 | 288567.82211777667 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.4285955 | 0.44585020000000003 | 0.49573541999999976 | 296670.60944996943 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.526958 | 0.59371445 | 0.6073301600000001 | 236854.19437135992 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.432519 | 0.4464666 | 0.4496602 | 295891.5823653055 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.272082 | 0.2971186 | 0.29929464 | 3633.600778462631 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.2719275 | 0.2817339 | 0.29624165999999996 | 3652.7782922107203 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.271919 | 0.27744854999999996 | 0.27842544 | 3669.5507868030627 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.275371 | 0.28083335 | 0.28415233 | 3625.133894320387 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.291582 | 0.306885 | 0.3176261 | 6806.9756797650125 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.295475 | 0.302281 | 0.30489125 | 6751.923656809443 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.29767750000000004 | 0.3031597 | 0.30547653 | 6704.940253618387 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.297319 | 0.30515135 | 0.30587727 | 6712.155391230059 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.3253835 | 0.33026625 | 0.33239507 | 12284.382974483986 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.3303825 | 0.3395077 | 0.36339709 | 12045.206140453367 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.33410300000000004 | 0.34222215 | 0.34502716 | 11942.789737751862 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.31989650000000003 | 0.32356035 | 0.32989675 | 12486.52002122958 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 0.670018 | 0.699071 | 0.70145172 | 11873.09607483755 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 0.6993255 | 0.73905765 | 0.7495624 | 11374.214383012566 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 0.707342 | 0.73841655 | 0.7716403399999999 | 11257.188453580604 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 0.706839 | 0.7312138 | 0.74353198 | 11266.216874224181 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 0.7898765000000001 | 0.82429975 | 0.82831647 | 20135.16585826912 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 0.8405665 | 0.9221147 | 0.9309311100000001 | 18824.0077630208 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 0.8582084999999999 | 0.9023131000000001 | 0.90872242 | 18588.555635070683 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 0.9117770000000001 | 0.9526347 | 1.0027749999999997 | 17492.573801114202 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 0.8284515 | 0.8496892 | 0.85705425 | 38541.56594671503 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 0.9135275 | 0.9693153499999999 | 0.98009079 | 34792.492032464965 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 0.89917 | 0.9437874 | 0.94756189 | 35533.90277448935 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 0.8944425 | 0.93697115 | 0.94916601 | 35649.918513198754 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 0.8801895 | 0.9109579 | 0.9165892 | 72510.72767891508 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.016546 | 1.0911378 | 1.1211026199999998 | 62143.12731287738 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.0514325 | 1.11854245 | 1.13412793 | 60400.72673399397 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.122411 | 1.19327015 | 1.22615578 | 56538.26220710994 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 0.9422225 | 1.0086951499999999 | 1.0285047 | 134380.79805900375 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.193063 | 1.22581085 | 1.25460185 | 107992.65629688559 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.262079 | 1.2863624999999999 | 1.29867887 | 102012.79858946904 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.277465 | 1.34183345 | 1.35602711 | 99988.25294259569 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 171.526866 | 0.08671000000000001 | 0.09266365 | 0.09472438999999999 | 11423.252762085402 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 172.88835 | 0.086253 | 0.08752495 | 0.08805381 | 11592.176115758544 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 173.628138 | 0.085815 | 0.08867214999999999 | 0.09104754 | 11600.85270907753 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 171.233955 | 0.0883375 | 0.08994205000000001 | 0.10809974999999993 | 11217.394679320485 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 173.311428 | 0.0962335 | 0.09830744999999999 | 0.10026472 | 20726.982323407665 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 173.154915 | 0.0950535 | 0.096967 | 0.09967981 | 20989.1603580247 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 172.87417 | 0.098082 | 0.09949465 | 0.10195003 | 20344.803733678382 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 175.575838 | 0.094539 | 0.0964261 | 0.09838531 | 21060.820912889707 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 172.557149 | 0.1043225 | 0.1093691 | 0.11047839 | 38089.92727109286 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 175.685299 | 0.1000315 | 0.10309975 | 0.10459700999999999 | 39789.61638234001 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 172.279347 | 0.09851499999999999 | 0.100828 | 0.12010093999999995 | 40223.384588651534 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 172.06523 | 0.096843 | 0.09926755 | 0.11715743999999993 | 40845.41012978425 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 172.503594 | 0.1007155 | 0.10394455 | 0.10777262 | 78953.92367445244 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 175.505318 | 0.1033345 | 0.10877835 | 0.11033050999999999 | 76988.265256139 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 171.138772 | 0.100659 | 0.10260620000000001 | 0.10538027 | 79176.26596416206 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 176.491767 | 0.10199849999999999 | 0.1063574 | 0.10739134 | 78022.54968719784 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 176.638474 | 0.11572 | 0.1202418 | 0.12104878 | 137823.532127699 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 172.470432 | 0.11646200000000001 | 0.1182502 | 0.11973104 | 137219.1745271813 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 174.38627 | 0.118218 | 0.12440115 | 0.12516699 | 134495.35074790343 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 172.899006 | 0.1159265 | 0.11845989999999999 | 0.12307029 | 137515.4769371934 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 173.208819 | 0.1339385 | 0.1352706 | 0.13645374 | 238922.75701928927 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 173.536235 | 0.131805 | 0.1377113 | 0.13888484999999998 | 241648.91554007932 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 173.261066 | 0.1316695 | 0.13359195 | 0.13431259 | 242860.39762015027 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 175.262647 | 0.1320735 | 0.13392315 | 0.13478152999999998 | 242279.64552670758 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 177.916661 | 0.156259 | 0.15876289999999998 | 0.16097715 | 408842.33991714305 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 173.349467 | 0.1559835 | 0.1587599 | 0.16239449 | 409625.58685265714 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 176.823697 | 0.158338 | 0.16307395 | 0.16549276000000002 | 402927.62173356337 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 177.121149 | 0.15733999999999998 | 0.16049365 | 0.16215511999999999 | 406096.67867486086 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 174.686789 | 0.20634950000000002 | 0.2101192 | 0.21129978000000002 | 618494.6323846688 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 175.215835 | 0.207459 | 0.21469885 | 0.21594572 | 614379.9859575775 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 176.103069 | 0.280284 | 0.28927185 | 0.29086274 | 456182.2025930823 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 175.802427 | 0.21061950000000002 | 0.21557815 | 0.21617447 | 607552.7348654105 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 170.912852 | 0.095207 | 0.09659645 | 0.09724596 | 10483.564915282312 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 172.518304 | 0.09923950000000001 | 0.10984434999999995 | 0.12186084 | 9968.078226288135 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 173.553384 | 0.0926035 | 0.09681949999999999 | 0.10022077 | 10735.993125528881 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 174.542992 | 0.0937385 | 0.0962348 | 0.09715443 | 10640.382934613357 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 171.787638 | 0.10990749999999999 | 0.11662815 | 0.13414713999999994 | 17981.60445900635 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 174.346688 | 0.10969100000000001 | 0.11474055000000001 | 0.11597446 | 18151.55129510411 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 175.55067 | 0.1095125 | 0.11559635 | 0.11665785 | 18145.062882622675 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 170.426669 | 0.10921 | 0.1106444 | 0.11265435 | 18289.20909229741 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 175.18606 | 0.1412245 | 0.14545935 | 0.14650737 | 28225.588499992027 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 175.144509 | 0.1402925 | 0.1458138 | 0.14648055 | 28333.620447353864 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 169.887912 | 0.14316299999999998 | 0.1446832 | 0.1462727 | 27888.183999544864 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 174.478581 | 0.144921 | 0.14857320000000002 | 0.14989805 | 27510.006076960344 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 180.626466 | 0.3550425 | 0.36092995 | 0.36325092000000003 | 22495.039562588452 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 179.284551 | 0.40118149999999997 | 0.4314117 | 0.44588369 | 19781.941678681724 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 181.487131 | 0.38479399999999997 | 0.39301135 | 0.39399558 | 20727.4108660446 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 184.61111 | 0.3887305 | 0.40565 | 0.42735846 | 20457.969025918865 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 182.231563 | 0.4391895 | 0.44631345 | 0.4500494 | 36426.803752871456 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 185.067894 | 0.47884000000000004 | 0.5770955 | 0.59849137 | 31931.353975224065 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 183.577471 | 0.4735645 | 0.56150125 | 0.56356684 | 32938.89148031043 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 186.483891 | 0.504804 | 0.57222525 | 0.58671847 | 31240.909627507215 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 181.219835 | 0.46680049999999995 | 0.4739825 | 0.47869693999999996 | 68541.21462486172 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 185.0049 | 0.5529995000000001 | 0.6350461999999999 | 0.6712496299999999 | 57097.311840715636 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 181.90132 | 0.5067075 | 0.5152007 | 0.52106474 | 63047.76936400918 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 183.383057 | 0.5575715 | 0.57298905 | 0.57510171 | 57349.5777924083 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 181.226258 | 0.5449360000000001 | 0.67264005 | 0.67510376 | 113239.81116836838 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 185.603324 | 0.7054725 | 0.78399195 | 0.8021154899999999 | 88881.56603542053 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 186.467587 | 0.72899 | 0.79033845 | 0.8122938699999999 | 86807.68901275622 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 189.024262 | 0.7402915 | 0.7779360999999999 | 0.79833816 | 86450.09800469391 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 181.498227 | 0.6237375000000001 | 0.7468942 | 0.75051015 | 199339.19679198446 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 186.345986 | 0.8498779999999999 | 0.90154455 | 0.91743115 | 149436.74605221392 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 185.472274 | 0.8792365 | 0.92518505 | 0.9361622399999999 | 144861.85633859574 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 190.021555 | 0.9258355 | 0.9558467 | 0.95990252 | 138421.85331548672 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.856717 | 0.2563205 | 0.26180765 | 0.26273392 | 3887.541812455964 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 497.365587 | 0.2596145 | 0.26594785 | 0.26731311 | 3843.8694049982296 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 507.64968 | 0.25441800000000003 | 0.25984305 | 0.26173709 | 3922.660819286938 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 534.137387 | 0.25740399999999997 | 0.25963575 | 0.26255219 | 3888.559182276445 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.795087 | 0.264189 | 0.26689105 | 0.26927987 | 7570.139041500788 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 511.492943 | 0.2641015 | 0.27051055 | 0.27612513 | 7549.315526209184 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 519.618327 | 0.2661265 | 0.29391694999999995 | 0.30547796 | 7441.62676342369 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 542.989536 | 0.260691 | 0.2645528 | 0.26916065 | 7662.430109059368 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 492.461995 | 0.2600925 | 0.26630985 | 0.26808518 | 15336.100221414945 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 505.741189 | 0.26027 | 0.2659377 | 0.26763376 | 15341.346885879171 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 515.805518 | 0.2593345 | 0.26418674999999997 | 0.26613704 | 15413.765185930468 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 541.246781 | 0.25825549999999997 | 0.26194019999999996 | 0.26323985 | 15473.713024208777 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 491.983472 | 0.2612085 | 0.26930299999999996 | 0.27288672999999997 | 30528.62760996869 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 507.873411 | 0.263726 | 0.2677793 | 0.27015342 | 30287.792331146124 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 514.711766 | 0.259837 | 0.26452634999999997 | 0.26532933 | 30717.20291607622 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 539.81588 | 0.2656495 | 0.27092194999999997 | 0.27413791 | 30052.278191835565 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 494.997506 | 0.27362 | 0.2806295 | 0.34742998999999974 | 57814.03263645503 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 515.507293 | 0.26887150000000004 | 0.2746554 | 0.27953744999999997 | 59373.22947174303 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.075664 | 0.26673800000000003 | 0.2716539 | 0.2746169 | 59886.183314302 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 535.546893 | 0.273453 | 0.31361490000000003 | 0.32285106999999996 | 57623.897321706085 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 493.245722 | 0.28306549999999997 | 0.33844065 | 0.35214220999999996 | 109596.3825522011 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 499.997575 | 0.2788315 | 0.2885742 | 0.3113293399999999 | 113976.83705728904 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 509.598235 | 0.274462 | 0.2784001 | 0.27974150000000003 | 116516.92178896295 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 541.422507 | 0.2754645 | 0.28006435 | 0.28162970000000004 | 116032.13045724493 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 489.878779 | 0.296509 | 0.2988691 | 0.30282648 | 215950.53868861878 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 511.592832 | 0.3086855 | 0.31557715000000003 | 0.37206355999999996 | 205522.8617208789 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 509.639602 | 0.2937225 | 0.3644208 | 0.36709096 | 212657.33818670543 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 544.857991 | 0.3010915 | 0.3337567499999999 | 0.37057397999999997 | 209723.1588764344 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 495.295139 | 0.3369345 | 0.3427946 | 0.38527430999999984 | 378352.55837270565 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 514.60548 | 0.34571799999999997 | 0.42800069999999996 | 0.43011604999999997 | 358140.273808312 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 505.861387 | 0.424422 | 0.46512765 | 0.49966135999999994 | 294983.8367293962 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 539.507595 | 0.3361 | 0.34239805 | 0.3829185199999998 | 378538.0069607225 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 496.491424 | 0.2714755 | 0.27606395 | 0.28124553999999996 | 3674.1534401429335 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 504.704026 | 0.26985899999999996 | 0.27319115 | 0.27459644 | 3707.9054472245657 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 503.411994 | 0.266632 | 0.27129759999999997 | 0.27337054 | 3745.636520735169 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 502.351459 | 0.270045 | 0.275402 | 0.27595825999999996 | 3694.5420604032506 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.4133 | 0.2784605 | 0.28200855 | 0.28711305 | 7176.959491159313 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 507.507709 | 0.278332 | 0.28508305 | 0.29285871999999996 | 7162.258165457761 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 507.934703 | 0.276613 | 0.27977399999999997 | 0.28090126 | 7225.7786300342805 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 504.026301 | 0.277742 | 0.2819099 | 0.28309543 | 7200.442164752453 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 492.649993 | 0.310918 | 0.32143655 | 0.33395623999999996 | 12780.518808546358 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 509.757567 | 0.30645449999999996 | 0.3119621 | 0.31487938 | 13030.407628938816 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 509.939367 | 0.306281 | 0.3134412 | 0.3146597 | 13048.193764633548 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 509.832258 | 0.309407 | 0.31904315 | 0.3251302 | 12886.140446946762 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.411716 | 0.549907 | 0.58359845 | 0.59545857 | 14468.87212067184 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 502.540684 | 0.5519265 | 0.5895581999999999 | 0.60790757 | 14369.90550924933 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 497.601512 | 0.5545535 | 0.56185615 | 0.57191907 | 14386.466305870877 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 495.574541 | 0.548227 | 0.583275 | 0.5948036699999999 | 14499.26410797428 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 492.054595 | 0.7411485 | 0.8477266499999999 | 0.87733191 | 21237.062049970064 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 508.629807 | 0.7712135 | 0.8088579 | 0.8162419000000001 | 20590.35587367312 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 511.250763 | 0.739711 | 0.83991885 | 0.8709523800000001 | 21299.451302172572 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 506.575101 | 0.8043724999999999 | 0.8448950999999999 | 0.86250662 | 19798.010794865386 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 489.887126 | 0.747998 | 0.7712143499999999 | 0.78377287 | 42550.35215868175 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 499.2547 | 0.801206 | 0.8334834 | 0.8438317000000001 | 39628.193428679595 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 511.935403 | 0.800672 | 0.8345802999999999 | 0.84922115 | 39822.040279744855 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 500.33113 | 0.8043785 | 0.84787895 | 0.85566887 | 39525.0433639132 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 492.776397 | 0.743591 | 0.7686506999999999 | 0.8397917399999998 | 85441.64828046947 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.224719 | 0.886127 | 0.92091515 | 0.92706664 | 71754.63216512607 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 499.684115 | 0.888651 | 0.92537435 | 0.92770495 | 71562.08583442825 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 511.458268 | 0.9803885 | 1.03294245 | 1.0490747599999999 | 64980.930026158676 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 489.660408 | 0.795051 | 0.83106325 | 0.83765897 | 160032.11844617216 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.514823 | 0.9598675000000001 | 1.0297182500000002 | 1.04670321 | 131132.00399083862 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 510.106754 | 1.027193 | 1.07224005 | 1.07943863 | 124775.43346400699 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 503.122912 | 1.0643905 | 1.12078765 | 1.12839785 | 120539.0634992441 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 8.783403 | 0.0309245 | 0.033330799999999994 | 0.037198840000000004 | 32055.535574271715 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 8.127673 | 0.032364000000000004 | 0.035128799999999995 | 0.03791695999999999 | 30700.348633159083 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.365861 | 0.0323345 | 0.0340918 | 0.03791919 | 30827.185878682692 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.716999 | 0.0319465 | 0.03240965 | 0.03915721999999999 | 31068.810580297035 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.08464 | 0.03214 | 0.032678200000000004 | 0.03374492 | 62142.525124222906 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.118284 | 0.032923499999999994 | 0.03359205 | 0.04425498999999999 | 60001.968064552515 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.442852 | 0.0334715 | 0.0341682 | 0.03929767 | 59609.99564250932 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.761155 | 0.0337775 | 0.03637364999999999 | 0.040202779999999994 | 58659.45546427491 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.072707 | 0.03325 | 0.03396635 | 0.03800928 | 119593.40633713499 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.043416 | 0.0342175 | 0.03496425 | 0.04425189999999998 | 115649.98762545132 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.356832 | 0.034475000000000006 | 0.035292 | 0.03778110999999999 | 115810.30734897467 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.00189 | 0.034527 | 0.03900174999999998 | 0.04646709999999998 | 113955.37393601293 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.051089 | 0.036935499999999996 | 0.04001189999999999 | 0.04589189999999999 | 214698.81781463439 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.95438 | 0.0385395 | 0.04142529999999999 | 0.04492784 | 207200.4218600589 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.462451 | 0.0387315 | 0.04223669999999999 | 0.04802204999999998 | 205659.33354036373 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 9.031847 | 0.0379995 | 0.041052799999999987 | 0.04762708999999999 | 209842.78054274785 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.87718 | 0.043565 | 0.0445832 | 0.045285109999999996 | 371873.877115207 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.19532 | 0.0454165 | 0.047695499999999995 | 0.05148339 | 351428.53503061604 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.433619 | 0.0455555 | 0.0462357 | 0.057338879999999995 | 349879.4228039162 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 9.18017 | 0.044822 | 0.04875994999999999 | 0.055336559999999986 | 353659.93828634074 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.821879 | 0.0583465 | 0.0597356 | 0.06016305 | 550193.2725805336 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.212342 | 0.0582 | 0.0615901 | 0.06390509 | 548497.8358332264 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.470288 | 0.059035500000000005 | 0.060723 | 0.06654044999999999 | 543897.6384644409 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.028148 | 0.058979000000000004 | 0.061791 | 0.06361361 | 543500.414419066 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.973859 | 0.0798915 | 0.0815891 | 0.08684658999999999 | 801533.7348015427 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.238877 | 0.1211155 | 0.12923315 | 0.1317238 | 533041.5819075027 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.260015 | 0.127647 | 0.13737615 | 0.141956 | 502568.12310908746 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.73644 | 0.1265345 | 0.13672784999999998 | 0.13895345 | 509469.44488846126 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.91355 | 0.1278415 | 0.1298782 | 0.13158414999999998 | 1001694.1151722851 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.275279 | 0.270127 | 0.29802235 | 0.30233063 | 476539.0872251189 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.436181 | 0.2681745 | 0.30189104999999994 | 0.3322519199999999 | 472228.01234433544 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.863526 | 0.26101949999999996 | 0.31798599999999994 | 0.33242055 | 481601.28816304554 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 264.287906 | 0.3920195 | 0.56742485 | 0.6883578299999998 | 2498.887745064672 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 257.98343 | 0.378411 | 0.4338452 | 0.44726793 | 2637.640383793557 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 257.531103 | 0.382965 | 0.44900924999999997 | 0.47168951 | 2591.0281298078144 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 260.603366 | 0.3773995 | 0.45904144999999985 | 0.5010478299999999 | 2615.8195980134 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 261.353261 | 0.5645155 | 0.9991179999999994 | 1.1983944399999997 | 3624.8445394798127 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 258.861832 | 0.525072 | 0.9893755999999996 | 2.506332509999999 | 3431.5404758263244 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 263.124929 | 0.554216 | 0.7282042499999998 | 0.9799314299999993 | 3484.862228933912 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 259.33908 | 0.557828 | 0.6402517999999999 | 0.7075421599999999 | 3526.542486761183 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 261.992747 | 0.6553525 | 0.80938755 | 1.3192833899999996 | 5968.460981007283 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 258.096424 | 0.6487605 | 0.8528475999999999 | 0.9075927 | 6016.683541458439 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 256.683078 | 0.632889 | 0.7480038499999999 | 0.97661912 | 6249.200102386895 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 257.297692 | 0.6138174999999999 | 0.7668321499999999 | 0.99702368 | 6718.06140395458 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 258.356143 | 0.7204205 | 0.9122819999999997 | 2.8020761699999963 | 11469.01773884301 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 257.829489 | 0.744273 | 0.8391235499999998 | 0.91742382 | 10567.797732774106 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 258.467021 | 0.707496 | 0.9149310999999998 | 1.5510607099999982 | 11325.322989009821 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 259.689731 | 0.713427 | 0.80966015 | 0.8446442999999999 | 11084.849562934081 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 260.230432 | 0.9426239999999999 | 1.9056803499999995 | 2.1376320799999995 | 15589.150738338227 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 260.270592 | 0.9748785 | 2.6397493999999995 | 2.8285339899999995 | 12804.237485570426 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 260.542173 | 0.9040975 | 1.0076789 | 1.0388659999999998 | 17434.879417758715 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 254.935144 | 1.00241 | 3.11232195 | 4.202048809999996 | 12028.368787501635 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 257.762547 | 1.5256975000000002 | 1.69112985 | 1.72346482 | 21274.83151796082 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 260.018376 | 1.5260015 | 1.73163075 | 1.7950199699999998 | 20977.157422632114 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 260.7472 | 1.498919 | 1.6732406 | 1.70103314 | 21312.9558741357 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 258.259306 | 1.520531 | 1.7370250999999999 | 1.9633580799999997 | 20749.289524874206 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 264.853142 | 1.8236555 | 2.0733033499999998 | 2.15192625 | 35004.42182419812 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 261.108274 | 1.8763139999999998 | 2.08713255 | 2.2567554100000002 | 33944.53833704116 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 257.03714 | 1.9162195 | 2.1999351 | 2.4316679 | 32973.10389067685 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 257.91059 | 1.8287705 | 2.11098595 | 2.2685932999999996 | 34688.46307571913 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 261.470084 | 1.817167 | 2.0463547 | 2.18145727 | 71138.48520021615 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 257.588383 | 1.9434545 | 2.1720259499999996 | 2.4008871399999996 | 65260.602512802405 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 259.66501 | 1.985977 | 2.27722845 | 2.32565243 | 64057.19346518539 | - |
| `separator1_grid_search_6ports_learned_dense_hd32_depth3_stages1_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 259.430944 | 1.8941175000000001 | 2.0880457999999997 | 2.14780694 | 68105.47789065799 | - |
