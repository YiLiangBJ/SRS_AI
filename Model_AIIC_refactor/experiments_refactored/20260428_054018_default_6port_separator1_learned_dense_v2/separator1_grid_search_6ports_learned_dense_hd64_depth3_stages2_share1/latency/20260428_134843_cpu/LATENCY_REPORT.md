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

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`122390.093` samples/s, p50=`1.044` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.533` ms, throughput=`1870.840` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`223785.120` samples/s, p50=`0.570` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.157` ms, throughput=`6366.226` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`139187.698` samples/s, p50=`0.808` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.475` ms, throughput=`2083.855` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`303538.111` samples/s, p50=`0.421` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.056` ms, throughput=`17910.600` samples/s

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`38389.366` samples/s, p50=`3.337` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.689` ms, throughput=`1447.923` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `78,624`
- MACs / sample: `153,600`
- FLOPs / sample estimate: `311,064`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.5536004999999999 | 0.57518945 | 0.6545251299999998 | 1792.3281044431167 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.5527029999999999 | 0.6329351999999997 | 0.68428172 | 1782.6978972044337 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.5332835 | 0.54173105 | 0.55637253 | 1870.8403268462816 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.5509795 | 0.56938545 | 0.6632853799999999 | 1799.340433770597 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.5713735 | 0.6150208999999999 | 0.6937417699999999 | 3460.876795139489 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.5682590000000001 | 0.6193972499999998 | 0.6931027 | 3474.7370631716926 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.5686035 | 0.5952608999999999 | 0.69153622 | 3484.143072572401 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.5780285000000001 | 0.6300906499999998 | 0.70379814 | 3419.6861070122377 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.5994305 | 0.6766797499999998 | 0.7064737 | 6568.065984367215 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.60518 | 0.70619635 | 0.71664692 | 6450.888322801936 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.5969265 | 0.6857412999999999 | 0.70750667 | 6611.095341812305 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.5951645 | 0.6897120999999999 | 0.70758609 | 6619.997449976982 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.6413610000000001 | 0.6502157 | 0.6967623599999999 | 12432.743906269296 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.6251705000000001 | 0.6941202499999998 | 0.75614311 | 12636.077923154451 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.622987 | 0.74600925 | 0.75367778 | 12489.139132382 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.6244620000000001 | 0.74119735 | 0.74982109 | 12564.997554380288 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.6523945 | 0.7434439999999999 | 0.80007686 | 24069.686315122675 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.654914 | 0.7328143499999997 | 0.81457606 | 24056.168989535356 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.647877 | 0.79662035 | 0.8060589300000001 | 23948.192396424965 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.6578615 | 0.80402485 | 0.81074775 | 23349.987712068967 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.7171225 | 0.7454413 | 0.7492189499999999 | 44440.44850744949 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.7192885 | 0.74344395 | 0.74910327 | 44380.76544282823 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.845186 | 0.90317395 | 0.91680144 | 37403.327301813035 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.714777 | 0.7412908 | 0.8089825799999997 | 44386.25899071417 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.8167735 | 0.8464652500000001 | 0.85109798 | 77993.96214616668 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.9728680000000001 | 1.05141335 | 1.11047947 | 64598.481677286654 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.9803345 | 1.0797701 | 1.08223656 | 64282.83935051512 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 1.0036875 | 1.08913205 | 1.0978760600000002 | 62646.851806777304 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 1.0437205 | 1.06337325 | 1.07798904 | 122390.09340926669 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 1.3483695 | 1.42685565 | 1.4566152399999999 | 93477.90762849106 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 1.307253 | 1.3737225999999998 | 1.39760478 | 96966.28992687362 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 1.2024075 | 1.30517565 | 1.3412201199999998 | 105011.96512893926 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.5842240000000001 | 0.58936425 | 0.59986608 | 1709.5412196544041 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.5793305 | 0.58532545 | 0.5865307200000001 | 1724.9826371872653 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.5820894999999999 | 0.589043 | 0.60380753 | 1715.5172792390022 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.5800190000000001 | 0.5844324 | 0.58452903 | 1724.774670262753 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.9387315 | 0.9538038 | 0.96285193 | 2126.633451204119 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.9256905 | 0.9332875 | 0.93746613 | 2160.7276258923102 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.956135 | 0.97478955 | 0.98049606 | 2087.647500385536 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.940023 | 0.9610978 | 0.96449237 | 2121.421736965385 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 1.274059 | 1.3325685999999999 | 1.3627024699999999 | 3129.508741922777 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 1.3446384999999998 | 1.43027845 | 1.44087065 | 2949.367384291009 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 1.332561 | 1.3742341 | 1.41331878 | 2989.7678931068285 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 1.3783115000000001 | 1.4277121 | 1.4376984499999999 | 2894.856468239792 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.6156515 | 1.6705960499999999 | 1.72929213 | 4927.025152993061 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.6433849999999999 | 1.7168097999999998 | 1.73034132 | 4844.497269380947 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.6494375 | 1.70842755 | 1.7277151499999999 | 4825.237090135465 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.6721065 | 1.70913505 | 1.72461037 | 4791.589284995333 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.58651 | 1.6328165000000001 | 1.64638465 | 10072.882720658426 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.6836965 | 1.76772285 | 1.77701483 | 9403.88568791722 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.7466145000000002 | 1.81104035 | 1.82559803 | 9134.053272561736 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.7697265 | 1.8364462000000001 | 1.8690465699999999 | 8992.131345544767 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.6522125 | 1.69565225 | 1.7675810299999999 | 19317.19253920282 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.809981 | 1.87097765 | 1.88681555 | 17654.991626678846 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.7905545 | 1.8307537 | 1.8403208199999999 | 17903.15516478836 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.802428 | 1.8509050500000002 | 1.8852372899999998 | 17761.794728115045 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.7682645 | 1.807307 | 1.81407568 | 36176.74687816438 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 2.145617 | 2.24991335 | 2.26305236 | 29688.367239536186 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 2.2042545000000002 | 2.3084736 | 2.35506948 | 28960.41084686848 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 2.2502525 | 2.4204809999999997 | 2.4908531099999998 | 28264.489839295704 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.9252565000000001 | 1.99097745 | 2.0630508599999997 | 66130.36593155324 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 2.5472080000000004 | 2.6213183 | 2.63673557 | 50175.91912444652 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 2.649795 | 2.72183455 | 2.7467989200000003 | 48240.29104030888 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 2.5827545 | 2.9445467 | 2.98604029 | 48434.50521980933 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 223.115008 | 0.1565745 | 0.1584673 | 0.16001896 | 6391.983532204987 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 221.349029 | 0.156529 | 0.16044285 | 0.16196127 | 6366.225797041921 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 221.550965 | 0.159301 | 0.1629958 | 0.16398428 | 6258.53899414013 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 222.630236 | 0.15735549999999998 | 0.1602229 | 0.161494 | 6343.994204126895 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 219.620543 | 0.1764635 | 0.1804834 | 0.18266014 | 11307.612284589984 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 218.865062 | 0.176976 | 0.1801777 | 0.18218574 | 11278.611016496096 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 222.495662 | 0.17387999999999998 | 0.17645235 | 0.17806913 | 11482.36928126339 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 219.206417 | 0.1731785 | 0.1794212 | 0.18167601 | 11481.021985468242 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 220.159554 | 0.17873450000000002 | 0.18411725 | 0.18575175 | 22306.628024151385 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 223.219541 | 0.18448350000000002 | 0.18737295 | 0.18928451 | 21664.336667258103 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 219.720901 | 0.180092 | 0.18432669999999998 | 0.18621232 | 22162.798833483244 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 222.799017 | 0.176051 | 0.18218765 | 0.182991 | 22631.633501516546 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 219.416993 | 0.20591500000000001 | 0.21070785 | 0.21204808 | 38793.94286893622 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 223.949571 | 0.20621 | 0.20899935 | 0.21079787 | 38754.30487662858 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 224.189007 | 0.2031865 | 0.20719405 | 0.20862613000000002 | 39301.54859891944 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 222.283082 | 0.2022045 | 0.20497454999999998 | 0.20577514 | 39541.04315796123 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 220.732608 | 0.2359205 | 0.238416 | 0.24220013999999998 | 67774.939419794 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 223.8901 | 0.22903400000000002 | 0.23222665 | 0.23569073999999998 | 69890.01757384492 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 222.365158 | 0.229284 | 0.2327292 | 0.23415379 | 69644.04233522045 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 220.240661 | 0.2331855 | 0.2364443 | 0.2381566 | 68530.98354296961 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 223.769921 | 0.2815445 | 0.28643945 | 0.28811748 | 113557.31709766519 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 220.999299 | 0.27602 | 0.27911005 | 0.28031829 | 115866.92538785547 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 221.651219 | 0.39917749999999996 | 0.4365288999999999 | 0.44923903 | 79455.08510806612 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 221.421444 | 0.2756465 | 0.2840305 | 0.28553681 | 115569.33463050281 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 224.277936 | 0.370821 | 0.37656239999999996 | 0.37877427 | 172407.01200868716 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 226.887828 | 0.5263665 | 0.5792430000000001 | 0.5924817099999999 | 120095.09580042087 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 223.685224 | 0.5529785 | 0.61078615 | 0.63176965 | 113685.07139156033 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 223.350425 | 0.5474965 | 0.6002441999999999 | 0.60596804 | 114977.91831113535 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 225.897508 | 0.5699479999999999 | 0.5817490999999999 | 0.60187834 | 223785.12040845817 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 227.142742 | 0.9477949999999999 | 1.0394663499999999 | 1.04996096 | 132380.96262471733 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 229.965633 | 0.956661 | 1.03461505 | 1.05879727 | 131415.13505235693 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 226.731513 | 0.789455 | 0.84061955 | 0.84965345 | 160474.10871865108 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 221.737915 | 0.2256355 | 0.2272281 | 0.22908234 | 4429.311946252958 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 219.152097 | 0.222282 | 0.23118879999999997 | 0.2340856 | 4468.219343100265 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 217.135834 | 0.23373850000000002 | 0.2402033 | 0.24183748 | 4262.652725518839 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 218.623125 | 0.225275 | 0.22714700000000002 | 0.23101207 | 4432.358879233734 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 223.518724 | 0.428589 | 0.46144345 | 0.5204232899999998 | 4593.097704052843 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 221.489993 | 0.43026600000000004 | 0.43622855 | 0.43717148 | 4643.984055345144 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 219.036845 | 0.4174645 | 0.42215284999999997 | 0.42295002 | 4787.920727162587 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 224.668636 | 0.438472 | 0.44252385 | 0.46001877999999996 | 4552.090873208018 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 231.791858 | 0.6823300000000001 | 0.6899464 | 0.70601291 | 5856.479874338683 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 229.26518 | 0.7300665 | 0.73848455 | 0.74181955 | 5476.917517430495 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 232.839424 | 0.7221835 | 0.7274747500000001 | 0.74194156 | 5536.481913988373 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 232.936087 | 0.741133 | 0.7508004 | 0.7546212299999999 | 5401.188974531341 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 229.899138 | 0.8412445 | 0.8496578499999999 | 0.86507474 | 9503.786403541035 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 231.965455 | 0.8885655 | 0.89496245 | 0.89689125 | 8998.48503755529 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 233.23349 | 0.941138 | 0.9782947999999999 | 0.98420199 | 8459.084676283517 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 238.521558 | 0.91655 | 1.055372 | 1.05806916 | 8528.771373127714 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 230.585788 | 0.8995365 | 0.9132177 | 0.9224448 | 17767.061475942668 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 232.159033 | 0.988491 | 1.0262200499999998 | 1.1834472299999999 | 16040.799774225743 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 235.343444 | 1.0043075 | 1.0752015999999998 | 1.17499207 | 15799.047491225801 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 239.131973 | 1.0384440000000001 | 1.07916805 | 1.23781254 | 15255.627877402116 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 233.568852 | 0.9358295 | 0.9950948999999999 | 1.19208495 | 33722.463158683226 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 237.137314 | 1.1347285 | 1.2755496 | 1.28664742 | 27486.742328162465 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 231.804759 | 1.113951 | 1.2443572 | 1.27250676 | 28096.428346908007 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 236.428487 | 1.1641465000000002 | 1.1817077999999999 | 1.18506624 | 27473.88937360271 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 235.501479 | 1.1238895 | 1.2773999500000002 | 1.28332626 | 55231.03296426231 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 239.54142 | 1.5662055000000001 | 1.6373938000000001 | 1.66179093 | 40633.30459034949 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 239.7833 | 1.499172 | 1.5672442 | 1.58557954 | 42355.19391220326 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 242.21209 | 1.6046855 | 1.64281245 | 1.67308001 | 39844.16895715645 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 237.174928 | 1.3338510000000001 | 1.4455869 | 1.45246766 | 93792.59915134404 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 237.816826 | 1.9544865 | 2.0424056 | 2.0677179 | 65121.731594983336 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 243.904824 | 2.0279795 | 2.1369769 | 2.14782717 | 62507.48258517117 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 241.925651 | 2.0471624999999998 | 2.1325886 | 2.15638369 | 62462.22072616529 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 496.871786 | 0.48174649999999997 | 0.5204414999999998 | 0.5869400499999998 | 2052.324253112001 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 502.114845 | 0.48430249999999997 | 0.49730425 | 0.56999919 | 2046.775534384437 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 506.716871 | 0.49065 | 0.4958665 | 0.5001368900000001 | 2037.726384778738 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 537.952026 | 0.4753755 | 0.5054611499999999 | 0.5658699399999999 | 2083.855339095777 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 494.826711 | 0.48590849999999997 | 0.49291805 | 0.49636026 | 4115.349289859442 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 512.037529 | 0.4818835 | 0.5067564499999999 | 0.5736616999999999 | 4105.984324337285 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 505.912526 | 0.493253 | 0.5060297 | 0.5736374399999999 | 4027.174082009695 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 534.318858 | 0.48777800000000004 | 0.49729035 | 0.5389622399999999 | 4087.9215449367434 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 496.837802 | 0.5059830000000001 | 0.5422070999999998 | 0.58503965 | 7835.363038422152 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 503.259455 | 0.5040450000000001 | 0.58715365 | 0.60645067 | 7751.259899473135 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 505.592379 | 0.499007 | 0.5581898999999999 | 0.6064767099999999 | 7909.832340820755 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 538.254832 | 0.5032814999999999 | 0.5590885999999998 | 0.61505744 | 7841.094265478438 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 493.376636 | 0.5111755 | 0.5689149999999998 | 0.63222139 | 15447.21880355961 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 510.252268 | 0.5179345 | 0.5742743499999998 | 0.6410481399999999 | 15242.170268607433 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 505.310962 | 0.5122504999999999 | 0.5744898499999997 | 0.63495034 | 15382.03889310079 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 541.013684 | 0.506634 | 0.51231245 | 0.60290156 | 15680.324595263384 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 502.649262 | 0.5519164999999999 | 0.7294979499999995 | 3.147000779999992 | 23929.40156515037 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 503.63324 | 0.5330775000000001 | 0.54037855 | 0.6081434099999999 | 29823.435568816047 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 512.491262 | 0.5274220000000001 | 0.5988622999999997 | 0.66821285 | 29850.109169311756 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 527.319629 | 0.5306215 | 0.6538634 | 0.68626505 | 29064.128874125763 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 494.098555 | 0.5844955000000001 | 0.6864153999999999 | 0.71157972 | 53310.424067097556 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 505.663757 | 0.586085 | 0.70506695 | 0.71523779 | 53062.386177036155 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 519.857643 | 0.7025865 | 0.7870233999999999 | 0.83258541 | 44652.39810448337 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 538.504672 | 0.5876319999999999 | 0.6961080999999999 | 0.72083747 | 53321.0215094668 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 488.30062 | 0.6724025 | 0.7931254499999999 | 0.81344671 | 93093.1785333414 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 510.494029 | 0.7972595 | 0.8390272999999999 | 0.86152664 | 79502.81128151847 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 514.313129 | 0.8275085 | 0.8920425499999999 | 0.9661867599999999 | 76258.41154110052 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 532.628801 | 0.8336045000000001 | 0.9569713999999999 | 0.99981487 | 75515.60046111714 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 497.891021 | 0.8077160000000001 | 0.9800525999999999 | 3.151657499999992 | 139187.69841722926 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 507.949243 | 1.314221 | 1.3740929499999999 | 1.4060417 | 96676.83646563262 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 514.929595 | 1.1366015 | 1.23196825 | 1.29039972 | 111497.13898341368 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 543.368932 | 1.0396239999999999 | 1.176924 | 1.21477072 | 120646.34242044727 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 501.877532 | 0.5561255 | 0.59578795 | 0.61046324 | 1786.2792603260516 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 508.348311 | 0.543126 | 0.55242635 | 0.55519351 | 1839.6701692147017 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 502.53668 | 0.548583 | 0.5926334499999999 | 0.60591115 | 1810.7144318648077 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 506.226487 | 0.553177 | 0.59896675 | 0.60929125 | 1793.5185253604595 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 492.965027 | 0.831592 | 0.90315195 | 0.91276357 | 2372.4473830315824 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 503.105245 | 0.8331189999999999 | 0.9109839999999999 | 0.9328924 | 2371.7844176989383 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 514.563847 | 0.81118 | 0.8640992 | 0.9033866899999999 | 2442.883789135548 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 507.638518 | 0.8514425 | 0.92818945 | 0.9402384 | 2324.556049893062 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 489.57784 | 1.0892095 | 1.1150642 | 1.12596962 | 3671.2361074001033 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 501.003596 | 1.166302 | 1.1820001 | 1.19045681 | 3432.5878702161867 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 504.530918 | 1.1614485 | 1.1730245499999998 | 1.17672472 | 3442.58823147082 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 503.937127 | 1.1522985000000001 | 1.16610225 | 1.17810187 | 3466.9845475458624 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 493.024934 | 1.43933 | 1.6955261 | 1.70819568 | 5416.791941265292 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 509.248832 | 1.5866755000000001 | 1.66869365 | 1.81409545 | 4985.553734859029 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 503.925629 | 1.614773 | 1.6678387 | 1.7203960099999998 | 4928.708267995604 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 502.570989 | 1.685663 | 1.7142826 | 1.7389765799999999 | 4750.475400856693 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 491.911057 | 1.5282645000000001 | 1.6573937 | 1.65989648 | 10380.514362791087 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 514.639226 | 1.6320655 | 1.68110955 | 1.6975794100000001 | 9776.719643410772 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.267616 | 1.631051 | 1.6901969 | 1.70429379 | 9773.759101248117 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 503.000103 | 1.6854710000000002 | 1.72581795 | 1.74066938 | 9492.738547447014 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 492.213224 | 1.4947455 | 1.6463027000000001 | 1.65268955 | 21134.95808059391 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 506.02029 | 1.6587815 | 1.7358536999999998 | 1.74520375 | 19209.77516302856 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 506.994048 | 1.636151 | 1.6814738 | 1.68869997 | 19494.785327688456 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.624125 | 1.7466215 | 1.8045591 | 1.8811067399999999 | 18265.355824485778 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.808954 | 1.545053 | 1.6099246 | 1.6746489199999999 | 41214.89400656231 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 499.423856 | 1.9132045 | 2.0316239 | 2.0606061799999997 | 33233.68179229495 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 501.258864 | 1.9631455 | 2.05466595 | 2.0835375800000002 | 32454.898568451954 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 503.617031 | 2.0405569999999997 | 2.10716745 | 2.1258133 | 31338.82690564037 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 492.374722 | 1.6522934999999999 | 1.6963313 | 1.7013268899999998 | 77109.59037879544 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 511.659907 | 2.0848885 | 2.17615455 | 2.22902729 | 61236.97483565072 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 502.689347 | 2.3442975 | 2.4396378 | 2.4726050600000002 | 54546.28358698065 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 517.738867 | 2.389997 | 2.4807557499999997 | 2.52710817 | 53542.68612847377 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 12.695263 | 0.055804 | 0.0574391 | 0.06275424999999998 | 17910.59960030706 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 13.057105 | 0.057370500000000005 | 0.06131535 | 0.06658167999999999 | 17266.838966697105 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 13.380304 | 0.0571535 | 0.060626 | 0.06340196 | 17350.730795430372 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 13.970929 | 0.0565455 | 0.06060784999999999 | 0.06642001 | 17491.12849962499 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 12.83152 | 0.056638999999999995 | 0.05889165 | 0.06227679999999999 | 35008.79595998495 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 13.171511 | 0.061021 | 0.0641806 | 0.06531473 | 32436.61237210244 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 13.291808 | 0.060784000000000005 | 0.06296425 | 0.06604774 | 32701.88179708613 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.187609 | 0.058202500000000004 | 0.0604781 | 0.0629514 | 34119.911015272075 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 12.956458 | 0.0629435 | 0.066833 | 0.06740681 | 63120.91252640821 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 13.090128 | 0.0642415 | 0.0684756 | 0.07205389 | 61405.994637414486 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 13.546801 | 0.06448999999999999 | 0.067095 | 0.06852725 | 61622.26782270041 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 14.042767 | 0.0659895 | 0.0705659 | 0.07362642 | 59838.63912573354 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 13.176976 | 0.0797205 | 0.08359005 | 0.08574619 | 100384.49772240123 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 13.046259 | 0.08033199999999999 | 0.08561935 | 0.08691198 | 99045.52305569305 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 13.257953 | 0.080788 | 0.08444109999999999 | 0.08741848999999999 | 99032.84523345013 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 13.878149 | 0.080525 | 0.08374435 | 0.08602433999999999 | 99289.45980328275 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 13.02263 | 0.10052 | 0.10464459999999999 | 0.10627025 | 158264.09608846327 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 13.258355 | 0.148326 | 0.1654697 | 0.17197752 | 106196.23152052828 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 13.212295 | 0.1479425 | 0.1600139 | 0.16205049000000002 | 107250.95557249823 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 14.349005 | 0.1484435 | 0.1576324 | 0.15956087 | 107290.38141730594 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 12.701444 | 0.148395 | 0.1522929 | 0.15379268 | 215482.81753480656 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 13.09256 | 0.2800155 | 0.297648 | 0.30038119 | 114473.45074493598 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 13.1569 | 0.306963 | 0.31772619999999996 | 0.32424710999999995 | 103869.33393594858 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 13.955619 | 0.303681 | 0.3148003 | 0.31816171 | 105051.48606160468 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 12.765693 | 0.2410565 | 0.2486766 | 0.3085982499999998 | 263085.9571704283 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 13.183887 | 0.554318 | 0.6275787 | 0.65040662 | 114380.54392666835 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 13.331884 | 0.4877405 | 0.6237116999999999 | 0.64736141 | 129998.24989856074 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.166162 | 0.496062 | 0.61089055 | 0.64167844 | 125706.5592504118 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 12.935726 | 0.4212025 | 0.4267008 | 0.42767419 | 303538.1113678485 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 13.154184 | 1.0601530000000001 | 1.10932165 | 1.11619419 | 119789.30558513895 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 13.179048 | 1.005887 | 1.069173 | 1.0804648399999999 | 126347.33890294778 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 13.982888 | 0.9732464999999999 | 1.0458003999999999 | 1.0853524700000001 | 130401.46290881166 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 304.155292 | 0.7110535 | 0.780925 | 0.82824709 | 1404.8545930540843 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 296.628624 | 0.7236640000000001 | 0.81155155 | 0.84065917 | 1366.6191839386604 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 292.113552 | 0.6894990000000001 | 0.7519775 | 0.77681723 | 1447.923295742156 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 300.03469 | 0.691194 | 0.8027078999999999 | 0.82383003 | 1427.2158589485994 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 300.90404 | 1.027508 | 1.1334629 | 1.16595414 | 1935.3865772115505 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 294.718217 | 1.036 | 1.14717805 | 1.2747426699999997 | 1922.8174906247266 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 295.637396 | 1.01475 | 1.1501202499999998 | 1.5234221699999986 | 1969.9255019483453 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 300.38202 | 1.014642 | 1.1374308 | 1.18095996 | 1949.104781046827 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 294.654574 | 1.112662 | 1.2282761500000001 | 1.30302379 | 3588.6852128602623 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 297.008779 | 1.1111119999999999 | 1.2835434 | 1.3198672599999999 | 3534.0773496794777 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 303.640192 | 1.1454865 | 1.2304401 | 1.2807489799999998 | 3500.8985318649425 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 297.333405 | 1.177095 | 1.9718782999999998 | 3.777797069999999 | 3070.916128750923 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 295.331881 | 1.313791 | 1.4721908499999998 | 1.5228526900000001 | 6076.5047750086 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 298.268796 | 1.3419965 | 2.17068655 | 9.453295189999972 | 4594.160319173775 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 293.892688 | 1.2779215000000002 | 1.4748498499999998 | 1.59461898 | 6204.16296543107 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 294.879028 | 1.31495 | 1.43075495 | 1.46845857 | 6115.17931058722 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 294.50521 | 2.6096515 | 2.9109743999999997 | 2.999187 | 6146.28343575308 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 298.391087 | 2.5969995 | 2.84012895 | 2.89134924 | 6116.3799859477695 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 303.30303 | 2.4864875 | 2.7472361000000003 | 2.79199893 | 6363.4117062785335 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 295.328208 | 2.4647955 | 2.74434845 | 2.77345509 | 6504.839169396436 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 293.950734 | 2.7843625000000003 | 3.0125385 | 3.13966409 | 11524.99920351451 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 299.487175 | 2.616969 | 2.93647395 | 3.008234 | 12125.621346204518 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 296.754091 | 2.680815 | 2.97571735 | 3.01117547 | 11931.169382113818 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 294.710005 | 2.5506905 | 2.82951565 | 2.9378656 | 12534.294024298826 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 285.319033 | 3.360337 | 3.65940255 | 3.73685936 | 19011.711921541704 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 299.894114 | 3.287692 | 3.59006605 | 3.61983247 | 19532.306251424485 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 304.760605 | 3.411899 | 3.76081135 | 3.84703382 | 18586.83535691211 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 288.832656 | 3.198001 | 3.41592845 | 3.6103558999999996 | 20050.12996149629 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 294.960692 | 3.426725 | 3.71439695 | 3.9241716099999993 | 37063.6048638893 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 295.166521 | 3.5488895 | 3.84557605 | 6.458986559999991 | 34931.79346273645 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 293.22492 | 3.8876495 | 7.439048749999985 | 14.309770649999985 | 29355.880472046778 | - |
| `separator1_grid_search_6ports_learned_dense_hd64_depth3_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 299.816999 | 3.337371 | 3.6517253 | 3.7629032799999997 | 38389.36622233515 | - |
