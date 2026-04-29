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

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1::pytorch::eager

- Runtime backend: `pytorch`
- Execution mode: `eager`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`197192.800` samples/s, p50=`0.643` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`bf16`, p50=`0.389` ms, throughput=`2567.493` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1::pytorch::jit

- Runtime backend: `pytorch`
- Execution mode: `jit`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`375348.539` samples/s, p50=`0.340` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.113` ms, throughput=`8827.754` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1::pytorch::compile

- Runtime backend: `pytorch`
- Execution mode: `compile`
- Best throughput config: threads=`8`, batch=`128`, precision=`fp32`, throughput=`245849.617` samples/s, p50=`0.521` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.359` ms, throughput=`2785.315` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`512110.698` samples/s, p50=`0.249` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.046` ms, throughput=`21714.084` samples/s

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1::openvino::openvino

- Runtime backend: `openvino`
- Execution mode: `openvino`
- Best throughput config: threads=`4`, batch=`128`, precision=`fp32`, throughput=`48664.425` samples/s, p50=`2.615` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.546` ms, throughput=`1813.621` samples/s

## Run References

### separator1_grid_search_6ports_learned_dense_depth2_stages2_share1

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share1`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share1/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share1/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260428_054018_default_6port_separator1_learned_dense_v2/separator1_grid_search_6ports_learned_dense_depth2_stages2_share1/MODEL_COMPLEXITY.md`
- Trainable parameters: `28,704`
- MACs / sample: `55,296`
- FLOPs / sample estimate: `112,920`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 1 | ok | 0.0 | 0.403087 | 0.47235969999999994 | 0.48127747 | 2438.773850842445 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 2 | ok | 0.0 | 0.39893 | 0.43451625 | 0.43594005 | 2478.336120470134 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 4 | ok | 0.0 | 0.395684 | 0.4015401 | 0.40599942 | 2523.670769987094 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 1 | 8 | ok | 0.0 | 0.3942315 | 0.40974659999999996 | 0.43399860999999995 | 2523.5067174234714 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 1 | ok | 0.0 | 0.422018 | 0.48262495 | 0.48972658 | 4677.2896958419315 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 2 | ok | 0.0 | 0.42376349999999996 | 0.43161905 | 0.43777133 | 4716.555974600591 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 4 | ok | 0.0 | 0.42868700000000004 | 0.4368516 | 0.44044745999999996 | 4655.757265204457 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 2 | 8 | ok | 0.0 | 0.434151 | 0.49918015 | 0.5528467899999998 | 4478.684484963601 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 1 | ok | 0.0 | 0.424272 | 0.42734154999999996 | 0.43629041 | 9425.772956976673 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 2 | ok | 0.0 | 0.4298555 | 0.4587805999999999 | 0.4725224 | 9238.88239087494 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 4 | ok | 0.0 | 0.4260045 | 0.4337187 | 0.45764622999999993 | 9351.25657510228 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 4 | 8 | ok | 0.0 | 0.42535999999999996 | 0.48001889999999997 | 0.48958551 | 9293.476030940397 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 1 | ok | 0.0 | 0.434315 | 0.43809005 | 0.45138575999999997 | 18399.534859758747 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 2 | ok | 0.0 | 0.4541015 | 0.46589825 | 0.48563681999999997 | 17562.704893650138 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 4 | ok | 0.0 | 0.444388 | 0.4523705 | 0.45505166 | 17967.734261522528 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 8 | 8 | ok | 0.0 | 0.444848 | 0.51887905 | 0.52312501 | 17727.51357971859 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 1 | ok | 0.0 | 0.460963 | 0.50408975 | 0.54473756 | 34278.550068749915 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 2 | ok | 0.0 | 0.448264 | 0.52195945 | 0.53816191 | 34984.53180791445 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 4 | ok | 0.0 | 0.4512755 | 0.4855306 | 0.5282441999999999 | 34965.719608495834 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 16 | 8 | ok | 0.0 | 0.450169 | 0.5346405999999999 | 0.53748253 | 34576.60728278435 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 1 | ok | 0.0 | 0.493798 | 0.50405535 | 0.5144235699999999 | 64758.63707191556 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 2 | ok | 0.0 | 0.485375 | 0.5034776999999999 | 0.51721384 | 65606.05458595957 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 4 | ok | 0.0 | 0.4874645 | 0.51141735 | 0.5232522199999999 | 65399.48622163625 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 32 | 8 | ok | 0.0 | 0.5086900000000001 | 0.5169115 | 0.51799408 | 62863.02167222318 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 1 | ok | 0.0 | 0.536797 | 0.5631632 | 0.5656199 | 118740.95551069538 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 2 | ok | 0.0 | 0.5478685 | 0.57654955 | 0.5792936200000001 | 116258.02757139051 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 4 | ok | 0.0 | 0.5410254999999999 | 0.5680584 | 0.57485759 | 117789.4844983701 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 64 | 8 | ok | 0.0 | 0.5421805 | 0.5683419 | 0.57347233 | 117571.01371895743 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 1 | ok | 0.0 | 0.6428695 | 0.66756825 | 0.77184593 | 197192.80026902026 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 2 | ok | 0.0 | 0.8255969999999999 | 0.89597955 | 0.90209495 | 153225.58286892014 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 4 | ok | 0.0 | 0.816162 | 0.9102624 | 0.9407415899999999 | 154282.42076928736 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `fp32` | 128 | 8 | ok | 0.0 | 0.667292 | 0.69134535 | 0.70265285 | 190899.7549891129 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 1 | ok | 0.0 | 0.3936765 | 0.399895 | 0.40244061 | 2537.374125310048 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 2 | ok | 0.0 | 0.399438 | 0.4056781 | 0.40962438 | 2501.5860055275043 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 4 | ok | 0.0 | 0.39367399999999997 | 0.4006128 | 0.4050024 | 2537.8367347705307 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 1 | 8 | ok | 0.0 | 0.389143 | 0.39493354999999997 | 0.39670847000000004 | 2567.4934988497116 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 1 | ok | 0.0 | 0.43076749999999997 | 0.4392433 | 0.44050806 | 4638.808672289923 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 2 | ok | 0.0 | 0.43614699999999995 | 0.44011695 | 0.44171071 | 4586.83236017569 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 4 | ok | 0.0 | 0.4323285 | 0.438394 | 0.43907440999999997 | 4618.374266677215 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 2 | 8 | ok | 0.0 | 0.436493 | 0.4406711 | 0.4411624 | 4578.745563825126 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 1 | ok | 0.0 | 0.8182325 | 0.8255622499999999 | 0.8305047799999999 | 4888.753959279613 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 2 | ok | 0.0 | 0.923022 | 0.93606455 | 0.93935313 | 4325.595347908716 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 4 | ok | 0.0 | 0.933181 | 0.9486781 | 0.9609854600000001 | 4280.842992163703 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 4 | 8 | ok | 0.0 | 0.9405 | 0.9722373 | 1.0431279299999998 | 4238.135672936798 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 1 | ok | 0.0 | 1.0579420000000002 | 1.1276835 | 1.2021075099999998 | 7477.603968558843 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 2 | ok | 0.0 | 1.137603 | 1.1533877000000001 | 1.15949025 | 7027.450345441349 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 4 | ok | 0.0 | 1.262115 | 1.3087346499999999 | 1.33903861 | 6344.044814205686 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 8 | 8 | ok | 0.0 | 1.193855 | 1.24033995 | 1.24738929 | 6689.749535944618 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 1 | ok | 0.0 | 1.1388235 | 1.2053277 | 1.2157920899999999 | 13945.191631685735 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 2 | ok | 0.0 | 1.186045 | 1.26635655 | 1.2901688 | 13438.724966028163 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 4 | ok | 0.0 | 1.2857375000000002 | 1.3384152 | 1.3584914799999999 | 12468.141560536287 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 16 | 8 | ok | 0.0 | 1.3140070000000001 | 1.35585295 | 1.37565796 | 12151.806503504058 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 1 | ok | 0.0 | 1.1338295 | 1.1618772499999999 | 1.1754770099999998 | 28091.480608889866 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 2 | ok | 0.0 | 1.3163695 | 1.3989825999999999 | 1.40730759 | 24069.314812797897 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 4 | ok | 0.0 | 1.3328545 | 1.4125638999999999 | 1.6206266899999993 | 23755.329136914846 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 32 | 8 | ok | 0.0 | 1.407273 | 1.4599068 | 1.46477065 | 22753.199092682058 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 1 | ok | 0.0 | 1.2054375 | 1.24403875 | 1.24928762 | 52878.5382455182 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 2 | ok | 0.0 | 1.48018 | 1.53295725 | 1.55049118 | 43223.64365725785 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 4 | ok | 0.0 | 1.4973230000000002 | 1.5427669 | 1.54763987 | 42665.93907018573 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 64 | 8 | ok | 0.0 | 1.57952 | 1.6284960499999999 | 1.63586866 | 40495.89655080251 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 1 | ok | 0.0 | 1.349741 | 1.41694605 | 1.42124325 | 93679.89605864232 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 2 | ok | 0.0 | 1.6483765 | 1.7090273 | 1.71751381 | 77547.26999864425 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 4 | ok | 0.0 | 1.7307744999999999 | 1.7648526 | 1.8323328899999998 | 73842.30580999455 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `eager` | `bf16` | 128 | 8 | ok | 0.0 | 1.8605019999999999 | 1.9161149 | 1.9257326899999998 | 68722.9784470136 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 1 | ok | 167.585522 | 0.1205675 | 0.13763095 | 0.15440636999999993 | 8080.021951803638 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 2 | ok | 164.491098 | 0.11626800000000001 | 0.11868675 | 0.11941959 | 8581.18807921947 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 4 | ok | 163.631529 | 0.11531949999999999 | 0.1208765 | 0.12309924 | 8612.245200266567 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 1 | 8 | ok | 164.480456 | 0.112932 | 0.11513125 | 0.11675003 | 8827.753751574652 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 1 | ok | 163.032792 | 0.125443 | 0.1284446 | 0.13093863 | 15877.65567684144 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 2 | ok | 166.6138 | 0.130436 | 0.13315055 | 0.13500021 | 15286.172434139527 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 4 | ok | 166.073554 | 0.12606 | 0.12806105 | 0.1305846 | 15838.716253041233 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 2 | 8 | ok | 163.866214 | 0.1266545 | 0.1299835 | 0.13352911 | 15726.593795921655 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 1 | ok | 164.588535 | 0.13142399999999999 | 0.13359415 | 0.14239678999999997 | 30291.124973003032 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 2 | ok | 165.309377 | 0.133864 | 0.13920925 | 0.15496204999999996 | 29599.994553601002 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 4 | ok | 165.549803 | 0.1326815 | 0.13558309999999998 | 0.13671601 | 30092.066678001345 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 4 | 8 | ok | 166.023212 | 0.1373585 | 0.14405849999999998 | 0.15159774999999998 | 28867.27692090077 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 1 | ok | 167.283243 | 0.14285350000000002 | 0.14870165 | 0.14991349 | 55788.682512605104 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 2 | ok | 164.573773 | 0.14343899999999998 | 0.14640485 | 0.16445794999999994 | 55379.50257853886 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 4 | ok | 165.511949 | 0.146644 | 0.1504271 | 0.15580246 | 54378.251309632215 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 8 | 8 | ok | 164.015266 | 0.1425005 | 0.14539315 | 0.16204003999999994 | 55812.83723162746 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 1 | ok | 166.029635 | 0.161792 | 0.1663382 | 0.19392409999999988 | 98084.00257274338 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 2 | ok | 166.391125 | 0.1602515 | 0.16375045 | 0.16553046 | 99517.20472094692 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 4 | ok | 164.246627 | 0.158975 | 0.16121495 | 0.16233154 | 100527.27813975276 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 16 | 8 | ok | 167.765803 | 0.1623875 | 0.16441275 | 0.16619788 | 98496.1727465276 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 1 | ok | 165.433594 | 0.18462499999999998 | 0.188584 | 0.18967881 | 173006.09662671632 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 2 | ok | 168.056182 | 0.186777 | 0.1902907 | 0.19180124999999998 | 170976.83231863927 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 4 | ok | 165.290664 | 0.1884425 | 0.19434135 | 0.21653185999999994 | 168476.56122490883 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 32 | 8 | ok | 164.198652 | 0.1868215 | 0.19227275 | 0.19402904999999998 | 170489.79373505418 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 1 | ok | 167.882939 | 0.2342255 | 0.2389693 | 0.2579092599999999 | 271662.679167904 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 2 | ok | 167.880563 | 0.236838 | 0.24580695 | 0.24924407 | 268814.15575344197 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 4 | ok | 165.143807 | 0.23579250000000002 | 0.2410711 | 0.26315379999999994 | 269595.98430041486 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 64 | 8 | ok | 166.860586 | 0.2367685 | 0.24196435 | 0.24313033 | 269523.2083907965 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 1 | ok | 167.149932 | 0.3401815 | 0.34380995 | 0.36202407999999997 | 375348.53897633887 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 2 | ok | 168.802785 | 0.524179 | 0.57984425 | 0.58827005 | 240466.16470106246 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 4 | ok | 170.893427 | 0.4933595 | 0.50816725 | 0.51034075 | 259412.01510523737 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `fp32` | 128 | 8 | ok | 169.885478 | 0.346951 | 0.3533638 | 0.35710439 | 368346.8501279545 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 1 | ok | 163.408604 | 0.1359995 | 0.14099615 | 0.14230913 | 7323.046990967315 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 2 | ok | 162.887925 | 0.1333835 | 0.135538 | 0.14055641 | 7473.474024969475 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 4 | ok | 165.153399 | 0.1349125 | 0.14127784999999998 | 0.1605199499999999 | 7333.265378407493 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 1 | 8 | ok | 162.264541 | 0.13539050000000002 | 0.13662465000000001 | 0.13707257 | 7384.6866411609435 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 1 | ok | 163.343663 | 0.1690105 | 0.172716 | 0.17555977 | 11808.696750896133 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 2 | ok | 164.810253 | 0.169107 | 0.17439955000000001 | 0.19027415999999994 | 11743.208902291806 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 4 | ok | 162.667106 | 0.1671525 | 0.17152845 | 0.19237455999999992 | 11865.015515287538 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 2 | 8 | ok | 162.847169 | 0.1699085 | 0.17198534999999998 | 0.17578641999999997 | 11741.872774694948 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 1 | ok | 172.99121 | 0.43074650000000003 | 0.4358927 | 0.43857368999999996 | 9278.91337243591 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 2 | ok | 172.125079 | 0.501718 | 0.50981375 | 0.51096555 | 7963.593952430825 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 4 | ok | 173.294589 | 0.4976225 | 0.5049328 | 0.5075187099999999 | 8039.693251935727 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 4 | 8 | ok | 176.910766 | 0.505362 | 0.51679065 | 0.52148836 | 7896.6835824157915 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 1 | ok | 172.677122 | 0.5837600000000001 | 0.59136 | 0.60523076 | 13697.526360290214 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 2 | ok | 172.967422 | 0.660801 | 0.66821555 | 0.6693409 | 12101.564680881587 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 4 | ok | 174.771304 | 0.6578539999999999 | 0.6637871 | 0.67098412 | 12158.734713430782 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 8 | 8 | ok | 179.095469 | 0.705713 | 0.7201259 | 0.72224842 | 11310.687564820844 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 1 | ok | 175.597211 | 0.6539215 | 0.7747262500000001 | 0.78080138 | 24021.09821098067 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 2 | ok | 174.854124 | 0.715688 | 0.8224896999999999 | 0.83851533 | 22063.185875501415 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 4 | ok | 174.531995 | 0.7202155 | 0.8248137 | 0.8430631199999999 | 21929.26911610024 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 16 | 8 | ok | 176.899445 | 0.763873 | 0.87398285 | 0.8936405399999999 | 20487.21044913292 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 1 | ok | 173.463244 | 0.6815359999999999 | 0.68889365 | 0.69339308 | 46941.66679355585 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 2 | ok | 175.894881 | 0.7761225 | 0.78696625 | 0.79209653 | 41171.49685950255 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 4 | ok | 178.752921 | 0.8174950000000001 | 0.8683291499999998 | 0.93503751 | 38892.67309196314 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 32 | 8 | ok | 178.355275 | 0.8638955 | 0.9066765 | 0.97474802 | 36651.562710388556 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 1 | ok | 173.430707 | 0.7452675 | 0.86161725 | 0.91501842 | 83972.44895439474 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 2 | ok | 176.669356 | 1.0303265000000001 | 1.0565136 | 1.11786409 | 61857.50829141908 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 4 | ok | 177.582914 | 1.026084 | 1.07670825 | 1.11738951 | 62034.500138307856 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 64 | 8 | ok | 181.648893 | 1.056839 | 1.1079709999999998 | 1.11663945 | 60326.052889924285 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 1 | ok | 177.544765 | 0.9501025000000001 | 1.0770068 | 1.09755824 | 131953.87923271707 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 2 | ok | 178.36937 | 1.2928555 | 1.3376017 | 1.35580247 | 98723.59492839 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 4 | ok | 178.205601 | 1.3458785 | 1.38796 | 1.42558639 | 94755.45017433449 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `jit` | `bf16` | 128 | 8 | ok | 178.763129 | 1.3698890000000001 | 1.39750495 | 1.40783276 | 94089.69550089979 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 1 | ok | 490.968591 | 0.361527 | 0.36538125 | 0.36768986 | 2765.6519305356733 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 2 | ok | 509.904068 | 0.36319049999999997 | 0.36876 | 0.37098478999999995 | 2749.8507518504434 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 4 | ok | 504.42092 | 0.3614135 | 0.36709915 | 0.36991743 | 2764.8788499209936 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 1 | 8 | ok | 540.20029 | 0.358528 | 0.3629957 | 0.3676513 | 2785.315150053283 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 1 | ok | 495.963992 | 0.373715 | 0.4289132999999999 | 0.44361702999999997 | 5284.259332847463 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 2 | ok | 511.39709 | 0.3662605 | 0.3719794 | 0.37425562 | 5448.421132006471 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 4 | ok | 510.545309 | 0.36790049999999996 | 0.4122974999999999 | 0.44334450000000003 | 5371.462408572338 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 2 | 8 | ok | 543.111438 | 0.3661995 | 0.37127 | 0.37353473 | 5461.151362527229 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 1 | ok | 493.837516 | 0.37165400000000004 | 0.44518474999999996 | 0.45094787 | 10531.6708669603 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 2 | ok | 510.235385 | 0.382986 | 0.4517771 | 0.46158338 | 10279.248110006047 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 4 | ok | 514.093999 | 0.371107 | 0.37533685 | 0.37713805 | 10777.463590764188 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 4 | 8 | ok | 534.883837 | 0.370858 | 0.3754989 | 0.37755542 | 10780.880754834147 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 1 | ok | 493.244344 | 0.377908 | 0.38485145 | 0.38620134 | 21158.83109307156 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 2 | ok | 506.494375 | 0.3746345 | 0.3840799 | 0.38626447 | 21300.440908476583 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 4 | ok | 502.738635 | 0.37565950000000004 | 0.4129672999999998 | 0.45908283 | 21035.87152509753 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 8 | 8 | ok | 532.966317 | 0.3741625 | 0.3786917 | 0.38211989999999996 | 21368.67198832074 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 1 | ok | 493.359582 | 0.3893525 | 0.40641155 | 0.45003378999999993 | 40713.52280830074 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 2 | ok | 505.600727 | 0.39203299999999996 | 0.4394508999999998 | 0.4884426 | 40179.96404998167 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 4 | ok | 507.251197 | 0.3857355 | 0.48037295 | 0.48197436 | 40490.86672826002 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 16 | 8 | ok | 540.272816 | 0.387756 | 0.47808355 | 0.48315956 | 40252.052313781045 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 1 | ok | 496.207268 | 0.410968 | 0.4188896 | 0.5210414099999999 | 77105.86230088134 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 2 | ok | 506.660056 | 0.4107885 | 0.4146313 | 0.41971195 | 77922.1399055983 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 4 | ok | 503.945526 | 0.41059100000000004 | 0.41891975 | 0.52673186 | 77027.50844895483 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 32 | 8 | ok | 550.046502 | 0.41284 | 0.419171 | 0.42558692 | 77427.42727241757 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 1 | ok | 496.080494 | 0.4605405 | 0.58744095 | 0.59583469 | 134793.06926051216 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 2 | ok | 508.706363 | 0.4636 | 0.47389595 | 0.47559872999999997 | 137823.25313441304 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 4 | ok | 511.782823 | 0.4479315 | 0.46425479999999997 | 0.5583557999999998 | 141508.76021035455 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 64 | 8 | ok | 537.727002 | 0.4433355 | 0.45165255 | 0.45257757 | 144175.4665889801 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 1 | ok | 490.224114 | 0.519218 | 0.6463094999999999 | 0.6715234099999999 | 238411.36375844787 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 2 | ok | 510.821114 | 0.8134345000000001 | 0.88537505 | 0.90672705 | 155934.0280161795 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 4 | ok | 518.120518 | 0.6694445 | 0.76896 | 0.7803881199999999 | 187897.11247350872 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `fp32` | 128 | 8 | ok | 534.95871 | 0.52086 | 0.5286465 | 0.53041775 | 245849.61670892333 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 1 | ok | 493.440355 | 0.388034 | 0.3946651 | 0.39778497 | 2570.844108383703 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 2 | ok | 503.889397 | 0.3930185 | 0.41163415 | 0.42269212 | 2524.1022739990763 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 4 | ok | 497.455537 | 0.380676 | 0.3876445 | 0.39071444 | 2621.7890490702953 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 1 | 8 | ok | 499.387589 | 0.387169 | 0.39370140000000003 | 0.39528959999999996 | 2578.0835774477705 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 1 | ok | 494.795741 | 0.4194195 | 0.44536800000000004 | 0.46587399999999995 | 4718.321625054136 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 2 | ok | 506.424695 | 0.4091285 | 0.42991245 | 0.4498053799999999 | 4845.599338129895 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 4 | ok | 502.866765 | 0.410785 | 0.43283745 | 0.4476826 | 4822.256449526888 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 2 | 8 | ok | 500.302011 | 0.41002550000000004 | 0.4336889 | 0.4521669 | 4840.46715735773 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 1 | ok | 489.780324 | 0.750274 | 0.7795046 | 0.78773492 | 5298.595599288515 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 2 | ok | 510.066205 | 0.7865275 | 0.8289077499999999 | 0.8376095 | 5041.8779644194165 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 4 | ok | 501.507974 | 0.7549925 | 0.76199015 | 0.76840533 | 5299.708590223458 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 4 | 8 | ok | 505.640049 | 0.7926375 | 0.8079889 | 0.81845559 | 5045.691766359445 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 1 | ok | 490.156608 | 1.0482815 | 1.22598615 | 1.2346717600000001 | 7519.331967509343 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 2 | ok | 503.585653 | 1.1864599999999998 | 1.2639478000000002 | 1.28164324 | 6641.881930751904 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 4 | ok | 504.030892 | 1.264757 | 1.3121047 | 1.32998175 | 6360.609104012957 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 8 | 8 | ok | 504.227241 | 1.2841974999999999 | 1.3210196 | 1.3530443399999998 | 6218.9630986008515 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 1 | ok | 493.125778 | 1.0112435 | 1.0350375 | 1.17352434 | 15715.254671988063 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 2 | ok | 509.886458 | 1.1993770000000001 | 1.30190305 | 1.3094076399999999 | 13091.343985356283 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 4 | ok | 503.371543 | 1.2827785 | 1.3286813999999998 | 1.3468506 | 12525.419752258158 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 16 | 8 | ok | 505.283563 | 1.3948450000000001 | 1.45255305 | 1.45706966 | 11405.175340670805 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 1 | ok | 496.944232 | 1.0903605 | 1.1317618 | 1.13682835 | 29165.591565238006 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 2 | ok | 502.686523 | 1.2312495 | 1.31058885 | 1.3298973299999999 | 25711.125954644063 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 4 | ok | 507.621502 | 1.2617495 | 1.31892055 | 1.3247517 | 25204.913981535985 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 32 | 8 | ok | 507.17173 | 1.373049 | 1.4194628 | 1.43725857 | 23236.862400827973 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 1 | ok | 490.797544 | 1.1573605 | 1.2191782500000001 | 3.612844999999991 | 50695.58380788757 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 2 | ok | 506.689756 | 1.3214185 | 1.348254 | 1.35188039 | 48587.20331799582 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 4 | ok | 503.053918 | 1.3904795 | 1.4473436 | 1.45104703 | 45959.51716252605 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 64 | 8 | ok | 499.572616 | 1.462248 | 1.5103156 | 1.52375712 | 43753.410373244486 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 1 | ok | 494.756605 | 1.226128 | 1.3173682 | 1.7841056699999989 | 101651.40812851356 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 2 | ok | 509.15979 | 1.455814 | 1.5139414 | 1.54594024 | 87908.58364409703 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 4 | ok | 501.862896 | 1.5392169999999998 | 1.5834966 | 1.5963579799999998 | 83447.34544215162 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `pytorch` | `compile` | `bf16` | 128 | 8 | ok | 505.012775 | 1.612945 | 1.66723195 | 1.66976146 | 79388.14071423898 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 10.43868 | 0.046066499999999996 | 0.0486255 | 0.05026682999999999 | 21714.08371126696 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.385128 | 0.047643000000000005 | 0.05018515 | 0.05394992999999999 | 20885.05866195277 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.268275 | 0.0467665 | 0.04984219999999999 | 0.05331635 | 21308.513177184548 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 10.753811 | 0.0475265 | 0.05032055 | 0.055774889999999994 | 20849.59601822755 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.931625 | 0.046935500000000005 | 0.0500728 | 0.05327803 | 42384.968255778025 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 10.033715 | 0.048547 | 0.05229324999999999 | 0.055814129999999997 | 40672.99158801188 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.449466 | 0.0485335 | 0.05128695 | 0.054524579999999996 | 41033.536298881714 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 11.275622 | 0.048444 | 0.054310199999999996 | 0.058712509999999996 | 40735.769468133425 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 10.107517 | 0.049543500000000004 | 0.052552350000000005 | 0.05423997999999999 | 80792.80362339565 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.102927 | 0.051304 | 0.0540685 | 0.0576158 | 77567.57396165136 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.261834 | 0.051943 | 0.05486765 | 0.06136135999999998 | 76622.6278113321 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.760092 | 0.0505075 | 0.05470194999999999 | 0.056145070000000005 | 78302.18929006146 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.680502 | 0.0560185 | 0.0594023 | 0.06265920999999999 | 140947.10818818133 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.103251 | 0.057119500000000004 | 0.06304939999999999 | 0.06513574 | 137536.68361233224 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.276166 | 0.0581635 | 0.06275384999999999 | 0.06649495 | 135798.5754050447 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.075605 | 0.0588795 | 0.0632746 | 0.06704170999999999 | 134786.7841558135 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.979281 | 0.070948 | 0.07324575 | 0.07385456 | 224844.72363913438 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 10.069145 | 0.0739185 | 0.07833645 | 0.07974498000000001 | 216454.31596377635 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.287548 | 0.073834 | 0.07971645000000001 | 0.08064990999999999 | 217023.48492514182 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 10.962687 | 0.07175300000000001 | 0.07499295 | 0.08009226 | 219857.29063265034 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.982616 | 0.09744900000000001 | 0.10089875 | 0.10397326 | 327637.9616905218 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 10.067084 | 0.0993085 | 0.10554075 | 0.10795139 | 319163.790867926 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.370211 | 0.101709 | 0.10671005 | 0.11011814999999998 | 313603.86109073774 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.019679 | 0.1007985 | 0.1062712 | 0.10764018 | 317258.5057996838 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.810642 | 0.1474455 | 0.154684 | 0.15661044 | 429758.1173586152 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.153443 | 0.2815335 | 0.3183062 | 0.32053146 | 229070.73733002594 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.173699 | 0.2937555 | 0.32044115 | 0.32348091 | 220825.8514199482 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.297659 | 0.297095 | 0.31802215 | 0.32296788 | 219621.34123713936 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.889594 | 0.249475 | 0.2538443 | 0.25508944 | 512110.6978484469 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.01485 | 0.6791205 | 0.6899491 | 0.6915906399999999 | 188874.81312460382 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.180833 | 0.6332765 | 0.67103895 | 0.6901452699999999 | 201113.5342149585 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 11.189723 | 0.6330979999999999 | 0.6957800000000001 | 0.69729329 | 199327.90989082842 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 1 | ok | 280.077671 | 0.575886 | 0.6328366 | 0.6619994099999998 | 1732.5433951876737 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 2 | ok | 272.560537 | 0.546418 | 0.6326649 | 0.63879946 | 1813.620595359409 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 4 | ok | 272.033244 | 0.546574 | 0.6076216999999999 | 0.62334757 | 1826.7903851930016 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 1 | 8 | ok | 270.883004 | 0.5478624999999999 | 0.6137924499999999 | 0.6536645299999999 | 1815.6637236815457 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 1 | ok | 269.721765 | 0.8387640000000001 | 0.9615948 | 1.0142313299999999 | 2345.29094776561 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 2 | ok | 271.668984 | 0.8685594999999999 | 1.00652585 | 1.3971566199999985 | 2298.13941943161 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 4 | ok | 272.81585 | 0.8687469999999999 | 1.5971580999999992 | 4.140103919999992 | 2024.7795354236496 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 2 | 8 | ok | 272.319494 | 0.8748480000000001 | 1.0942051999999998 | 2.464971459999998 | 2152.9068505237638 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 1 | ok | 268.499202 | 1.00217 | 1.1730604999999998 | 1.5677612699999985 | 4102.598438157185 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 2 | ok | 273.240951 | 1.018603 | 1.1427490999999999 | 1.1548862500000001 | 3909.264485764892 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 4 | ok | 270.194367 | 0.9856855 | 1.0977208999999999 | 1.16928753 | 4004.514288957943 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 4 | 8 | ok | 270.809118 | 0.9920835 | 1.1504033999999999 | 1.761623779999998 | 4011.5385487894255 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 1 | ok | 271.774475 | 1.154556 | 1.302594 | 1.3471885700000001 | 6854.655713863639 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 2 | ok | 270.940368 | 1.1384585 | 1.33569195 | 1.40595998 | 6950.79682110173 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 4 | ok | 269.578902 | 1.124687 | 1.2955601 | 1.4258024299999996 | 7021.913953957941 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 8 | 8 | ok | 271.930122 | 1.1342385 | 1.23503595 | 1.26301157 | 7021.556371152237 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 1 | ok | 268.322776 | 1.830128 | 2.03435955 | 2.2145703599999997 | 8751.679953339542 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 2 | ok | 268.354855 | 1.8456480000000002 | 2.06447175 | 2.1094364999999997 | 8707.152808485607 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 4 | ok | 274.122386 | 1.972872 | 2.39814985 | 2.4657083899999996 | 8008.612542013807 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 16 | 8 | ok | 269.807693 | 1.7738245 | 2.0269944 | 2.0691872 | 8978.788812343842 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 1 | ok | 271.595429 | 1.8879885 | 2.2508406999999995 | 2.32004951 | 16936.687212350214 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 2 | ok | 269.263105 | 1.8484185000000002 | 2.1028485 | 2.17569469 | 17171.140648501827 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 4 | ok | 270.325876 | 1.8953965 | 2.0973236 | 2.21669789 | 17357.1249614821 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 32 | 8 | ok | 268.588527 | 1.943042 | 2.2228741 | 2.2696777299999997 | 16446.10605392779 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 1 | ok | 271.137552 | 2.5854885 | 2.8166184000000003 | 2.9178393199999997 | 24490.82540770681 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 2 | ok | 272.886175 | 2.568618 | 2.7922495499999997 | 2.8552889699999997 | 25263.5599177548 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 4 | ok | 270.240657 | 2.543142 | 2.83905085 | 2.94205471 | 25367.815487828244 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 64 | 8 | ok | 269.670914 | 2.4501165 | 2.68589045 | 2.7709542 | 26217.83567243947 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 1 | ok | 268.283608 | 2.617655 | 8.453559049999981 | 18.24430828 | 37079.22817755762 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 2 | ok | 272.432177 | 2.653689 | 2.8820891499999997 | 3.026724089999999 | 47942.53021032995 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 4 | ok | 276.592522 | 2.6148355 | 2.9687538 | 2.98665804 | 48664.425375863866 | - |
| `separator1_grid_search_6ports_learned_dense_depth2_stages2_share1` | `openvino` | `openvino` | `fp32` | 128 | 8 | ok | 268.990163 | 2.6431265 | 2.96030895 | 10.995994139999972 | 43088.02706865264 | - |
