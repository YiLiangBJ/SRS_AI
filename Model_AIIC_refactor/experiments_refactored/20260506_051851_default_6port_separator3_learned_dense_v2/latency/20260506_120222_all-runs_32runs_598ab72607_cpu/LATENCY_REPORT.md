# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32', 'bf16']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128]`
- Thread counts: `[1, 2, 4, 8]`

## Hardware Summary

- Runtime backend: `onnxruntime`
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
- ONNX Runtime version: `1.23.2`
- ONNX Runtime providers: `['CPUExecutionProvider']`

## CPU Thread Scaling Highlights

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`857258.893` samples/s, p50=`0.148` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.017` ms, throughput=`55591.072` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`565835.990` samples/s, p50=`0.226` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.021` ms, throughput=`44504.417` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`421146.087` samples/s, p50=`0.302` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.026` ms, throughput=`38246.066` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`334680.702` samples/s, p50=`0.380` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.030` ms, throughput=`32646.324` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`650835.037` samples/s, p50=`0.196` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`51465.427` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`429432.181` samples/s, p50=`0.298` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.024` ms, throughput=`39474.484` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`315924.557` samples/s, p50=`0.404` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.029` ms, throughput=`33758.921` samples/s

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`248036.001` samples/s, p50=`0.514` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.039` ms, throughput=`25505.728` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`563273.874` samples/s, p50=`0.227` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`54127.199` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`357851.951` samples/s, p50=`0.358` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.025` ms, throughput=`36162.808` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`258760.509` samples/s, p50=`0.494` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.033` ms, throughput=`29664.929` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`200134.340` samples/s, p50=`0.640` ms
- Lowest batch-1 p50 latency: threads=`2`, precision=`fp32`, p50=`0.047` ms, throughput=`20986.227` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`311060.071` samples/s, p50=`0.411` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.031` ms, throughput=`31826.092` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`195221.931` samples/s, p50=`0.653` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.054` ms, throughput=`18477.587` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`142096.814` samples/s, p50=`0.902` ms
- Lowest batch-1 p50 latency: threads=`4`, precision=`fp32`, p50=`0.076` ms, throughput=`13109.527` samples/s

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`111451.724` samples/s, p50=`1.147` ms
- Lowest batch-1 p50 latency: threads=`8`, precision=`fp32`, p50=`0.097` ms, throughput=`10092.575` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1365760.179` samples/s, p50=`0.094` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`66866.195` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`943387.187` samples/s, p50=`0.135` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`51383.237` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`728389.066` samples/s, p50=`0.175` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`45409.095` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`571660.860` samples/s, p50=`0.222` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`38954.159` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1280305.737` samples/s, p50=`0.100` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.016` ms, throughput=`62116.663` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`904582.644` samples/s, p50=`0.140` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.018` ms, throughput=`52833.514` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`696064.516` samples/s, p50=`0.182` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`43214.539` samples/s

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`558677.018` samples/s, p50=`0.227` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.027` ms, throughput=`36147.148` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1119308.295` samples/s, p50=`0.115` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.014` ms, throughput=`66330.856` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`768880.707` samples/s, p50=`0.166` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`49860.838` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`586835.357` samples/s, p50=`0.217` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.023` ms, throughput=`40415.504` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`468644.189` samples/s, p50=`0.271` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.026` ms, throughput=`37891.878` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`1005045.801` samples/s, p50=`0.127` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.017` ms, throughput=`55892.151` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`683782.527` samples/s, p50=`0.187` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.021` ms, throughput=`48161.994` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`518127.332` samples/s, p50=`0.246` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.025` ms, throughput=`40149.775` samples/s

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`128`, precision=`fp32`, throughput=`415417.143` samples/s, p50=`0.306` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.029` ms, throughput=`34446.388` samples/s

## Run References

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `59,200`
- MACs / sample: `58,368`
- FLOPs / sample estimate: `117,784`

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `96,480`
- MACs / sample: `95,232`
- FLOPs / sample estimate: `191,928`

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `133,760`
- MACs / sample: `132,096`
- FLOPs / sample estimate: `266,072`

### separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `171,040`
- MACs / sample: `168,960`
- FLOPs / sample estimate: `340,216`

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `92,224`
- MACs / sample: `91,136`
- FLOPs / sample estimate: `183,576`

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `146,016`
- MACs / sample: `144,384`
- FLOPs / sample estimate: `290,616`

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `199,808`
- MACs / sample: `197,632`
- FLOPs / sample estimate: `397,656`

### separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `253,600`
- MACs / sample: `250,880`
- FLOPs / sample estimate: `504,696`

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `117,824`
- MACs / sample: `116,736`
- FLOPs / sample estimate: `234,776`

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `192,096`
- MACs / sample: `190,464`
- FLOPs / sample estimate: `382,776`

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `266,368`
- MACs / sample: `264,192`
- FLOPs / sample estimate: `530,776`

### separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `340,640`
- MACs / sample: `337,920`
- FLOPs / sample estimate: `678,776`

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `249,408`
- MACs / sample: `247,808`
- FLOPs / sample estimate: `497,432`

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `389,472`
- MACs / sample: `387,072`
- FLOPs / sample estimate: `776,760`

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `529,536`
- MACs / sample: `526,336`
- FLOPs / sample estimate: `1,056,088`

### separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `669,600`
- MACs / sample: `665,600`
- FLOPs / sample estimate: `1,335,416`

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `15,232`
- MACs / sample: `14,592`
- FLOPs / sample estimate: `30,040`

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `24,768`
- MACs / sample: `23,808`
- FLOPs / sample estimate: `48,792`

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `34,304`
- MACs / sample: `33,024`
- FLOPs / sample estimate: `67,544`

### separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `43,840`
- MACs / sample: `42,240`
- FLOPs / sample estimate: `86,296`

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `17,344`
- MACs / sample: `16,640`
- FLOPs / sample estimate: `34,200`

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `27,936`
- MACs / sample: `26,880`
- FLOPs / sample estimate: `55,032`

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,528`
- MACs / sample: `37,120`
- FLOPs / sample estimate: `75,864`

### separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `49,120`
- MACs / sample: `47,360`
- FLOPs / sample estimate: `96,696`

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `29,888`
- MACs / sample: `29,184`
- FLOPs / sample estimate: `59,288`

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `48,672`
- MACs / sample: `47,616`
- FLOPs / sample estimate: `96,504`

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `67,456`
- MACs / sample: `66,048`
- FLOPs / sample estimate: `133,720`

### separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `86,240`
- MACs / sample: `84,480`
- FLOPs / sample estimate: `170,936`

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,208`
- MACs / sample: `37,376`
- FLOPs / sample estimate: `75,800`

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3/MODEL_COMPLEXITY.md`
- Trainable parameters: `61,152`
- MACs / sample: `59,904`
- FLOPs / sample estimate: `121,272`

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4/MODEL_COMPLEXITY.md`
- Trainable parameters: `84,096`
- MACs / sample: `82,432`
- FLOPs / sample estimate: `166,744`

### separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260506_051851_default_6port_separator3_learned_dense_v2/separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5/MODEL_COMPLEXITY.md`
- Trainable parameters: `107,040`
- MACs / sample: `104,960`
- FLOPs / sample estimate: `212,216`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.233296 | 0.017275 | 0.020467299999999997 | 0.032825919999999974 | 54383.4709053869 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.525567 | 0.017727 | 0.020520749999999997 | 0.02707853999999999 | 53551.36593469089 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.94482 | 0.0172235 | 0.02005925 | 0.026369009999999977 | 55591.07207382494 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.367075 | 0.017272 | 0.0177902 | 0.02798122999999999 | 56540.87452901452 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.273848 | 0.0182415 | 0.021367949999999997 | 0.025300659999999992 | 105514.39321837891 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.400515 | 0.0186185 | 0.019133249999999997 | 0.02678274999999997 | 105561.17375580323 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.701405 | 0.019016 | 0.019305549999999998 | 0.02471350999999998 | 104134.016313635 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.744443 | 0.019237999999999998 | 0.020097749999999998 | 0.026806239999999988 | 102386.42274125313 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.171155 | 0.0206755 | 0.02123655 | 0.022932419999999995 | 200158.52555223735 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.492732 | 0.029482500000000002 | 0.042167949999999996 | 0.06500352999999993 | 128572.135718175 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.694635 | 0.0314655 | 0.0437395 | 0.047034409999999985 | 116250.00726562548 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.395383 | 0.0317605 | 0.044534199999999996 | 0.04962589999999999 | 114790.46435612594 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.273222 | 0.0255535 | 0.030553849999999994 | 0.03269977 | 304837.8529355504 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.421701 | 0.051996 | 0.056154800000000005 | 0.060728179999999986 | 153764.0083816761 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.695739 | 0.0486045 | 0.0546542 | 0.06527903999999997 | 165747.32152328416 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.369486 | 0.0501715 | 0.05679235 | 0.058282560000000004 | 160518.60350420137 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.255728 | 0.033759 | 0.03852604999999999 | 0.04097894 | 462281.57917698857 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.455539 | 0.069305 | 0.0730914 | 0.07749004999999999 | 229759.6799907177 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.665185 | 0.067834 | 0.07632984999999999 | 0.07843538 | 233292.67236464575 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.356159 | 0.0682875 | 0.09204984999999997 | 0.09754207 | 223938.83782461332 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.10041 | 0.0511555 | 0.055111850000000004 | 0.05822257999999999 | 624387.0762801984 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.273454 | 0.1150695 | 0.12497934999999999 | 0.12772813 | 273397.62506318046 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.53764 | 0.1030645 | 0.11896785 | 0.14143485999999994 | 303163.85589560855 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.638871 | 0.1840765 | 0.20612825 | 0.21086001 | 174588.0921318821 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.136783 | 0.08411450000000001 | 0.08790099999999999 | 0.09098828999999999 | 759019.9439606088 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.455389 | 0.1654305 | 0.1756619 | 0.17748617 | 402650.0412464636 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.678216 | 0.189096 | 0.2791172499999999 | 0.3190821 | 323184.07413519477 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.745766 | 0.2902465 | 0.32875315 | 0.3746399099999999 | 216192.4350483609 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.109923 | 0.1483555 | 0.15396754999999998 | 0.15503843 | 857258.8932573374 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.451837 | 0.222748 | 0.2432625 | 0.3334641099999997 | 561033.0792993082 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.452063 | 0.3581235 | 0.36907885 | 0.37226812 | 359509.48077056813 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.744359 | 0.391906 | 0.4051413 | 0.41007211 | 327103.46052461775 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.351095 | 0.022192 | 0.023985149999999997 | 0.037392969999999984 | 43773.13957590832 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.216358 | 0.0213615 | 0.026786050000000002 | 0.031563939999999985 | 44504.41661830521 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.402066 | 0.021624499999999998 | 0.026828249999999994 | 0.03319216999999998 | 45066.80703474831 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.254934 | 0.021615 | 0.024482649999999988 | 0.028506989999999992 | 45464.2582733584 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.876246 | 0.022231 | 0.02765155 | 0.02796026 | 85759.39515613785 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.177813 | 0.023099500000000002 | 0.024898999999999994 | 0.02893477 | 85583.46527450897 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.416702 | 0.0235565 | 0.025266249999999997 | 0.03289128999999999 | 83486.04322072456 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.559187 | 0.023835500000000003 | 0.02777695 | 0.029568259999999996 | 82921.62745302904 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.821919 | 0.0259065 | 0.029738699999999993 | 0.03339446 | 151481.87140703935 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.020555 | 0.039049 | 0.051814099999999995 | 0.059099959999999986 | 101254.74884772096 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.328487 | 0.041026499999999994 | 0.05827479999999998 | 0.06372473999999999 | 95257.78184634348 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.351387 | 0.041103 | 0.05262879999999998 | 0.06095794999999999 | 97704.72070128539 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.056127 | 0.0339025 | 0.03585089999999999 | 0.04276482999999999 | 234049.52487946453 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.214723 | 0.0718695 | 0.0801428 | 0.08391480999999999 | 112261.29039862861 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.380514 | 0.0656565 | 0.0778301 | 0.08073335 | 117963.0610470637 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.849904 | 0.0680375 | 0.07424235 | 0.08870744999999997 | 115359.21417303305 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.865533 | 0.04494 | 0.04722459999999999 | 0.05338415999999999 | 357613.0224782136 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.092833 | 0.094263 | 0.11033074999999999 | 0.12307242999999995 | 164650.35160053518 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.380384 | 0.095526 | 0.1033178 | 0.1044739 | 167136.80858113803 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.857056 | 0.10568050000000001 | 0.1123589 | 0.11863156999999999 | 151711.6007982306 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.870908 | 0.070384 | 0.0752585 | 0.07880111999999999 | 447466.58403450414 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.964497 | 0.1327685 | 0.18790584999999999 | 0.19132911 | 217288.59415501828 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.507327 | 0.1403155 | 0.16132445 | 0.16502552 | 225105.44783321937 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.493131 | 0.2583375 | 0.28508354999999996 | 0.29745664 | 124684.74242158343 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.773506 | 0.12250050000000001 | 0.13006955 | 0.13230366 | 518047.56291809067 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.090046 | 0.2100525 | 0.27143835 | 0.27643833 | 292282.9002538021 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.680914 | 0.299842 | 0.37119905000000003 | 0.38000036 | 210715.68264637823 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.113374 | 0.42399549999999997 | 0.4679741 | 0.47441825 | 148176.93975074694 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.01894 | 0.2257295 | 0.23121319999999998 | 0.23236529 | 565835.9899627769 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.028451 | 0.3175195 | 0.32832435 | 0.33333672 | 403499.8060048589 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.438296 | 0.5841095000000001 | 0.5991421499999999 | 0.60851658 | 219681.71003538315 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.031137 | 0.572726 | 0.5920025999999999 | 0.5957673 | 223519.0518210062 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.952919 | 0.026063 | 0.03298559999999998 | 0.037014359999999996 | 37395.17198413547 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.729971 | 0.0259345 | 0.02662725 | 0.03185136999999999 | 38267.78571876851 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.03588 | 0.026136 | 0.02952974999999999 | 0.0321599 | 37782.67105348666 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.618894 | 0.025855000000000003 | 0.02847944999999999 | 0.03342811999999999 | 38246.06600965025 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.493039 | 0.0266855 | 0.02908524999999999 | 0.034354819999999994 | 74183.70109903153 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.707791 | 0.0279065 | 0.0314117 | 0.03490168 | 70513.4578459972 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.963894 | 0.0285655 | 0.031130149999999995 | 0.03456924 | 69250.20723124515 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.043676 | 0.028439 | 0.029695999999999997 | 0.035825619999999996 | 69766.92266476156 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.631019 | 0.031382 | 0.0319785 | 0.03502167 | 127035.26371885571 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.975952 | 0.04741 | 0.05502984999999999 | 0.08285838999999993 | 82409.79423922575 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.211315 | 0.050918 | 0.06342519999999999 | 0.06856145999999999 | 76378.64407284537 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.266385 | 0.04654 | 0.0560426 | 0.14919738999999965 | 77941.86705845446 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.511598 | 0.0414205 | 0.04232425 | 0.044526489999999995 | 196754.2436201207 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.692616 | 0.089906 | 0.09575205 | 0.10198381999999998 | 88519.56559023187 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.951756 | 0.083086 | 0.1009516 | 0.11700731999999994 | 91451.92041030819 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.59161 | 0.0886155 | 0.09971364999999999 | 0.10346372999999999 | 93245.44018141834 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.571436 | 0.0574225 | 0.0680403 | 0.0714002 | 275179.62349923915 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.968225 | 0.1180595 | 0.1244296 | 0.14494759999999993 | 137109.65310229445 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.880727 | 0.131237 | 0.14858835 | 0.14937789999999998 | 121212.1212121212 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.901907 | 0.140816 | 0.18699934999999998 | 0.20522650999999997 | 108580.96353118322 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.507142 | 0.09214800000000001 | 0.09546575 | 0.10099266999999999 | 345005.9632124454 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.625613 | 0.181179 | 0.1932238 | 0.19635919999999998 | 178089.21624160296 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.74118 | 0.18516 | 0.22145130000000002 | 0.2270149 | 169541.3461540499 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.853798 | 0.367412 | 0.3935117 | 0.40253302999999996 | 86512.88460722203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.450824 | 0.16529 | 0.17819634999999998 | 0.18426905 | 381444.5806916806 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.672289 | 0.27738949999999996 | 0.2962899 | 0.3057368 | 229501.89482501912 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.964481 | 0.4540555 | 0.58420005 | 0.6243344799999999 | 137984.6819754922 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.747886 | 0.600732 | 0.6489185 | 0.6713361 | 106061.7751506276 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.67025 | 0.302109 | 0.31271125 | 0.31363225 | 421146.0872205391 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.669722 | 0.42237199999999997 | 0.4382528 | 0.44022376999999996 | 304936.4300453674 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.038133 | 0.8062715 | 0.8208143999999999 | 0.85218926 | 159116.81416712466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.743189 | 0.7734099999999999 | 0.79667 | 0.8062564 | 165402.71425854097 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.597124 | 0.030561 | 0.0313055 | 0.03394818999999999 | 32620.722770830285 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.495658 | 0.0302045 | 0.03362229999999999 | 0.038715339999999994 | 32646.324089200203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.723454 | 0.0305825 | 0.03108085 | 0.03439905999999999 | 32596.691566192792 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.546482 | 0.0310055 | 0.03179695 | 0.03541239999999999 | 32068.919957258546 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.169545 | 0.03161 | 0.0320486 | 0.03700497999999998 | 63241.866621638466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.437114 | 0.0322375 | 0.038584299999999995 | 0.04170072 | 61354.925358665554 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.012208 | 0.0353025 | 0.03720529999999999 | 0.04880316999999999 | 55757.02426991752 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.427389 | 0.0329065 | 0.0335935 | 0.03947145999999998 | 60316.06826090077 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.084698 | 0.036778500000000006 | 0.03775105 | 0.041493459999999996 | 110284.58938290259 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.442542 | 0.0566195 | 0.0655492 | 0.08696409999999992 | 68070.02090770692 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.563772 | 0.0558215 | 0.0645085 | 0.07106538999999999 | 70823.91227750226 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.468821 | 0.057355 | 0.0670023 | 0.07067499 | 69176.09882774184 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.343667 | 0.049986 | 0.05220225 | 0.05461674999999999 | 160952.71128865983 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.475478 | 0.109865 | 0.1260117 | 0.12937785999999998 | 72408.75320453987 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.647233 | 0.1342315 | 0.1541997 | 0.16378504999999996 | 58665.950942065174 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.805601 | 0.10121749999999999 | 0.11781064999999999 | 0.12343613999999999 | 77008.2154289425 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.249411 | 0.0674585 | 0.07042895 | 0.07089106 | 234577.20830250843 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.326311 | 0.134485 | 0.1781641 | 0.18596548 | 115539.23242955178 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.607258 | 0.1647605 | 0.20394885 | 0.22017061 | 94757.12387018415 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.624946 | 0.1710925 | 0.22390844999999998 | 0.24055088999999996 | 91389.10178107079 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.164134 | 0.1213175 | 0.13436615000000002 | 0.13736279 | 258072.75780742677 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.185953 | 0.20981 | 0.23974 | 0.24602954 | 149189.3516846182 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.751787 | 0.22753 | 0.30657855 | 0.3579680399999998 | 133098.9128730357 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.824428 | 0.4666585 | 0.5051987 | 0.51658439 | 68641.79131050693 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.04543 | 0.2011185 | 0.21485425 | 0.21627967 | 312845.71896052314 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.32528 | 0.34414350000000005 | 0.3617944 | 0.36893774 | 185353.4621014727 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.721819 | 0.49616000000000005 | 0.76496205 | 0.79368686 | 120818.30388242446 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.554651 | 0.706645 | 0.7430936 | 0.74913482 | 90370.57981255674 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.046177 | 0.37978500000000004 | 0.39799080000000003 | 0.39978469 | 334680.70205970877 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.369599 | 0.5272725 | 0.53889745 | 0.54242928 | 243480.53291345117 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.850184 | 1.020871 | 1.0507278 | 1.05989049 | 125442.5402114393 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.445241 | 0.9540645 | 0.98133755 | 0.9942778999999999 | 134076.0756871179 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.906117 | 0.019139999999999997 | 0.0198751 | 0.02593001999999998 | 51465.426555748374 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.811638 | 0.019194999999999997 | 0.01966255 | 0.02927136999999997 | 51097.837028558584 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.906996 | 0.019411499999999998 | 0.020023449999999998 | 0.03026419999999997 | 50400.38062367447 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.85414 | 0.019211 | 0.022463649999999998 | 0.02859967 | 50081.181595366084 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.506828 | 0.020035999999999998 | 0.021438849999999995 | 0.026227689999999994 | 98916.76253349568 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.8212 | 0.020851500000000002 | 0.02377615 | 0.025969079999999995 | 92256.96517022795 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.091124 | 0.020879 | 0.024347399999999998 | 0.028824799999999994 | 94090.63751111444 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.712389 | 0.020484 | 0.02440955 | 0.028422899999999994 | 94900.25034686041 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.539336 | 0.023212499999999997 | 0.02528525 | 0.028590589999999996 | 170143.98434675345 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.798459 | 0.038025 | 0.0555205 | 0.05793101999999999 | 98960.32284815724 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.144027 | 0.037476 | 0.055055599999999996 | 0.058773589999999994 | 101120.9254587098 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.103309 | 0.0368325 | 0.054855949999999994 | 0.06194058999999999 | 100953.9135289349 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.66739 | 0.0300255 | 0.0333081 | 0.03558735999999999 | 262859.7564341497 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.849449 | 0.06744700000000001 | 0.07661295 | 0.07888208 | 118758.18132533536 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.022564 | 0.06729450000000001 | 0.0727469 | 0.07746578999999999 | 118701.688412816 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.843436 | 0.06618550000000001 | 0.0724844 | 0.1002014499999999 | 119238.47157742362 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.565937 | 0.0397765 | 0.04493804999999999 | 0.05672685999999997 | 402983.08226647764 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.871203 | 0.097455 | 0.105475 | 0.12495228999999992 | 162312.59742560048 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.280571 | 0.0971485 | 0.10820529999999999 | 0.12346808999999996 | 163497.47253344647 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.566026 | 0.0933285 | 0.10124169999999999 | 0.10528962999999998 | 170547.21994174534 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.638078 | 0.061523999999999995 | 0.06477959999999999 | 0.06568405999999999 | 512975.7208591318 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.780178 | 0.12583149999999999 | 0.17150985 | 0.19302514999999992 | 243151.96678787287 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.081684 | 0.13665 | 0.1497604 | 0.15838744 | 239299.6059483301 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.630377 | 0.277117 | 0.29538295000000003 | 0.30100475 | 115634.74222703262 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.752198 | 0.10447100000000001 | 0.10947445 | 0.11048953 | 606167.2593530188 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.761162 | 0.19217299999999998 | 0.2752872 | 0.28135262 | 317040.7731277033 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.149998 | 0.27903900000000004 | 0.35600659999999995 | 0.40314574999999997 | 232569.75112056828 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.228186 | 0.398675 | 0.42432505 | 0.43294256 | 159732.63153470564 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.622778 | 0.1960535 | 0.20119925 | 0.20429756999999998 | 650835.0366059116 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.764557 | 0.269393 | 0.28414045 | 0.28619093 | 472937.6224834545 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.062327 | 0.504069 | 0.53519525 | 0.54469282 | 253705.80940319353 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.689289 | 0.47223099999999996 | 0.4913554 | 0.49339782 | 270883.90009158413 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.819906 | 0.024015 | 0.030649899999999997 | 0.03552515999999999 | 39474.48408823021 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.457658 | 0.0247295 | 0.0253515 | 0.03611451999999998 | 39759.85050296211 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.774892 | 0.024956 | 0.025465599999999998 | 0.03706811999999998 | 39346.254118569144 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.518685 | 0.0243775 | 0.0248166 | 0.029171659999999985 | 40824.75809289592 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.244469 | 0.0255675 | 0.025932550000000002 | 0.02886194999999999 | 78037.87802523588 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.469815 | 0.026435 | 0.0270662 | 0.035955069999999985 | 74613.83609130942 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.842 | 0.026919 | 0.0273062 | 0.03304682 | 73769.92341206552 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.542931 | 0.026364 | 0.02885504999999999 | 0.03308986 | 75035.5293231345 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.484836 | 0.029856 | 0.0302468 | 0.03728588999999999 | 132856.55544137274 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.450622 | 0.046335 | 0.055317149999999995 | 0.06253025999999998 | 83344.86270600767 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.697004 | 0.048472 | 0.058449949999999994 | 0.07100831999999997 | 81585.96590848829 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.678171 | 0.050384 | 0.057832749999999995 | 0.062254439999999994 | 79061.22699071228 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.353688 | 0.0404645 | 0.0420202 | 0.04314746999999999 | 199595.71887142592 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.506857 | 0.093183 | 0.1081369 | 0.11168505 | 86673.66412048331 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.679352 | 0.0856755 | 0.09947705 | 0.10644622999999999 | 91051.2550277365 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.710751 | 0.0842065 | 0.10092895 | 0.10598516999999999 | 90837.70073428655 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.393473 | 0.0545795 | 0.058232099999999995 | 0.06219393999999999 | 288353.75238342397 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.584544 | 0.114813 | 0.14934624999999999 | 0.15209111 | 134464.17706242835 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.737839 | 0.137744 | 0.15760299999999997 | 0.17264842999999996 | 114146.35788935395 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.41273 | 0.14574399999999998 | 0.21256809999999998 | 0.21946109 | 105207.45331682278 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.720866 | 0.092174 | 0.09721869999999999 | 0.09836001 | 343188.8594032461 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.510733 | 0.17467149999999998 | 0.22420835 | 0.22796961999999998 | 174489.3895183351 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.772364 | 0.178778 | 0.22060685 | 0.22418401999999998 | 175260.99100012906 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.409783 | 0.415116 | 0.43994585 | 0.44139362000000004 | 77193.37621007251 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.535468 | 0.158926 | 0.16767274999999998 | 0.16886381 | 400112.63170582516 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.562188 | 0.274898 | 0.3089169 | 0.31263228 | 229377.97066872212 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.798346 | 0.503609 | 0.69401085 | 0.69903205 | 121710.62302869203 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.933458 | 0.641223 | 0.672234 | 0.68693472 | 99737.10234856253 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.496632 | 0.29769049999999997 | 0.30450045 | 0.30642949 | 429432.18128533213 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.351054 | 0.3996505 | 0.41680029999999996 | 0.42618434 | 318786.93601174135 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.768976 | 0.7125805000000001 | 0.75752175 | 0.76291781 | 179436.99679662907 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.850936 | 0.6956169999999999 | 0.7164688 | 0.73542571 | 183699.71571894927 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.255509 | 0.0300765 | 0.03115805 | 0.03694205999999999 | 33008.46403034666 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.496766 | 0.0300285 | 0.03412589999999999 | 0.03748583 | 32912.31894933346 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.193479 | 0.0293435 | 0.0299328 | 0.03688997 | 33758.92079482003 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.479814 | 0.029862 | 0.03105055 | 0.03621071999999998 | 33202.47132634576 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.329012 | 0.031546000000000005 | 0.034902449999999995 | 0.05236045999999995 | 61519.79737839535 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.554437 | 0.0342885 | 0.0353821 | 0.04142883 | 58348.91249296896 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.749021 | 0.032986 | 0.03918869999999999 | 0.04440727999999999 | 60274.98654360926 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.498978 | 0.034378000000000006 | 0.035059850000000004 | 0.04239518999999999 | 57816.80283487348 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.318601 | 0.0371645 | 0.0382266 | 0.04017358999999999 | 108920.59688487092 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.419369 | 0.059189 | 0.0683839 | 0.0718393 | 66263.2340102675 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.068297 | 0.0593305 | 0.0680973 | 0.06982169 | 66751.70834309577 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.306678 | 0.06272349999999999 | 0.07308955 | 0.07767200999999999 | 63100.06186961066 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.229898 | 0.052304 | 0.05524465 | 0.05845490999999999 | 153551.21739243928 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.661535 | 0.11563999999999999 | 0.12574405 | 0.12906723 | 68994.42352571854 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.812468 | 0.107168 | 0.12939465 | 0.13396761999999998 | 71647.58273594176 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.55438 | 0.10423199999999999 | 0.1308703 | 0.1337051 | 73138.36352180237 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.201076 | 0.067857 | 0.0718747 | 0.09237406999999992 | 230872.03542269638 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.423248 | 0.14336549999999998 | 0.1675992 | 0.17066403 | 105630.3344969395 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.883247 | 0.16647800000000001 | 0.17437575000000002 | 0.17629222 | 96035.49856168834 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.343433 | 0.1881255 | 0.22663729999999999 | 0.25348366 | 83038.58097029751 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.133111 | 0.1200125 | 0.12564755 | 0.12759429 | 264311.562540576 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.393206 | 0.2247805 | 0.25722870000000003 | 0.26247428 | 139849.05217741916 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.044334 | 0.233986 | 0.24422525 | 0.25592842 | 136117.42237499892 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.530027 | 0.5707774999999999 | 0.6005412 | 0.61033639 | 56389.4795980572 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.09326 | 0.2073345 | 0.21832645 | 0.22910764999999994 | 304653.18709599535 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.460359 | 0.371794 | 0.39095745 | 0.39539631 | 171339.51465619466 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.927712 | 0.5240225000000001 | 0.5820546 | 0.59918467 | 125703.46803297957 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 16.416228 | 0.8456405 | 0.9020653 | 0.9063865600000001 | 75303.52261407142 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.068823 | 0.40416850000000004 | 0.41294225 | 0.41512927 | 315924.5574131426 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.422318 | 0.5323595 | 0.5495604 | 0.55282285 | 240072.09965332836 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.941129 | 1.0344225 | 1.06704335 | 1.07607812 | 125974.75178594307 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.070561 | 0.9292315 | 0.9660717499999999 | 0.97664466 | 137172.82646602226 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.475749 | 0.045093999999999995 | 0.05107449999999999 | 0.05972891999999998 | 21841.109423521386 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.600349 | 0.0385885 | 0.04500015 | 0.047668779999999994 | 25505.727566182264 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.143867 | 0.0420645 | 0.049030249999999984 | 0.052891719999999996 | 23358.613582192887 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.518218 | 0.0401425 | 0.04788199999999999 | 0.057940569999999976 | 24406.816726284385 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.389262 | 0.040605 | 0.0449494 | 0.04580486 | 49412.509962797325 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.623014 | 0.040732000000000004 | 0.044696 | 0.04690785 | 48843.364702170256 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.022873 | 0.0427385 | 0.04536514999999999 | 0.05105144999999999 | 46742.863766988696 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.728479 | 0.041939500000000005 | 0.048638499999999994 | 0.05725433999999998 | 46834.87568384773 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.350917 | 0.04686 | 0.0489253 | 0.05219167999999999 | 86154.68384246442 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.526011 | 0.0729335 | 0.0825095 | 0.08922211999999999 | 54907.79608342691 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.26299 | 0.0790885 | 0.08922329999999999 | 0.09162407 | 50565.15408592991 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.481017 | 0.075476 | 0.08615165 | 0.09324495999999999 | 52783.59570299304 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.220219 | 0.06683800000000001 | 0.07689149999999999 | 0.0790601 | 117110.58899306852 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.713624 | 0.136993 | 0.14382455 | 0.16721161 | 58063.60490503915 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.017741 | 0.134631 | 0.1472283 | 0.15464326 | 59938.239637877145 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.679318 | 0.132141 | 0.14133965 | 0.14581787999999998 | 60641.73816201259 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.35142 | 0.096681 | 0.10339205 | 0.10607407999999999 | 163607.61331667806 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.536225 | 0.1811625 | 0.19005005 | 0.19388940999999998 | 88446.55782580671 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.887115 | 0.2177265 | 0.25665335 | 0.28741416999999997 | 71382.47874229783 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.492124 | 0.2430765 | 0.34216589999999997 | 0.35414061999999996 | 62109.98809351529 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.327105 | 0.15174900000000002 | 0.1633872 | 0.16840957999999998 | 209377.6863320733 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.689945 | 0.2768025 | 0.2991795 | 0.30518291 | 114954.15628247456 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.165787 | 0.297107 | 0.4752100999999999 | 0.52097719 | 97924.99386132693 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.842074 | 0.681814 | 0.7309896499999999 | 0.76568576 | 47057.96402953935 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.482537 | 0.268031 | 0.28742965 | 0.29127560999999996 | 235465.05488874388 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.5929 | 0.443898 | 0.46284915 | 0.47250693 | 144424.28171147648 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.988723 | 0.617487 | 0.9074042499999999 | 0.9810059199999998 | 98674.4658142159 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.544504 | 1.1057899999999998 | 1.17065465 | 1.20245402 | 57903.648004013296 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.332447 | 0.5143245000000001 | 0.5258156 | 0.5301735799999999 | 248036.00056521207 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.630347 | 0.655126 | 0.67172985 | 0.67262431 | 195161.20956282612 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.166492 | 1.1916134999999999 | 1.2704182 | 1.4023087199999997 | 106791.79639436521 | - |
| `separator3_grid_search_6ports_learned_dense_hd128_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.695268 | 1.0939385000000001 | 1.13941025 | 1.1970866499999997 | 116519.38464729299 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.343398 | 0.018094 | 0.0188147 | 0.026915789999999988 | 54127.19891745603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.868417 | 0.020046 | 0.025294699999999993 | 0.029778519999999996 | 48195.74412301097 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.18505 | 0.020141 | 0.0207352 | 0.025630929999999996 | 49252.83450062551 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.774125 | 0.0200925 | 0.021322499999999994 | 0.025762239999999992 | 49383.15501076059 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.654156 | 0.020453 | 0.0209299 | 0.028170269999999994 | 96641.88768666382 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.981264 | 0.0427155 | 0.052089899999999995 | 0.06442017 | 45279.745056923435 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.179917 | 0.0353705 | 0.04306195 | 0.04779107999999999 | 56633.511521238426 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.729392 | 0.0322655 | 0.03459485 | 0.038349119999999993 | 61846.6672686409 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.657976 | 0.0239545 | 0.027496749999999997 | 0.029984179999999996 | 162448.02678443064 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.933512 | 0.048417 | 0.0506084 | 0.07206873999999992 | 82728.7584605667 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.283827 | 0.044337 | 0.049728699999999994 | 0.05369418999999999 | 91446.2963335522 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.108415 | 0.0466385 | 0.0505862 | 0.058171459999999994 | 86703.94936489356 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.639423 | 0.031824500000000006 | 0.03476974999999999 | 0.038017779999999994 | 249590.04834559234 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.862037 | 0.069054 | 0.07326405 | 0.07432069 | 115865.1885358348 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.15304 | 0.06353 | 0.0731647 | 0.08210305999999999 | 123730.33345015538 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.126805 | 0.0689035 | 0.09806359999999999 | 0.10222250999999999 | 110069.2445617538 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.637611 | 0.043127 | 0.046406300000000004 | 0.04953098999999999 | 376413.01918530103 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.793434 | 0.0980635 | 0.1109636 | 0.12140819999999998 | 165005.5421236461 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.228913 | 0.083458 | 0.0971888 | 0.10199455 | 191513.18879513783 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.930997 | 0.14728000000000002 | 0.15985095 | 0.16346217 | 109033.16611734395 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.735378 | 0.07279949999999999 | 0.07802985 | 0.09447517999999994 | 430124.4081084902 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.865712 | 0.1244465 | 0.1824208 | 0.18531931 | 217850.71480223443 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.033565 | 0.224442 | 0.2547448 | 0.25607071000000003 | 145477.42508594535 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.434742 | 0.2542835 | 0.2704125 | 0.27742568999999995 | 126231.76562421845 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.780761 | 0.119185 | 0.12717685 | 0.12974982 | 529713.1851405519 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.80644 | 0.2030165 | 0.2887650499999999 | 0.31117901 | 304020.24014748784 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.161362 | 0.3761815 | 0.40466625 | 0.41120803 | 170758.26595407818 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.712609 | 0.38453899999999996 | 0.4090569 | 0.41558645 | 166839.2584036674 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.788126 | 0.2268505 | 0.23285989999999998 | 0.24031394999999997 | 563273.8744951878 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.952088 | 0.31696250000000004 | 0.32845205 | 0.33705723 | 404068.33679474687 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.436382 | 0.582174 | 0.6473833 | 0.6862187799999999 | 216089.1616294257 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.90007 | 0.6227195 | 0.6486698999999999 | 0.65422108 | 206057.7700344416 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.119319 | 0.026265 | 0.02767645 | 0.03804926999999997 | 37437.899883568134 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.604787 | 0.025512 | 0.028599649999999987 | 0.039198549999999985 | 38322.04621332197 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.281314 | 0.025431000000000002 | 0.028996199999999986 | 0.03517485 | 38566.32018684611 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 15.743254 | 0.0249165 | 0.0372033 | 0.041685699999999985 | 36162.807853983264 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.510251 | 0.027008 | 0.030514799999999995 | 0.03245513 | 73055.09084400546 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.78508 | 0.043979 | 0.0560832 | 0.07472154999999993 | 43669.30458379222 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.109748 | 0.044968 | 0.05467565 | 0.05889215 | 43049.82155848964 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.82138 | 0.044461 | 0.06133699999999999 | 0.06697067999999999 | 43356.82406060937 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.534207 | 0.032055 | 0.0326862 | 0.03648849 | 125716.74257862638 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.694477 | 0.070062 | 0.0768993 | 0.07789903 | 57819.326745977436 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.050135 | 0.060013 | 0.0704817 | 0.07561408999999998 | 64860.85400989257 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 15.833031 | 0.052796499999999996 | 0.057030199999999996 | 0.059713749999999996 | 75274.19566699147 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.458943 | 0.044102 | 0.04793294999999999 | 0.05214359999999999 | 181601.58066015807 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.845704 | 0.0810015 | 0.10638260000000001 | 0.11821962999999998 | 89347.51963467586 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.235795 | 0.08859 | 0.09749564999999999 | 0.1220886299999999 | 88422.37245183301 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.931266 | 0.09045900000000001 | 0.10780709999999996 | 0.12536872 | 86485.25231856151 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.41011 | 0.0607015 | 0.06490444999999999 | 0.06749996 | 258945.93467829854 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.804188 | 0.110305 | 0.1608865 | 0.16734816 | 129077.99747749323 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.053657 | 0.1274775 | 0.1890180999999999 | 0.21525 | 121010.16258471654 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.046665 | 0.236491 | 0.25544269999999997 | 0.26448616 | 68288.04683467405 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.382224 | 0.1035365 | 0.111897 | 0.11310318999999999 | 304334.6768441255 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.732244 | 0.1835815 | 0.19219275 | 0.27496983 | 171568.90866840116 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.932837 | 0.27419099999999996 | 0.3978268 | 0.41367225999999996 | 112026.0606224826 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.823388 | 0.38934250000000004 | 0.4068148 | 0.40990993 | 81966.13394997822 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.491404 | 0.18975750000000002 | 0.20400085 | 0.20945898 | 334878.4134967721 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.679844 | 0.2964245 | 0.30803094999999997 | 0.31729568999999996 | 215587.47283176772 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.816957 | 0.6490705 | 0.69122885 | 0.69965932 | 98495.37844294442 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.65163 | 0.5878559999999999 | 0.6334706999999999 | 0.64966001 | 108085.71318652118 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.428873 | 0.3578235 | 0.36375405 | 0.36585958 | 357851.9511681937 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.588569 | 0.43986400000000003 | 0.4605895 | 0.46302617 | 289578.9101851554 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.000325 | 1.0185595 | 1.07639895 | 1.09579264 | 126639.7121954241 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.168684 | 0.908741 | 0.94244935 | 0.9474324599999999 | 140616.7847664564 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.797354 | 0.0393795 | 0.04408164999999999 | 0.05642043999999996 | 24915.67290505285 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.013577 | 0.033914 | 0.0354988 | 0.04455129999999999 | 29204.822767612553 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 7.353766 | 0.033677 | 0.03919779999999999 | 0.04186858999999999 | 29351.576707997603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.071304 | 0.033176 | 0.03931774999999999 | 0.04253849 | 29664.928697377385 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.774091 | 0.0349585 | 0.03863215 | 0.04218292999999999 | 56909.26898336713 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.234246 | 0.0576985 | 0.07860104999999999 | 0.08827549999999998 | 32320.892160050473 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 7.303165 | 0.062753 | 0.0741399 | 0.07566413 | 31653.452932739892 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.135867 | 0.057800500000000005 | 0.06493999999999998 | 0.06778010999999999 | 34519.88535255677 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.728678 | 0.042869000000000004 | 0.047679549999999994 | 0.05429303999999999 | 93091.62403458168 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.972123 | 0.0879605 | 0.0987487 | 0.10639308 | 47167.731045824155 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 7.366227 | 0.07258500000000001 | 0.0958784 | 0.10118060999999999 | 50299.23011998378 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.767908 | 0.072911 | 0.09008285 | 0.09501691999999999 | 51794.57855787319 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.809577 | 0.056792499999999996 | 0.06324025 | 0.06561232 | 137425.35655867666 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.027346 | 0.107908 | 0.1155514 | 0.11630582 | 75151.6089771603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.526911 | 0.11160049999999999 | 0.1188316 | 0.12365355 | 71311.02284787345 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.092225 | 0.1341185 | 0.1587727 | 0.1920744999999999 | 59282.35742222525 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.691691 | 0.0867365 | 0.09126845 | 0.09422802 | 182428.23843188336 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.941918 | 0.139992 | 0.1702246 | 0.17706127 | 108915.06571730848 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.634092 | 0.152035 | 0.2226885999999998 | 0.2968224 | 99823.69886983352 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.397295 | 0.29406750000000004 | 0.31411455 | 0.31704814000000003 | 54558.91943877695 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.655275 | 0.1474425 | 0.1585772 | 0.16164073 | 214250.32472314843 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.154654 | 0.2344245 | 0.26708035 | 0.27299661000000003 | 133832.0920825018 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.439212 | 0.487336 | 0.56745855 | 0.5722499499999999 | 63985.29106129083 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.162329 | 0.477842 | 0.5021097 | 0.6223959799999995 | 66391.99468565278 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.69099 | 0.2629 | 0.27956785 | 0.28845488999999996 | 241538.52796155674 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.958267 | 0.3849315 | 0.3932829 | 0.39788562 | 166843.91225407532 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.285772 | 0.849515 | 0.87911445 | 0.8910593699999999 | 75746.69255025835 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.933983 | 0.8083100000000001 | 0.8371474999999999 | 0.8497961199999999 | 79309.0869923035 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.672109 | 0.4940015 | 0.5030759 | 0.50402573 | 258760.50886062693 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.028822 | 0.5867100000000001 | 0.6074998500000001 | 0.60907591 | 217481.0522181861 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.524009 | 1.285018 | 1.35057935 | 1.39397697 | 100520.82193802857 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.944944 | 1.269381 | 1.31174445 | 1.32511622 | 100904.45073765883 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.355889 | 0.056207 | 0.06801459999999998 | 0.07892829999999997 | 17301.33739338051 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.73576 | 0.0470045 | 0.052461049999999995 | 0.05864813 | 20986.22673939094 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.559758 | 0.0481925 | 0.053822699999999994 | 0.05975294999999999 | 20510.67478068961 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 9.072677 | 0.05191949999999999 | 0.05807719999999999 | 0.06789084999999999 | 19004.41206430485 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.561537 | 0.05174 | 0.05656285 | 0.061942359999999995 | 37964.78386648544 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.787648 | 0.07354150000000001 | 0.0828541 | 0.08432738000000001 | 27025.507755239705 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 17.717153 | 0.072002 | 0.07954934999999999 | 0.08639491 | 27607.66852687137 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.800499 | 0.0742525 | 0.0901071 | 0.09634043999999999 | 26167.214782592313 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.500139 | 0.06820100000000001 | 0.07020755 | 0.07234945 | 58818.21847262614 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.87119 | 0.096786 | 0.10772474999999998 | 0.11003112000000001 | 41830.572794399384 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.314423 | 0.087389 | 0.0931183 | 0.10026222 | 45407.81668319511 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 9.126291 | 0.0883505 | 0.09530844999999999 | 0.10417399999999996 | 44987.442880006114 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.517453 | 0.0847675 | 0.09217345 | 0.09294146 | 93141.63233039199 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.943398 | 0.134264 | 0.140862 | 0.14254515 | 61532.592275813695 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 8.302183 | 0.13951 | 0.14915015 | 0.15161989 | 56792.13326728408 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.745277 | 0.188247 | 0.22404505 | 0.23687842999999997 | 42555.246817239895 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.49744 | 0.1186865 | 0.12908904999999998 | 0.21705860999999965 | 130301.92421737814 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.884942 | 0.178579 | 0.1903651 | 0.19224138 | 89327.3970811603 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 8.477064 | 0.18055300000000002 | 0.20501775 | 0.21632142999999998 | 87395.34134763178 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.94477 | 0.35524599999999995 | 0.38249869999999997 | 0.38668374 | 44817.06849090062 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.572566 | 0.19485999999999998 | 0.20804185 | 0.21422201 | 162535.75278637075 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.966281 | 0.2929395 | 0.31223744999999997 | 0.31544195 | 108403.15512803158 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 8.20975 | 0.4831845 | 0.6654944 | 0.68273423 | 65132.27245073811 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.839726 | 0.6487130000000001 | 0.68388825 | 0.68499002 | 49826.22946766186 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.590887 | 0.33832949999999995 | 0.3487331 | 0.35470659 | 188629.5298350022 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.842668 | 0.46242249999999996 | 0.4744827 | 0.48014192 | 138223.66351107802 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 8.512531 | 1.0587645 | 1.1204998 | 1.1347346900000002 | 60656.30884372775 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.814956 | 1.018704 | 1.0690612000000002 | 1.09954945 | 62539.81416489658 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.583507 | 0.6396385 | 0.6507370499999999 | 0.65316709 | 200134.340175843 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.836919 | 0.7255130000000001 | 0.74278565 | 0.7473744800000001 | 176165.4770771692 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 8.239095 | 1.636868 | 1.69484325 | 1.73415981 | 78573.18752343231 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.889936 | 1.477617 | 1.5191686999999998 | 1.54933696 | 86666.8871255608 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.353228 | 0.0309855 | 0.0335329 | 0.03862414999999999 | 31826.092048696468 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.113196 | 0.038678000000000004 | 0.0441646 | 0.04857875999999999 | 25721.2363272338 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.670706 | 0.037968 | 0.043757849999999994 | 0.04937278999999999 | 25756.600128783004 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.592132 | 0.040899500000000005 | 0.04657135 | 0.04878419999999999 | 24088.212962927275 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.014947 | 0.029477999999999997 | 0.030664399999999998 | 0.03381355 | 67456.67426453675 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.21967 | 0.0510805 | 0.06265035 | 0.06418895 | 38995.46131825717 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.566687 | 0.0469955 | 0.0504359 | 0.05618461999999999 | 42770.79473269109 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.584766 | 0.050247 | 0.057506999999999996 | 0.063581 | 39435.06897587915 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.14454 | 0.0349495 | 0.037695349999999996 | 0.04313261999999999 | 114034.96768249015 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.347125 | 0.071408 | 0.0769917 | 0.07799141999999999 | 55752.92089552572 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.539175 | 0.0815765 | 0.0930209 | 0.10079718999999997 | 48694.53601046154 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.497989 | 0.0692815 | 0.08162264999999999 | 0.08489116999999999 | 56816.551570537325 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.892647 | 0.046592999999999996 | 0.049212 | 0.05769123999999998 | 172221.00949928034 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.283442 | 0.083665 | 0.11642315 | 0.12113751 | 88858.44006342716 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.699158 | 0.086652 | 0.09960785 | 0.10135517 | 90330.99308313002 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 7.720282 | 0.128934 | 0.15439915 | 0.1687411 | 60839.84230921271 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.870943 | 0.0668815 | 0.07130155 | 0.07326418999999999 | 240658.8276064554 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.138513 | 0.118658 | 0.17158774999999998 | 0.17738831 | 127956.74038521059 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.450032 | 0.1162465 | 0.1517519 | 0.16103717999999997 | 129442.42352460716 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.022245 | 0.2360315 | 0.24710079999999998 | 0.25051527 | 68516.09888517456 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.849884 | 0.1160635 | 0.12249484999999999 | 0.12614276 | 272346.59099516633 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.149032 | 0.1902525 | 0.22989759999999998 | 0.2324533 | 159835.28973392918 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.782725 | 0.382869 | 0.41703640000000003 | 0.42221165 | 83879.61241537792 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.258537 | 0.362854 | 0.3807989 | 0.5090195799999995 | 87285.09318992974 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.997 | 0.2124925 | 0.2182741 | 0.24024931 | 299646.56687437167 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.141993 | 0.322549 | 0.33469015 | 0.33542278000000003 | 198162.48878214534 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.459238 | 0.6843275 | 0.72301665 | 0.73651194 | 94424.00853168129 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.182892 | 0.5969359999999999 | 0.66871985 | 0.67257032 | 107397.85272760675 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.940452 | 0.411266 | 0.41765050000000004 | 0.41866464999999997 | 311060.0713377391 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.976528 | 0.4827855 | 0.49916035000000003 | 0.50144608 | 264975.9325258661 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.590204 | 0.997989 | 1.0926136 | 1.09732065 | 126573.29117651653 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.380086 | 0.9817210000000001 | 1.02142645 | 1.03131327 | 129931.55997191122 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.790253 | 0.0604405 | 0.06712575 | 0.07305896999999997 | 16410.665751170327 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 7.593855 | 0.054523 | 0.0613697 | 0.06167151 | 18035.70926148099 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 8.102867 | 0.059632 | 0.06532719999999999 | 0.06895049999999998 | 16649.77269730314 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 8.380346 | 0.053651000000000004 | 0.057769299999999996 | 0.06692961999999998 | 18477.58724100428 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 7.323915 | 0.058233 | 0.0666831 | 0.07082975 | 33946.566746079 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 7.687988 | 0.0694665 | 0.08644285 | 0.08998309 | 28055.716408329627 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 8.135644 | 0.075427 | 0.0850771 | 0.08779659999999999 | 26133.412639529557 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 8.674255 | 0.06734599999999999 | 0.07272139999999999 | 0.07464422999999999 | 29685.297258176884 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 7.302068 | 0.067049 | 0.07046775 | 0.07350854999999999 | 59898.51394781822 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 7.834493 | 0.09642400000000001 | 0.10551625 | 0.10705840999999999 | 43445.638861602616 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 8.15259 | 0.09183 | 0.11060855 | 0.11722990999999998 | 41747.517692598 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 8.67811 | 0.0929195 | 0.10887059999999998 | 0.11512619 | 41997.554902353586 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 7.235318 | 0.0923725 | 0.0988671 | 0.10227460999999999 | 85622.99936510545 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 7.640633 | 0.11371500000000001 | 0.14250279999999999 | 0.14296987 | 64673.44196423591 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 7.951184 | 0.12859199999999998 | 0.18875784999999998 | 0.20083259999999997 | 58693.88497759362 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 8.526738 | 0.1989545 | 0.22244585 | 0.23075087999999996 | 39674.10499930421 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 7.49107 | 0.1226495 | 0.12745945 | 0.12950348 | 130308.12169801256 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 7.655686 | 0.170073 | 0.17822515 | 0.182147 | 94446.88891357626 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 7.936732 | 0.169133 | 0.3101364 | 0.33044393999999994 | 84025.33447859864 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 8.689966 | 0.3485785 | 0.3670208 | 0.36916525 | 46605.93447345431 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 7.224936 | 0.20338699999999998 | 0.2132754 | 0.21865282 | 156239.10598421167 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 7.485898 | 0.279824 | 0.29233135 | 0.29627972 | 113967.34650325745 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 7.815573 | 0.5245635 | 0.6436279 | 0.6484346400000001 | 61148.769835801024 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 8.659484 | 0.582065 | 0.60754135 | 0.6115181199999999 | 54933.05154508993 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 7.333591 | 0.351225 | 0.3609158 | 0.36218263 | 181549.54007544406 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 7.68043 | 0.45502299999999996 | 0.47403755000000003 | 0.47726465 | 141552.45263732347 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 7.871692 | 1.038277 | 1.1238313 | 1.16905983 | 61738.03567513099 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 8.554858 | 0.9369195 | 1.00743125 | 1.02565511 | 68253.72648816426 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 7.368608 | 0.6529085 | 0.6648284 | 0.7326673799999998 | 195221.9310365107 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 7.819764 | 0.7155715 | 0.72919825 | 0.8288967699999996 | 177894.24909586637 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 7.809361 | 1.5649015 | 1.6405794 | 1.6707796899999998 | 81325.01009129013 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 8.500823 | 1.574929 | 1.62326005 | 1.6731244899999997 | 81177.87060805522 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 7.933373 | 0.10459550000000001 | 0.12841124999999998 | 0.14102337999999998 | 9348.574501314317 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 9.011477 | 0.0781605 | 0.08440825 | 0.09350729999999999 | 12709.98483698809 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 9.534048 | 0.07602300000000001 | 0.08183275 | 0.08880581 | 13109.527480191504 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.001104 | 0.077187 | 0.08609525 | 0.09080829 | 12757.09696439775 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 8.710801 | 0.08945349999999999 | 0.10218895 | 0.10631062 | 21978.060620765926 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 8.893034 | 0.0874125 | 0.09803825000000001 | 0.09965027 | 22543.08039020269 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 9.463165 | 0.0899415 | 0.0979521 | 0.10142438999999999 | 22023.525970912648 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 9.635123 | 0.09477350000000001 | 0.10317409999999999 | 0.11013106 | 21017.017689393277 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 8.492807 | 0.098977 | 0.1158648 | 0.12033428 | 39853.633545441015 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 8.938042 | 0.1083645 | 0.11486769999999999 | 0.12365298999999998 | 36937.14258697404 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 9.47872 | 0.12323400000000001 | 0.13315725 | 0.13740459 | 32420.22840699317 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 10.08087 | 0.134486 | 0.14858785 | 0.15755148 | 29722.964140432683 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 8.500294 | 0.128563 | 0.13884995 | 0.14153849 | 62216.32273224225 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 8.953729 | 0.15801749999999998 | 0.1703178 | 0.17449378 | 50639.532992098844 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 9.335957 | 0.1667475 | 0.18403044999999998 | 0.18902227 | 47426.43981335799 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 10.768949 | 0.2759515 | 0.3022892 | 0.32769353 | 29450.09096764974 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 8.65323 | 0.1579795 | 0.1718084 | 0.17453787 | 100578.03453673694 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 8.958266 | 0.2128405 | 0.22296944999999999 | 0.22765627 | 74799.36472899535 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 9.223386 | 0.21714050000000001 | 0.4431954499999999 | 0.46776855 | 66400.48372752396 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 10.142777 | 0.442784 | 0.4705644 | 0.47462967 | 36143.66111341903 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 8.666014 | 0.300514 | 0.31529365 | 0.31966729 | 105401.55985085943 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 8.872512 | 0.3619375 | 0.3744258 | 0.3769922 | 88265.35788265087 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 9.252095 | 0.7818125 | 0.81508435 | 0.83553377 | 41215.44448834748 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 9.95747 | 0.7230814999999999 | 0.7736666 | 0.7957640399999999 | 43986.62069954837 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 8.669644 | 0.4918575 | 0.5047239 | 0.51261057 | 130124.24710415605 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 8.998192 | 0.5847135 | 0.5997125999999999 | 0.6036613399999999 | 109755.03807350823 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 9.358857 | 1.337584 | 1.4084906 | 1.41962859 | 48089.8125338883 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 10.132276 | 1.1719385 | 1.21959525 | 1.22629013 | 54729.56854560345 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 8.674828 | 0.901976 | 0.90846095 | 0.91042206 | 142096.81388969714 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 8.907439 | 0.953943 | 0.9749023499999999 | 0.9783827 | 133961.41701875147 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 9.254605 | 2.0668230000000003 | 2.12685515 | 2.15222744 | 61829.712778570865 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 9.91766 | 2.0963255 | 2.18505045 | 2.18715619 | 60902.030054028946 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 9.057979 | 0.153488 | 0.19677675 | 0.20558434999999997 | 6325.492379489567 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 10.133129 | 0.101489 | 0.1205892 | 0.12605623 | 9661.604243685755 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 10.722678 | 0.097883 | 0.11525155 | 0.11947793 | 10052.300106996683 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 11.6775 | 0.097442 | 0.11080224999999999 | 0.11621504 | 10092.575154865519 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 9.459702 | 0.1270155 | 0.14818599999999998 | 0.15432236 | 15449.613095339253 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 9.911023 | 0.11693300000000001 | 0.1231733 | 0.12949919999999998 | 16999.725284439402 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 10.630996 | 0.1450845 | 0.15655195 | 0.16524142999999997 | 13701.100088728324 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 10.966571 | 0.117778 | 0.12674345 | 0.12928833 | 16853.942052776434 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 9.719632 | 0.13311050000000002 | 0.15321574999999998 | 0.16190826 | 29507.285865257632 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 10.049261 | 0.13786749999999998 | 0.1547149 | 0.16385386999999998 | 28383.14693884213 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 10.527116 | 0.156968 | 0.17255405000000001 | 0.17939172999999997 | 25202.025738828885 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 11.030026 | 0.15933599999999998 | 0.1745688 | 0.17998330999999998 | 24790.676821423705 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 9.624197 | 0.1661665 | 0.19097689999999998 | 0.19612453999999999 | 47276.339185041004 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 10.13667 | 0.20011600000000002 | 0.2064221 | 0.21289786999999996 | 39940.197542220034 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 10.457767 | 0.21966449999999998 | 0.35294259999999994 | 0.38193558 | 33788.36975592887 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 11.092907 | 0.3367465 | 0.3824818 | 0.4097534599999999 | 23649.165574666444 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 9.585134 | 0.1979735 | 0.22260535 | 0.22457365999999998 | 78966.81399899989 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 9.815758 | 0.29233050000000005 | 0.3074829 | 0.32008208 | 54395.751256660835 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 10.419371 | 0.2938535 | 0.5799957499999999 | 0.61568592 | 47781.64171987153 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 11.026907 | 0.5788025 | 0.5994704 | 0.61413526 | 27859.67555039749 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 9.939713 | 0.3647985 | 0.38486005 | 0.42152120999999987 | 86880.11917345948 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 9.898762 | 0.446864 | 0.464497 | 0.5354804799999997 | 70991.85741143455 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 10.24274 | 0.9231475 | 1.0677301499999998 | 1.07253751 | 34171.756603206086 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 11.341162 | 0.968529 | 1.0098475 | 1.02216906 | 33017.55413346657 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 9.764207 | 0.639594 | 0.65201325 | 0.66182814 | 99919.52731069215 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 10.088061 | 0.7302755000000001 | 0.7532057 | 0.75556175 | 87633.4520547525 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 10.455387 | 1.5712335 | 1.65935255 | 1.66475295 | 40888.76692678354 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 11.202923 | 1.4903265 | 1.55646655 | 1.58726192 | 43062.28504686643 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 9.652884 | 1.147322 | 1.16607735 | 1.17915418 | 111451.72383251968 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 10.02265 | 1.2005745 | 1.2180826 | 1.2376363199999998 | 106478.52798093502 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 10.644293 | 1.2184245 | 1.2545769 | 1.26407859 | 104801.66685741115 | - |
| `separator3_grid_search_6ports_learned_dense_hd256_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 10.890059 | 2.656588 | 2.7706327 | 2.81939817 | 47915.988059605304 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.392195 | 0.014804999999999999 | 0.01527015 | 0.018828439999999988 | 66866.19539372153 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.03507 | 0.0154625 | 0.01607595 | 0.017435239999999998 | 64455.49927874296 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.159007 | 0.015373 | 0.0176048 | 0.020376099999999998 | 62652.08794348281 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.874511 | 0.0153975 | 0.01588415 | 0.020636189999999985 | 64217.16704683229 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.843214 | 0.015428 | 0.01597405 | 0.01622831 | 130980.57296141837 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 3.950627 | 0.015812 | 0.01613495 | 0.020672559999999986 | 124952.36191202105 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.260644 | 0.016132 | 0.0181795 | 0.020727069999999993 | 121477.45742822504 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.95709 | 0.0163605 | 0.018777649999999996 | 0.024321209999999992 | 118696.47530816573 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 3.817819 | 0.0168885 | 0.0172532 | 0.017474669999999998 | 236964.29777407588 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.032117 | 0.017528500000000002 | 0.01785755 | 0.01808351 | 228459.9401206497 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.215848 | 0.017375 | 0.019525249999999997 | 0.024069789999999983 | 224596.96075392707 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.854317 | 0.0179765 | 0.021471399999999998 | 0.025608269999999995 | 214586.7488390857 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.699122 | 0.020332 | 0.021148749999999997 | 0.024180209999999994 | 390563.5930288305 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 3.958766 | 0.020595000000000002 | 0.0211566 | 0.025836739999999987 | 386491.35418840684 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.073703 | 0.020461 | 0.023705999999999998 | 0.029133969999999995 | 376429.49099204224 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.730879 | 0.020713000000000002 | 0.022235599999999994 | 0.027950739999999988 | 381915.1709118129 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.752553 | 0.025661 | 0.0293273 | 0.034114439999999996 | 594163.3845328874 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 3.93744 | 0.0459045 | 0.05619749999999999 | 0.05897428 | 359521.15377528674 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.346558 | 0.044264 | 0.05391699999999999 | 0.06145531999999997 | 386182.02110291657 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 4.74908 | 0.044312000000000004 | 0.05378615 | 0.05755520999999999 | 373024.8914847276 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.67199 | 0.035362500000000005 | 0.038924099999999996 | 0.040437499999999994 | 890289.6112105267 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 3.969928 | 0.0709295 | 0.0776709 | 0.07923755 | 462401.68545414344 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.192403 | 0.06242 | 0.0742478 | 0.07623463999999999 | 506318.0582449307 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.844843 | 0.061618 | 0.07454325 | 0.07633646 | 505887.4229746877 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.817662 | 0.055892 | 0.05951775 | 0.07676087999999995 | 1119910.183203307 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 3.980939 | 0.1027585 | 0.1244738 | 0.12542362 | 591336.3314085336 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.257418 | 0.12492900000000001 | 0.14466335 | 0.15693675999999995 | 508306.8457812914 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.426112 | 0.1173755 | 0.14258094999999998 | 0.16599968999999995 | 539342.8713872876 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.848962 | 0.09436349999999999 | 0.09706745 | 0.09753192000000001 | 1365760.1789145835 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 3.842875 | 0.1527505 | 0.2032653 | 0.20875141 | 784800.6673748675 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.210342 | 0.2011205 | 0.28852944999999997 | 0.30355579 | 601235.9720609401 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.969648 | 0.28367549999999997 | 0.3137348 | 0.31855864 | 445057.10882418207 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.969688 | 0.018126499999999997 | 0.02153345 | 0.03259466999999998 | 51383.23673284827 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.379018 | 0.018445000000000003 | 0.021979749999999985 | 0.02562401 | 53037.342532130024 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.76522 | 0.018695000000000003 | 0.0221502 | 0.041518679999999926 | 49013.64932106293 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.51828 | 0.0184355 | 0.02075734999999999 | 0.027141649999999996 | 53175.64980644064 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.260026 | 0.018868000000000003 | 0.02272945 | 0.024722969999999993 | 101837.55687627551 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.446484 | 0.0194375 | 0.0236798 | 0.027341789999999987 | 100294.56513780975 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.727113 | 0.019964500000000003 | 0.020596200000000002 | 0.02681425999999999 | 98699.92459325762 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.2496 | 0.0198445 | 0.0203497 | 0.02974068999999998 | 99035.98373433003 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.290759 | 0.020971 | 0.0262418 | 0.02779263 | 182780.2706975809 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.467539 | 0.0214885 | 0.0219498 | 0.024876979999999996 | 185248.9931717221 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.665133 | 0.0219 | 0.02377 | 0.030915059999999987 | 179515.3265697942 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.474521 | 0.0217295 | 0.02434714999999999 | 0.03245010999999998 | 180160.05419214428 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.256173 | 0.0258395 | 0.0269324 | 0.029806069999999997 | 308931.3600857593 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.432536 | 0.025820000000000003 | 0.030240599999999992 | 0.032283730000000004 | 306105.2696022162 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.671838 | 0.0258495 | 0.028287999999999994 | 0.03627628999999999 | 306006.21345616423 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.364831 | 0.026138 | 0.030572999999999996 | 0.03593912 | 300906.6316812556 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.301917 | 0.032747 | 0.03594615 | 0.0387914 | 482506.1368749284 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.427108 | 0.0464035 | 0.058814849999999995 | 0.07248199999999999 | 335273.44902502483 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.722597 | 0.047535 | 0.06631479999999997 | 0.08459243999999998 | 317296.3809968025 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.620449 | 0.0459065 | 0.0667868 | 0.07746210999999999 | 331039.23145932023 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.277968 | 0.048536499999999996 | 0.05495955 | 0.05707353 | 654179.2033978067 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.457886 | 0.093967 | 0.11324275 | 0.11562647999999999 | 336913.601875935 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.792335 | 0.1056365 | 0.1201636 | 0.12749641999999997 | 303820.5241017973 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.605104 | 0.0873475 | 0.10634405 | 0.10779614 | 358125.62420177157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.283076 | 0.07615050000000001 | 0.0848205 | 0.0852733 | 826141.172036154 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.442004 | 0.1445775 | 0.18334895 | 0.18571957 | 429312.14534791996 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.529882 | 0.182819 | 0.2078616 | 0.23256713999999995 | 345772.57370844757 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.539948 | 0.1846775 | 0.20972984999999997 | 0.21505485 | 346077.11852730345 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.272856 | 0.134678 | 0.1419753 | 0.14883507999999998 | 943387.1874753558 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.503885 | 0.2216865 | 0.277821 | 0.291894 | 556558.5203891735 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.725372 | 0.275874 | 0.3538671 | 0.37006107999999993 | 451575.4160932306 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.382495 | 0.4132045 | 0.43490435 | 0.43974975 | 309420.91440745484 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.522705 | 0.021079 | 0.02655825 | 0.028400829999999995 | 45409.09507847146 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.907355 | 0.021888499999999998 | 0.027117549999999997 | 0.03139098999999999 | 43812.37787299667 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.200621 | 0.022144 | 0.02265565 | 0.02949844999999999 | 44667.8631698281 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.753475 | 0.022002 | 0.0274082 | 0.03304040999999998 | 43109.39439922748 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.754722 | 0.022098 | 0.0279158 | 0.02833399 | 86380.84971114244 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.941073 | 0.0228165 | 0.023468199999999998 | 0.030132199999999984 | 86570.19284376157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.225338 | 0.023420499999999997 | 0.02396965 | 0.031387119999999984 | 84441.13059918581 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.901627 | 0.0234315 | 0.029075 | 0.030585799999999996 | 82308.38813014273 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.636247 | 0.024834500000000002 | 0.029419699999999993 | 0.03964253999999997 | 156733.5474836037 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.963989 | 0.0257175 | 0.0268558 | 0.03466511999999998 | 153593.19766446183 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.218179 | 0.0259515 | 0.0266899 | 0.03532875999999998 | 152274.44728182498 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.752506 | 0.025807499999999997 | 0.026216299999999998 | 0.031363449999999994 | 153939.34635814143 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.787222 | 0.030321 | 0.0313009 | 0.03765256999999998 | 262685.57915274706 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.860567 | 0.031498 | 0.0371812 | 0.041728339999999996 | 250757.28700676042 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.232034 | 0.031686 | 0.03550904999999999 | 0.04138421999999999 | 250669.9153487696 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.069008 | 0.031412499999999996 | 0.034662649999999996 | 0.037686109999999995 | 253155.90479819046 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.721737 | 0.040521 | 0.04479345 | 0.05253724 | 392227.61753100564 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.921245 | 0.0548115 | 0.07533295 | 0.08247434 | 271055.9662806378 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.185456 | 0.0539945 | 0.0685009 | 0.07010952 | 285523.3929315829 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.842719 | 0.05634 | 0.07255015 | 0.07673322999999999 | 270738.4832079532 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.659051 | 0.059799500000000005 | 0.06823175 | 0.08775931999999993 | 520825.8735877481 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.904961 | 0.13169 | 0.14099015 | 0.14482557000000001 | 247858.732915175 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.206886 | 0.1144985 | 0.12877235 | 0.133797 | 276333.83752744773 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.887786 | 0.113184 | 0.13423844999999998 | 0.14479577999999999 | 276980.83275326045 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.750629 | 0.09825300000000001 | 0.1076953 | 0.12504128999999994 | 638574.3189754554 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.93109 | 0.185418 | 0.1937901 | 0.19514561 | 344750.61171686667 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.218262 | 0.21854 | 0.252452 | 0.25637828 | 287611.9338460196 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 14.690897 | 0.25227 | 0.2952803 | 0.29928663 | 249703.96424551363 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.783839 | 0.174672 | 0.1826501 | 0.18330211 | 728389.0662882989 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.846693 | 0.28875399999999996 | 0.3096966 | 0.31259451 | 440792.5394760712 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.181055 | 0.3655405 | 0.46128939999999996 | 0.5164945799999998 | 334464.3978587798 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.722106 | 0.4833205 | 0.550179 | 0.56755875 | 266074.4514563377 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.936977 | 0.0247135 | 0.03186725 | 0.03369244999999999 | 38954.15874598772 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.367949 | 0.025177 | 0.0256876 | 0.032323409999999976 | 39375.56651596325 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.611723 | 0.0248235 | 0.02536395 | 0.03596060999999998 | 39584.42685312515 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.202403 | 0.024943 | 0.028501749999999985 | 0.0328964 | 39445.459945896604 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.143366 | 0.0253995 | 0.02601845 | 0.02912652999999999 | 78490.90251194435 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.372568 | 0.026041500000000002 | 0.029324999999999986 | 0.03613248999999998 | 75409.49239608384 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.565349 | 0.0266815 | 0.03263919999999999 | 0.03705870999999999 | 73305.82898629767 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.298631 | 0.026996 | 0.02754895 | 0.029897949999999993 | 73886.40261144102 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.262116 | 0.0289305 | 0.02966025 | 0.030260099999999998 | 137968.9997454472 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.401673 | 0.02996 | 0.033472199999999994 | 0.040529639999999985 | 131143.41319092907 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.644221 | 0.0291695 | 0.0296997 | 0.0346281 | 136704.46545136397 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.290333 | 0.0305425 | 0.0370028 | 0.0381382 | 128840.82553470554 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.282005 | 0.0358005 | 0.037294999999999995 | 0.04075091999999999 | 223275.5868240611 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.474294 | 0.03714 | 0.03786965 | 0.04672978999999998 | 214027.1236573811 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.707958 | 0.036352999999999996 | 0.036795749999999995 | 0.042793659999999976 | 222327.82794049397 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.542749 | 0.0370205 | 0.03989079999999999 | 0.043659729999999994 | 216987.04804310232 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.278364 | 0.0463975 | 0.04855345 | 0.050769129999999996 | 342595.2532571173 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.382587 | 0.070186 | 0.0824686 | 0.08432257 | 222838.92895811377 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.676044 | 0.068463 | 0.08174124999999999 | 0.08695425 | 227054.18761714577 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.513112 | 0.073114 | 0.08382869999999999 | 0.08854077999999999 | 217786.10083659808 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.343476 | 0.071726 | 0.07412685 | 0.07719710999999999 | 443535.4431944753 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.46694 | 0.143374 | 0.15543274999999998 | 0.15925837999999998 | 222643.85402077404 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.681619 | 0.13979 | 0.1597755 | 0.17274812999999997 | 222470.52404600466 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.425628 | 0.13913199999999998 | 0.15365555 | 0.16306547999999998 | 226356.75408941766 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.330944 | 0.12034600000000001 | 0.13411035 | 0.13470457 | 518406.25716352393 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.382611 | 0.226134 | 0.2433666 | 0.24711515 | 280004.49407212983 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.662761 | 0.2732325 | 0.2958552 | 0.31339866999999993 | 231524.49037118585 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.599119 | 0.2949375 | 0.3178495 | 0.3240847 | 216257.36356322933 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.228516 | 0.2223655 | 0.23461135 | 0.23878189 | 571660.8596939703 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.38808 | 0.35467550000000003 | 0.37937455 | 0.39324649999999994 | 358764.22317959944 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.739722 | 0.4452775 | 0.5502751499999998 | 0.60521249 | 280431.8387411765 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.188275 | 0.667768 | 0.69572825 | 0.72257533 | 191750.52106706047 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.805887 | 0.0157275 | 0.01755804999999999 | 0.02375882999999999 | 62116.66254626138 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.23209 | 0.0161865 | 0.0186238 | 0.023844759999999996 | 60196.72289040584 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.50914 | 0.016013 | 0.018292799999999998 | 0.022129169999999993 | 59952.90100097364 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.243952 | 0.0159465 | 0.0166064 | 0.018533269999999994 | 62341.57447386828 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.071103 | 0.0165145 | 0.018588749999999998 | 0.02260705999999999 | 116858.88004786539 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.143373 | 0.016940999999999998 | 0.018379599999999992 | 0.023181969999999986 | 115912.86598038522 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.353883 | 0.01693 | 0.01930704999999999 | 0.02264179 | 116472.60389647448 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.172373 | 0.017313500000000002 | 0.017681699999999998 | 0.02331894 | 114159.94264604482 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.02723 | 0.0178925 | 0.018497299999999998 | 0.02166748999999999 | 221475.11284157 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.097301 | 0.0183325 | 0.021167299999999997 | 0.022700719999999997 | 209461.37008682176 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.235758 | 0.0185425 | 0.01898745 | 0.023884249999999992 | 213352.90490647676 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 4.922647 | 0.018873 | 0.020938899999999993 | 0.02691569999999999 | 208268.03262727 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.038978 | 0.0211835 | 0.024176699999999995 | 0.027042159999999992 | 372023.11751652247 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.053547 | 0.021648 | 0.021990100000000002 | 0.02667868999999999 | 368892.40976199985 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.291695 | 0.021406 | 0.0244851 | 0.027517249999999993 | 367504.8533609697 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.042583 | 0.021478499999999998 | 0.027873299999999983 | 0.03343375 | 358941.0878965963 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.950009 | 0.026344 | 0.029861099999999998 | 0.03634058 | 578027.5820311455 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.238829 | 0.033298 | 0.045760249999999995 | 0.04892004 | 464998.40738045477 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.339398 | 0.033462000000000006 | 0.04197325 | 0.04312161 | 455676.61997310375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.186934 | 0.036684999999999995 | 0.04773859999999999 | 0.057209709999999976 | 410043.81830753386 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.013035 | 0.038007 | 0.04337664999999999 | 0.049071609999999995 | 815101.7985730624 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.161034 | 0.067905 | 0.07948229999999999 | 0.08342965999999999 | 473182.9479061063 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.370004 | 0.0659815 | 0.0766959 | 0.08051534 | 479092.83771179273 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.25413 | 0.07034 | 0.08715655 | 0.08788799 | 445992.7827217936 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.052948 | 0.058554 | 0.061539949999999996 | 0.06515676999999999 | 1085990.4181707918 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.176731 | 0.11370849999999999 | 0.1317939 | 0.13241659 | 552344.3825799557 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.411839 | 0.13492949999999998 | 0.15166065 | 0.16476550999999998 | 474138.69742246345 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 4.813579 | 0.1420315 | 0.16038644999999996 | 0.16932751999999998 | 449235.8568268479 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.00774 | 0.10008 | 0.1026353 | 0.12043026999999994 | 1280305.737009998 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.178311 | 0.1714685 | 0.2271987 | 0.23146432 | 700136.3953215136 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.512033 | 0.200164 | 0.25442475000000003 | 0.27887906 | 605973.0957413821 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.30094 | 0.2864025 | 0.3083483 | 0.31400564999999997 | 447915.7918311357 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.60242 | 0.0184735 | 0.01965935 | 0.029761109999999966 | 52833.51420006361 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.677882 | 0.0197745 | 0.0238426 | 0.028008189999999992 | 48046.80142833531 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.974265 | 0.0197845 | 0.02038455 | 0.027040329999999977 | 49963.227064880244 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.694962 | 0.019766 | 0.020125249999999997 | 0.03281094999999997 | 49545.22438536672 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.6055 | 0.019903999999999998 | 0.023816449999999996 | 0.024119830000000002 | 98349.78889217816 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.782606 | 0.020156 | 0.024684400000000002 | 0.027785169999999998 | 96537.39665671688 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.964837 | 0.020887 | 0.0250181 | 0.029324229999999982 | 91198.7713701521 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.747433 | 0.020915 | 0.025601699999999998 | 0.030177529999999998 | 92379.09424145678 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.61129 | 0.0220425 | 0.022925749999999998 | 0.025465619999999994 | 180513.88693332177 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.795454 | 0.022387 | 0.02814315 | 0.051940899999999915 | 167488.7573171651 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.964019 | 0.022551500000000002 | 0.02732855 | 0.028039579999999998 | 174061.67701463337 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.571857 | 0.0232505 | 0.026413849999999996 | 0.03298600999999999 | 168583.3185120499 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.555834 | 0.026437000000000002 | 0.027032149999999998 | 0.030055769999999992 | 301442.70478510146 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.740332 | 0.027145000000000002 | 0.03130189999999999 | 0.042290029999999985 | 289163.5943034772 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.85567 | 0.0272905 | 0.029209199999999998 | 0.03579433999999999 | 291083.95297829824 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.674562 | 0.02779 | 0.0282443 | 0.03360569999999998 | 287729.4192541334 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.588192 | 0.034394 | 0.039049549999999995 | 0.04156426999999999 | 453997.5620330919 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.690592 | 0.047326 | 0.06242174999999999 | 0.06538920999999999 | 323226.0546058097 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.043437 | 0.0461445 | 0.0504283 | 0.051132899999999995 | 347126.25025112415 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.687465 | 0.045355 | 0.05506724999999999 | 0.060726869999999995 | 341094.07630445034 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.58682 | 0.048959 | 0.0563944 | 0.05733576 | 637785.354136297 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.665302 | 0.09850149999999999 | 0.11583435 | 0.11712364 | 325630.1452106929 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.953938 | 0.09303249999999999 | 0.10275885 | 0.10938246 | 346230.7160310881 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.344097 | 0.0870845 | 0.10785604999999998 | 0.11247940999999999 | 360115.1738354719 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.595093 | 0.079126 | 0.0834452 | 0.08494465 | 796316.835556343 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.606695 | 0.1546365 | 0.19367625 | 0.20066319 | 404604.551523039 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.819932 | 0.15984399999999999 | 0.1761365 | 0.17993596999999997 | 396019.3618816266 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.703782 | 0.15507749999999998 | 0.16644955 | 0.16868645 | 410896.19670628174 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.544462 | 0.1399245 | 0.14873505 | 0.15244750999999998 | 904582.6439424199 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.714754 | 0.2460215 | 0.31054855 | 0.31763484 | 513123.8232727791 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.981719 | 0.25163650000000004 | 0.2754853 | 0.27878071 | 504570.10430015926 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.001792 | 0.23291 | 0.25403129999999996 | 0.25851909 | 545779.5168435661 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.715945 | 0.0228455 | 0.023262249999999998 | 0.031032269999999987 | 43214.539445367314 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.442404 | 0.023314500000000002 | 0.0237598 | 0.03188988999999997 | 42350.16300577741 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.463281 | 0.0234425 | 0.02412715 | 0.02766596999999999 | 42384.87843169168 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.133635 | 0.023418 | 0.02399845 | 0.033351849999999975 | 42018.11316822456 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.207573 | 0.0234235 | 0.02386435 | 0.02671581999999999 | 85011.17046779947 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.301897 | 0.024194 | 0.027772299999999986 | 0.03610066999999998 | 80759.46198046429 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.518008 | 0.025245499999999997 | 0.02562345 | 0.03014878 | 78756.83904701074 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.897819 | 0.0249505 | 0.027047749999999992 | 0.03527813999999999 | 78727.51148634394 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.145092 | 0.026147 | 0.0266506 | 0.03023228999999999 | 152109.64671774002 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.311271 | 0.02735 | 0.02990464999999999 | 0.03467941 | 144395.15756399592 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.37918 | 0.0273785 | 0.030888399999999986 | 0.036865979999999986 | 143875.26589947578 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.55398 | 0.027791 | 0.034525850000000004 | 0.03891248999999999 | 139818.29214752506 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.131662 | 0.032143500000000005 | 0.03274505 | 0.035607079999999985 | 248355.11304193415 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.278894 | 0.033702 | 0.03600385 | 0.043226109999999984 | 237897.13209059834 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.485552 | 0.0336245 | 0.03892289999999998 | 0.04320618 | 236693.81347462375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.373075 | 0.033505 | 0.03417265 | 0.039835609999999994 | 240264.48314304382 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.091178 | 0.0422275 | 0.0458705 | 0.048893 | 377787.42185113 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.212949 | 0.0593925 | 0.07088054999999999 | 0.07567426 | 266652.978480438 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.475802 | 0.0602265 | 0.0782652 | 0.08278474 | 259406.7367929545 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.25861 | 0.0597945 | 0.0669172 | 0.07092404999999999 | 265566.513124961 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.190097 | 0.062919 | 0.0672883 | 0.07215072999999998 | 501331.0338949912 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.223574 | 0.1160215 | 0.12902705 | 0.13061187 | 274770.4250166107 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.525318 | 0.1176605 | 0.1340477 | 0.13827894999999998 | 269235.9403729894 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.93545 | 0.117565 | 0.1282288 | 0.1333501 | 271984.0454158959 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.197573 | 0.10197300000000001 | 0.10704025 | 0.10955427999999999 | 620992.6567618338 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.312177 | 0.200878 | 0.26291844999999997 | 0.265348 | 312396.3356690817 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.552164 | 0.20322099999999998 | 0.23656449999999998 | 0.24175504 | 309115.21854397655 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.138052 | 0.198624 | 0.22127555 | 0.2298988 | 317463.4039100777 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.089825 | 0.1819335 | 0.19383799999999998 | 0.19580273 | 696064.5164798712 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.307105 | 0.309293 | 0.3234683 | 0.32622934 | 411234.1193752249 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.581444 | 0.3150665 | 0.33244255 | 0.33332113 | 405339.6147462299 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.141363 | 0.30764400000000003 | 0.3195226 | 0.32670934 | 416637.7244497404 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.314465 | 0.026564499999999998 | 0.034041749999999996 | 0.03689776 | 36147.14780930211 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.926781 | 0.027074 | 0.03195609999999999 | 0.03882840999999998 | 36097.82510603736 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.937727 | 0.026924 | 0.0295899 | 0.033112220000000005 | 36656.327138658424 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.737789 | 0.0271465 | 0.029345999999999994 | 0.03266776 | 36624.56306896259 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.77729 | 0.0267805 | 0.0272319 | 0.03415850999999998 | 73972.8682313901 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.783867 | 0.0282865 | 0.0290315 | 0.03316648 | 70383.72503049375 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.303655 | 0.028763999999999998 | 0.03219535 | 0.035312199999999995 | 68742.74334415574 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.706741 | 0.028396 | 0.030142299999999993 | 0.03439539999999999 | 69746.58276617737 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.842618 | 0.030747 | 0.0311949 | 0.03639690999999998 | 130169.51976559073 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.981922 | 0.0312265 | 0.033192049999999994 | 0.037910379999999994 | 126755.88590956222 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.148744 | 0.032208 | 0.0338553 | 0.035903229999999994 | 123238.38432013389 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.203729 | 0.0328615 | 0.037714799999999986 | 0.04101128 | 120043.40769622294 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.752659 | 0.0386235 | 0.03985785 | 0.043369639999999994 | 207141.62169104203 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.801081 | 0.0391585 | 0.042300349999999994 | 0.04475221 | 203069.70316793813 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.160809 | 0.0393665 | 0.0412677 | 0.046239529999999994 | 202554.5162820916 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.911259 | 0.039393 | 0.04137129999999999 | 0.043859989999999995 | 205450.7100633353 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.904492 | 0.0513265 | 0.0591709 | 0.06130716 | 307633.73896123626 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.803464 | 0.07082350000000001 | 0.07855055 | 0.08628043999999999 | 220889.58309576198 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.016557 | 0.0716935 | 0.0858519 | 0.08800835 | 217290.98440692734 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.074704 | 0.07308500000000001 | 0.08410454999999999 | 0.09293401999999996 | 214487.21736617206 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.760988 | 0.0757605 | 0.0783024 | 0.08062857999999999 | 420448.29248065024 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.926718 | 0.1521615 | 0.1601008 | 0.16166320999999997 | 211341.51004829817 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.144844 | 0.14205050000000002 | 0.16010139999999998 | 0.17210713 | 223011.6524982253 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.863969 | 0.14471050000000002 | 0.16176545 | 0.16442294999999998 | 219628.62172744775 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.591261 | 0.1406435 | 0.15169065 | 0.15340502 | 450390.2842933848 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.87579 | 0.2458825 | 0.2698057 | 0.27541509000000003 | 258262.08662530058 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.201457 | 0.2521865 | 0.28387995 | 0.29200307999999997 | 249194.7117144254 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.444593 | 0.24896400000000002 | 0.2717871 | 0.27532660000000003 | 255012.16447870212 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.813582 | 0.22745749999999998 | 0.2394726 | 0.24144195999999998 | 558677.0179042891 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.841889 | 0.3818625 | 0.4167008 | 0.42509769999999997 | 331369.7947112346 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.035438 | 0.39918549999999997 | 0.42289715 | 0.42918502 | 320472.00719579833 | - |
| `separator3_grid_search_6ports_learned_dense_hd32_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.866365 | 0.376939 | 0.3988855 | 0.4054587 | 337024.52422142593 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.564461 | 0.014298 | 0.0163915 | 0.022518559999999976 | 66330.85565477177 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.044886 | 0.015989499999999997 | 0.01965325 | 0.02635924999999998 | 60439.321338948546 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.538639 | 0.016952000000000002 | 0.01827995 | 0.02382571999999998 | 58279.88096917111 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 4.993977 | 0.015921499999999998 | 0.01646785 | 0.021063409999999987 | 61887.31557579958 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 3.990899 | 0.016348 | 0.0187831 | 0.019194370000000002 | 120011.56911526273 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.17769 | 0.016641000000000003 | 0.01709745 | 0.0954425799999997 | 101482.66168725073 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.39172 | 0.017395 | 0.01798 | 0.022780449999999983 | 114153.81770320235 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.132152 | 0.017021 | 0.018297349999999997 | 0.022874959999999986 | 115901.58101346661 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.014963 | 0.0183455 | 0.01879565 | 0.02358366999999999 | 215720.4090921838 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.18224 | 0.0189475 | 0.019518849999999997 | 0.023419739999999984 | 209187.95328414626 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.281728 | 0.018805500000000003 | 0.019411849999999998 | 0.023669039999999985 | 210792.8022689737 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.288728 | 0.019452 | 0.02246745 | 0.02771661999999999 | 197069.96377854067 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 3.816036 | 0.02237 | 0.024713199999999998 | 0.027741129999999992 | 353789.4386776766 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.044166 | 0.039777 | 0.04806319999999999 | 0.05680341999999998 | 205282.1140773236 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.216858 | 0.039006 | 0.0455121 | 0.04727024999999999 | 218558.2151661917 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 4.803919 | 0.039986499999999994 | 0.0455951 | 0.04769831 | 207650.89731144003 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 3.930009 | 0.028376 | 0.03189595 | 0.03389317 | 553302.385286583 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.110504 | 0.052825 | 0.06390725 | 0.07007898 | 294680.139442642 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.384593 | 0.051196500000000006 | 0.059767499999999994 | 0.06724663999999998 | 307672.54564802954 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.063132 | 0.052668 | 0.0575275 | 0.05915781999999999 | 308149.0794431438 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 3.947215 | 0.041236499999999995 | 0.04547615 | 0.048530069999999995 | 774672.0958269383 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.122877 | 0.08768899999999999 | 0.09236805 | 0.09917184999999998 | 363825.71838525054 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.408688 | 0.0779135 | 0.08522025 | 0.08657967999999999 | 410470.1704913508 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.05536 | 0.074087 | 0.08127895 | 0.08398868 | 431029.9541572704 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 3.98453 | 0.065584 | 0.0687033 | 0.06905691 | 969993.5525741054 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.148935 | 0.1140805 | 0.1537918 | 0.15678224 | 507590.94324514625 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.293968 | 0.1326325 | 0.1432935 | 0.14454381 | 496546.05666397384 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.251112 | 0.112843 | 0.13541579999999998 | 0.1366375 | 545082.6813775398 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 3.997257 | 0.115094 | 0.11727625 | 0.1198012 | 1119308.2954561156 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.074802 | 0.179994 | 0.2518457 | 0.2540025 | 673906.1029779067 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.343991 | 0.178869 | 0.21722439999999998 | 0.21960358000000002 | 700371.306225316 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 4.830581 | 0.16862850000000001 | 0.22291535 | 0.22699036 | 711906.8647844145 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.982766 | 0.019174999999999998 | 0.0237924 | 0.03540678999999998 | 49860.83840002553 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.756705 | 0.019702499999999998 | 0.024432049999999997 | 0.0254842 | 49524.956616138006 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.950655 | 0.019512500000000002 | 0.02188994999999999 | 0.025642149999999996 | 50420.20196316098 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.576294 | 0.0196635 | 0.021260149999999995 | 0.02984874999999998 | 49837.132251801115 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.781709 | 0.0204225 | 0.020892149999999998 | 0.023452299999999992 | 97500.28762584849 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.810811 | 0.0209175 | 0.025811149999999998 | 0.028331899999999993 | 91878.74578161708 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.841306 | 0.0213315 | 0.023230599999999994 | 0.028933119999999996 | 92109.95349368449 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 14.558646 | 0.0202835 | 0.02415739999999999 | 0.029506909999999997 | 95035.80473943558 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.524841 | 0.0228415 | 0.0244243 | 0.027019449999999993 | 173900.34119246944 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.673945 | 0.023577 | 0.027401549999999993 | 0.03417142999999998 | 165816.5803315005 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.067812 | 0.023418500000000002 | 0.0292552 | 0.03247253999999999 | 163398.02533486384 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.316652 | 0.023804 | 0.0297532 | 0.033951899999999986 | 161296.56631869622 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.489379 | 0.0286755 | 0.030640299999999995 | 0.03372796 | 277891.82921601855 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.704884 | 0.042868 | 0.061432299999999995 | 0.06630057999999998 | 182270.62722512567 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.012054 | 0.0409015 | 0.05324965 | 0.06877804999999995 | 184780.3792802065 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.446213 | 0.048637 | 0.0701954 | 0.07138600999999999 | 163680.72111178492 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.541622 | 0.0377 | 0.039173549999999994 | 0.04209356999999999 | 429674.8489290086 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.79336 | 0.071962 | 0.08367879999999998 | 0.08935537999999998 | 219656.8136857178 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.934454 | 0.0755755 | 0.08353764999999999 | 0.0886361 | 215011.7436726747 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.935711 | 0.072255 | 0.08512639999999999 | 0.08787054 | 216550.6406380015 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.429224 | 0.0558285 | 0.06311295 | 0.06855127 | 566863.4982067628 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.734191 | 0.1313085 | 0.14060915000000002 | 0.14890003 | 259571.49612954565 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.970003 | 0.1125585 | 0.12630015 | 0.12842975 | 282559.0596999611 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.628089 | 0.1197055 | 0.12633525 | 0.13388556 | 272559.86521914665 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.515608 | 0.091468 | 0.09887554999999999 | 0.09970834999999999 | 691338.2233990335 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.714649 | 0.1701495 | 0.1855095 | 0.19620616 | 380527.09662106214 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.978644 | 0.163022 | 0.2078509 | 0.21684493 | 371731.23367141513 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.372284 | 0.16642099999999999 | 0.18331215 | 0.18519154 | 381275.8299481203 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.416478 | 0.165679 | 0.17298945 | 0.17364624 | 768880.7067167016 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.736352 | 0.2526775 | 0.30931585 | 0.31593121 | 489652.68246658565 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.902864 | 0.252693 | 0.26457275 | 0.27007575 | 504693.0937639333 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.347651 | 0.238738 | 0.2560052 | 0.25686632 | 531482.8429449099 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.402757 | 0.023143 | 0.0297 | 0.043054209999999954 | 40415.50371054739 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.3126 | 0.023532 | 0.02582249999999999 | 0.03144222 | 41838.938495923634 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.585391 | 0.023166 | 0.029420449999999997 | 0.029850309999999998 | 41732.4648529181 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.093688 | 0.0231935 | 0.02571304999999999 | 0.028974879999999998 | 42635.92332355549 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.085464 | 0.024118 | 0.025598599999999996 | 0.03039306 | 81907.66221607733 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.257952 | 0.024311 | 0.026672199999999993 | 0.03247576999999999 | 81102.93503411594 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.554187 | 0.0251875 | 0.030384949999999997 | 0.03670188999999999 | 77426.46808326132 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.295326 | 0.0254345 | 0.028510399999999995 | 0.03224194999999999 | 77652.53287031717 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.043985 | 0.026973999999999998 | 0.02947004999999999 | 0.03149425 | 147230.41191388495 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.375281 | 0.028255500000000003 | 0.03213004999999999 | 0.040331189999999975 | 138762.6534194587 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.499768 | 0.027842 | 0.032010699999999996 | 0.04037802999999999 | 140632.93257635314 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.602666 | 0.0289775 | 0.030774849999999992 | 0.03374604999999999 | 137269.0191373603 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.186735 | 0.034716 | 0.03635769999999999 | 0.037998359999999995 | 231894.9469511322 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.288211 | 0.0543495 | 0.0739048 | 0.08366005999999998 | 141492.09784320058 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.564715 | 0.0511905 | 0.07113659999999998 | 0.08534331999999997 | 147626.64336134057 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.370926 | 0.049652 | 0.0685583 | 0.07223964999999999 | 151561.6533860768 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.065536 | 0.044649 | 0.0482096 | 0.05231272999999999 | 349380.9842411707 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.233436 | 0.098801 | 0.1109445 | 0.11531039 | 165325.94216154577 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.444416 | 0.099488 | 0.11059834999999998 | 0.11302589 | 161711.71854075388 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 15.634502 | 0.0886465 | 0.09354769999999998 | 0.09692411 | 180633.39098549064 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.080717 | 0.07036300000000001 | 0.07463884999999999 | 0.07843132 | 449511.31096883456 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.192825 | 0.140476 | 0.18480415 | 0.18817668999999998 | 210645.8018489172 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.457534 | 0.1494055 | 0.16422499999999998 | 0.17126611 | 215060.58525512367 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.920137 | 0.142349 | 0.1713086 | 0.17603294 | 215194.88384923394 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.053256 | 0.11866550000000001 | 0.12766735 | 0.13044578 | 533127.1908195498 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.381911 | 0.209837 | 0.2257746 | 0.22891797 | 301646.21536747954 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.546808 | 0.2122685 | 0.23726204999999997 | 0.24471474 | 294306.0786897199 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.454174 | 0.21437250000000002 | 0.2339127 | 0.23796941 | 295417.2112652184 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.220295 | 0.2167715 | 0.22557655 | 0.22843286999999998 | 586835.3573804403 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.223332 | 0.3288105 | 0.35651625 | 0.36215034 | 386646.76724637905 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.501122 | 0.3264825 | 0.33912175 | 0.34293196000000004 | 390671.73563243647 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.658867 | 0.3113365 | 0.32041929999999996 | 0.32618763 | 411781.6653827443 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.11601 | 0.026146 | 0.027583849999999997 | 0.03190355999999999 | 37891.877800209775 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.876212 | 0.027097 | 0.02768815 | 0.0326362 | 36592.69331736916 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.09099 | 0.02715 | 0.029028199999999994 | 0.03255816999999999 | 36521.5155552439 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.827519 | 0.026908 | 0.0274623 | 0.03360186 | 36826.200221104504 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.690671 | 0.027416000000000003 | 0.02781075 | 0.029444239999999997 | 72701.3117497679 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.800347 | 0.028687 | 0.0305759 | 0.03743242999999999 | 68770.39901960919 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.096098 | 0.029529 | 0.03194394999999999 | 0.03929843999999998 | 66847.19866786902 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 15.901321 | 0.0284325 | 0.0293697 | 0.03290074999999999 | 69841.2299319956 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.692436 | 0.0319305 | 0.034741449999999986 | 0.03977749 | 124451.3252199366 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.892812 | 0.032248 | 0.0328532 | 0.03775124999999999 | 124296.17292083576 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.091597 | 0.032591999999999996 | 0.036323149999999985 | 0.040374299999999995 | 121574.04345542609 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.584327 | 0.033317 | 0.03390795 | 0.04400458999999999 | 118526.38515860015 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.701851 | 0.041098499999999996 | 0.041807 | 0.04568801 | 196130.63673320887 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.10201 | 0.0661305 | 0.08593929999999997 | 0.09576359999999998 | 118768.58357181077 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.146752 | 0.060441999999999996 | 0.0702406 | 0.07384969999999999 | 128975.38408063282 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.713261 | 0.061267 | 0.07419049999999999 | 0.08042590999999998 | 127054.4304356347 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.771634 | 0.055579500000000004 | 0.07122715 | 0.07439317999999999 | 281110.4989149135 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.819705 | 0.116454 | 0.13734995 | 0.14973219000000001 | 135933.88447726797 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.006126 | 0.119526 | 0.1336079 | 0.13524216 | 134247.78449203155 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.919264 | 0.11833199999999999 | 0.12928444999999997 | 0.19971142999999975 | 133104.03944138897 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.649818 | 0.08591299999999999 | 0.0899035 | 0.09062825 | 370065.8370253183 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.814817 | 0.1740985 | 0.20976240000000002 | 0.21775857999999998 | 177023.22611110288 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.039781 | 0.1774925 | 0.18939215 | 0.19323358 | 179840.28608992713 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 6.656391 | 0.1779465 | 0.192544 | 0.19506823999999998 | 179716.86282200628 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.585966 | 0.147538 | 0.1611203 | 0.16171154 | 423919.1617952393 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.697798 | 0.257742 | 0.2928632499999999 | 0.31210591 | 245126.20169290283 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.254881 | 0.266942 | 0.29809805 | 0.30684763 | 237901.71284029144 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.363916 | 0.26594799999999996 | 0.2838076 | 0.28554899 | 239856.59573782326 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.729123 | 0.270658 | 0.28475110000000003 | 0.28679813 | 468644.18892921833 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.895246 | 0.4019955 | 0.43445564999999997 | 0.44104616 | 316092.63667635707 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.138384 | 0.40521450000000003 | 0.4267357 | 0.42784521000000003 | 314538.92795205716 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth2_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.518339 | 0.389976 | 0.40089949999999996 | 0.40525972 | 329658.907079765 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.698373 | 0.0168285 | 0.0192815 | 0.029939469999999975 | 55892.15050638288 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 4.337925 | 0.0171065 | 0.019128299999999994 | 0.022924339999999994 | 57536.6047879661 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 4.603741 | 0.016959000000000002 | 0.0194883 | 0.024773079999999986 | 56691.592069980106 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.081863 | 0.017197 | 0.0197298 | 0.025624659999999983 | 55223.05142701887 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.177955 | 0.0176755 | 0.01820305 | 0.021102619999999996 | 112439.64801892584 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 4.237712 | 0.018281 | 0.023210099999999994 | 0.02436205 | 104617.50271743962 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 4.578086 | 0.018708500000000003 | 0.021492449999999996 | 0.022486359999999997 | 104770.51590049735 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 4.990201 | 0.018585 | 0.020481299999999997 | 0.02459453 | 105855.27889690331 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.273391 | 0.019912 | 0.021414299999999997 | 0.025917099999999995 | 198785.61865563274 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 4.368604 | 0.0199555 | 0.021546249999999992 | 0.028790429999999992 | 197271.73194717066 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 4.588096 | 0.020132999999999998 | 0.022000299999999993 | 0.02643738 | 195372.21826593482 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 5.38213 | 0.020446 | 0.0236089 | 0.027867899999999994 | 191668.55140742217 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.169778 | 0.024121 | 0.026053599999999993 | 0.03006051 | 328912.746026734 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 4.37185 | 0.0323155 | 0.042953649999999996 | 0.04408943 | 233135.8284306811 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 4.515159 | 0.036123 | 0.04201405 | 0.047037489999999994 | 226208.39108716318 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.906344 | 0.035047499999999995 | 0.045660599999999996 | 0.049204929999999994 | 231348.2687919861 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.184395 | 0.0315955 | 0.035281299999999995 | 0.037805349999999995 | 488854.42465195083 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 4.356792 | 0.056823 | 0.06631465 | 0.06863069 | 282717.28059902135 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 4.779397 | 0.058474 | 0.06887775 | 0.08150809999999996 | 263376.47988774895 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.009173 | 0.0600375 | 0.06986085 | 0.07230639999999999 | 262786.6221900309 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.143329 | 0.0455585 | 0.0488536 | 0.052330949999999994 | 702355.7010212252 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 4.382288 | 0.096995 | 0.10980799999999999 | 0.1115402 | 331382.04538939876 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 4.578838 | 0.09449350000000001 | 0.1005001 | 0.10189578999999999 | 338743.84888806276 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 4.990535 | 0.093731 | 0.10558375 | 0.10901075 | 339088.6568486372 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.270292 | 0.07305249999999999 | 0.07689235 | 0.08163691 | 873393.6720990842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 4.341178 | 0.137034 | 0.18556555 | 0.1899653 | 444235.28366505215 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 4.64218 | 0.13664300000000001 | 0.16962454999999999 | 0.17733926 | 449303.72665147553 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.17464 | 0.132087 | 0.1510097 | 0.15613058999999999 | 475249.5951467511 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.129658 | 0.1271205 | 0.13331825 | 0.13490723 | 1005045.8010364223 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 4.392948 | 0.2030165 | 0.2980582 | 0.30166265 | 607315.1489260107 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 4.475878 | 0.197295 | 0.24197819999999998 | 0.25233286 | 633538.4777119679 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.077069 | 0.192195 | 0.24086155 | 0.2701637499999999 | 637020.0203448269 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.348228 | 0.020574000000000002 | 0.021615549999999997 | 0.025220669999999997 | 48161.99382948536 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.102987 | 0.0211765 | 0.02341595 | 0.030127619999999994 | 46371.736241274 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.203705 | 0.021207499999999997 | 0.025858 | 0.030699289999999983 | 45039.688973923825 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 5.745194 | 0.020875499999999998 | 0.02140125 | 0.02668091 | 47502.05683906114 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.902013 | 0.0212395 | 0.021803649999999997 | 0.02550891999999999 | 93477.86257258557 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.012244 | 0.022447500000000002 | 0.024332599999999992 | 0.030602989999999986 | 87614.31477720555 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.100063 | 0.022625 | 0.027534799999999998 | 0.032478869999999986 | 85867.49786404599 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 5.763518 | 0.022527 | 0.023279499999999998 | 0.03424456999999998 | 86903.4728365815 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.837728 | 0.024796 | 0.0262166 | 0.028238019999999996 | 160498.83036477372 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.176373 | 0.0254155 | 0.0258849 | 0.03305742 | 156129.2437880077 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.149749 | 0.0251925 | 0.0257996 | 0.036466309999999974 | 156209.8491872011 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.212437 | 0.025968 | 0.030740299999999998 | 0.033069789999999995 | 151366.5370969109 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.92212 | 0.031146 | 0.03564614999999999 | 0.03985258 | 256016.87663250763 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.082851 | 0.0432 | 0.052287699999999986 | 0.059213099999999984 | 180503.9489751437 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.168172 | 0.044747999999999996 | 0.057615049999999994 | 0.06466812999999999 | 170886.05702211394 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 5.668665 | 0.042715 | 0.0521438 | 0.05893495999999998 | 183205.12791153023 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.879197 | 0.0421465 | 0.049435400000000004 | 0.052548529999999996 | 365979.79883005406 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 5.026824 | 0.084982 | 0.09905305 | 0.10023664 | 186744.93769488586 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.213114 | 0.08866750000000001 | 0.09714655 | 0.10370512 | 180706.2225058306 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 5.884115 | 0.0815485 | 0.09378979999999999 | 0.10022376999999999 | 193955.2394947175 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.831476 | 0.06061950000000001 | 0.06347345 | 0.06477343999999999 | 523477.6452143347 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.088568 | 0.1377885 | 0.14690804999999998 | 0.149732 | 234676.43546442027 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.174592 | 0.14323249999999998 | 0.1536305 | 0.15537581 | 229944.82330249346 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 5.801112 | 0.137486 | 0.14704725 | 0.14909171 | 234072.2946911672 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.914819 | 0.10505149999999999 | 0.11132635 | 0.11166729 | 607012.1666104232 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.089979 | 0.18909700000000002 | 0.20907845 | 0.21048835999999999 | 330683.3540509899 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.265183 | 0.19594 | 0.23023765 | 0.24051617 | 319670.73913868715 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 5.780383 | 0.1907885 | 0.22519009999999998 | 0.22762401000000002 | 328316.48642262566 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.846575 | 0.187216 | 0.19371159999999998 | 0.1945326 | 683782.5268136237 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.044761 | 0.288971 | 0.31476375 | 0.35487169 | 435805.00368153094 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.209664 | 0.2857605 | 0.31520885 | 0.32385197 | 441524.98313926475 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages3` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 5.621028 | 0.2790455 | 0.289374 | 0.29282419 | 458925.91544605524 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.935495 | 0.024898499999999997 | 0.0254935 | 0.026285 | 40149.77471961405 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 5.763405 | 0.0252315 | 0.029151849999999997 | 0.03727605999999998 | 38631.09927110842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 5.959213 | 0.0250375 | 0.02572635 | 0.029414989999999988 | 39614.095328943564 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 6.892395 | 0.024977 | 0.0253254 | 0.029332849999999987 | 39844.76479635341 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 5.655878 | 0.026023 | 0.02947989999999999 | 0.03355018 | 75730.7831244552 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 5.771008 | 0.0268705 | 0.030994099999999983 | 0.034632039999999996 | 73319.91090164427 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 5.991526 | 0.0272155 | 0.02773385 | 0.03140704 | 73108.39342843273 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 6.728008 | 0.027069 | 0.030616599999999987 | 0.03697284999999999 | 72577.26212439453 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.471342 | 0.0293505 | 0.03006945 | 0.036608579999999995 | 135654.28776072754 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 5.679266 | 0.0304885 | 0.031559449999999996 | 0.037194849999999995 | 130399.09294390949 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 5.916745 | 0.030462000000000003 | 0.03396664999999999 | 0.036548029999999995 | 131450.35087384906 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 6.613184 | 0.031293 | 0.03505174999999999 | 0.03920034999999999 | 126164.33914488334 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 5.506396 | 0.038043999999999994 | 0.04159899999999999 | 0.045531049999999997 | 210477.24135898842 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 5.79208 | 0.054165000000000005 | 0.06581695 | 0.06682407 | 146180.55783962677 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 5.838356 | 0.053479 | 0.06742039999999999 | 0.07493685 | 144291.24321695883 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.640757 | 0.055973999999999996 | 0.06509435 | 0.07137167999999998 | 140961.31387221435 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 5.485188 | 0.05178 | 0.0623387 | 0.06406844 | 305814.722914659 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.757998 | 0.1089785 | 0.12094975000000001 | 0.12401675999999999 | 147089.1606687997 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 5.882525 | 0.110637 | 0.12110555 | 0.12194224 | 144039.44952443574 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 6.451096 | 0.112166 | 0.13117715 | 0.13758176 | 142155.07448303973 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 5.474211 | 0.07816799999999999 | 0.0811885 | 0.08170837 | 406635.4777255251 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 5.736623 | 0.1696195 | 0.20495439999999998 | 0.20938788 | 183224.34538522002 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 5.865777 | 0.173326 | 0.1996041 | 0.20883327000000002 | 182055.24671273652 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.046687 | 0.17675649999999998 | 0.1928772 | 0.19913403 | 179311.48871122109 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 5.458758 | 0.13324999999999998 | 0.14390950000000002 | 0.14504648 | 470358.658766405 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 5.807811 | 0.2528705 | 0.27615565 | 0.30910827 | 250218.00243462116 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 5.94119 | 0.2517255 | 0.2707005 | 0.27542743000000003 | 253188.0529735869 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 6.645595 | 0.2408595 | 0.25885515 | 0.26239929 | 264625.1497923065 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 5.523463 | 0.245838 | 0.2542433 | 0.25645911 | 518127.3323825697 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 5.540576 | 0.3826555 | 0.39685745 | 0.39801344 | 333642.717444864 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 5.86395 | 0.3715625 | 0.39708555 | 0.39976258 | 342297.4803642922 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages4` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 6.584839 | 0.367243 | 0.37796165 | 0.38113216 | 348625.9941424296 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 6.606091 | 0.028776 | 0.030028699999999995 | 0.03610175999999999 | 34446.38798620216 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 2 | ok | 6.456003 | 0.0295265 | 0.03480754999999999 | 0.03773723999999999 | 33390.43097029254 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 4 | ok | 6.573907 | 0.0297635 | 0.03189925 | 0.03627246 | 33304.15889014557 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 8 | ok | 7.611034 | 0.029485499999999998 | 0.0332485 | 0.0350703 | 33567.97346787377 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 6.180386 | 0.0293265 | 0.030468099999999998 | 0.0353907 | 67557.61651324332 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 2 | ok | 6.297563 | 0.0311005 | 0.034944399999999994 | 0.03878196 | 63301.87652082758 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 4 | ok | 6.496309 | 0.031163999999999997 | 0.03548849999999999 | 0.03860515 | 63255.507973356776 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 8 | ok | 7.357848 | 0.0311805 | 0.03577999999999999 | 0.039764009999999995 | 63029.48708494296 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 6.133083 | 0.0349075 | 0.03732525 | 0.04122902999999999 | 113696.44186985168 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 2 | ok | 6.354061 | 0.035834000000000005 | 0.03831694999999999 | 0.04480750999999998 | 111745.3903629658 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 4 | ok | 6.577356 | 0.034875 | 0.040574 | 0.04216672 | 113303.7912581593 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 8 | ok | 7.341443 | 0.035937 | 0.03934109999999999 | 0.04260225 | 110276.86661516738 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 6.210715 | 0.04532 | 0.04619835 | 0.04693127 | 179430.93479031255 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 2 | ok | 6.211749 | 0.06585150000000001 | 0.07452945 | 0.22160757999999944 | 111147.8825355833 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 4 | ok | 6.62659 | 0.06760749999999999 | 0.07872955 | 0.08640581999999998 | 118622.7307842297 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 8 | ok | 6.963624 | 0.06726299999999999 | 0.0790229 | 0.08360701 | 116337.17071490937 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 6.212245 | 0.05946 | 0.06262645 | 0.06737855999999999 | 266130.59094630386 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 2 | ok | 6.494082 | 0.13093349999999998 | 0.148337 | 0.15313239 | 120455.07928955596 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 4 | ok | 6.473363 | 0.1274925 | 0.13870775 | 0.15067382999999995 | 124214.82643695205 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 8 | ok | 7.034278 | 0.12940249999999998 | 0.14978339999999998 | 0.15183932 | 120698.49425610952 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 6.24474 | 0.09747900000000001 | 0.10162439999999999 | 0.10269979 | 325956.4632175448 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 2 | ok | 6.424932 | 0.20406950000000001 | 0.217367 | 0.2193253 | 156537.77574496082 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 4 | ok | 6.444652 | 0.214391 | 0.23088655 | 0.23759653 | 147868.67634984694 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 8 | ok | 7.370197 | 0.216489 | 0.23441285 | 0.2448622 | 147136.069966144 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 6.197916 | 0.169324 | 0.1816198 | 0.2685108799999997 | 366231.11105821107 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 2 | ok | 6.295288 | 0.30969199999999997 | 0.33463384999999995 | 0.34143756999999997 | 204765.3635292283 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 4 | ok | 6.601904 | 0.3020255 | 0.32857 | 0.33649005 | 209474.10805106536 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 8 | ok | 7.249864 | 0.292596 | 0.31511554999999997 | 0.31695070999999997 | 217809.1368482555 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 6.183901 | 0.306014 | 0.31798174999999995 | 0.31828181 | 415417.14274661225 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 2 | ok | 6.321275 | 0.4672975 | 0.48928815 | 0.49069925 | 272970.9070166404 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 4 | ok | 6.444789 | 0.4561195 | 0.493629 | 0.49692069 | 277285.1273139498 | - |
| `separator3_grid_search_6ports_learned_dense_hd64_depth3_stages5` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 8 | ok | 7.122019 | 0.44536299999999995 | 0.4666204 | 0.47541221999999994 | 286311.6417221377 | - |
