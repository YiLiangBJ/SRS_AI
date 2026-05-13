# Latency Report

- Device: `cpu`
- Runtime backends: `['onnxruntime']`
- Execution modes: `['eager', 'jit', 'compile']`
- Precision profiles: `['fp32']`
- Batch sizes: `[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`
- Thread counts: `[1]`

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

### separator3_default_hidden_dim64_mlp_depth3_num_stages2::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`256`, precision=`fp32`, throughput=`1050564.662` samples/s, p50=`0.240` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.019` ms, throughput=`54947.712` samples/s

## Run References

### separator3_default_hidden_dim64_mlp_depth3_num_stages2

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260512_072121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260512_072121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260512_072121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260512_072121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,208`
- MACs / sample: `37,376`
- FLOPs / sample estimate: `75,800`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 4.877448 | 0.018647 | 0.01910725 | 0.02472944999999998 | 54947.711757491576 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.677914 | 0.0175625 | 0.018032199999999998 | 0.019766289999999992 | 113872.9087240313 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 4.721142 | 0.0197865 | 0.02334015 | 0.028384589999999998 | 196589.371002478 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.657413 | 0.024272000000000002 | 0.027572599999999996 | 0.035901029999999994 | 323459.7252209634 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.671585 | 0.032585 | 0.03555495 | 0.043655199999999984 | 479309.6981726917 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.623534 | 0.042965 | 0.04901875 | 0.05257323 | 726662.9909630374 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.952798 | 0.0708985 | 0.081085 | 0.08289658999999999 | 883197.9050545692 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.806198 | 0.1280685 | 0.14002859999999998 | 0.14247757 | 991594.532037877 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 256 | 1 | ok | 4.845937 | 0.239709 | 0.25714135 | 0.27237412 | 1050564.662090801 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 512 | 1 | ok | 4.768895 | 0.514575 | 0.5467645 | 0.6531830599999997 | 984003.7127690088 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 1024 | 1 | ok | 4.662921 | 1.1057645 | 1.1542096 | 1.16950632 | 924059.9751741904 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 2048 | 1 | ok | 4.901144 | 2.1150545 | 2.18296695 | 2.20504109 | 966939.1329284935 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 4096 | 1 | ok | 4.931552 | 4.3344249999999995 | 4.370766000000001 | 4.40456073 | 945621.2991889183 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2` | `onnxruntime` | `onnxruntime` | `fp32` | 8192 | 1 | ok | 5.153073 | 8.834914000000001 | 8.8907917 | 8.94648371 | 926433.2476482098 | - |
