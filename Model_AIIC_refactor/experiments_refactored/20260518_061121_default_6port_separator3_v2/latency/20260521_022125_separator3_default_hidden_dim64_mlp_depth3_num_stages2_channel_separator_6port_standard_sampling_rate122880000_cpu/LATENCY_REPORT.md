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

### separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000::onnxruntime::onnxruntime

- Runtime backend: `onnxruntime`
- Execution mode: `onnxruntime`
- Best throughput config: threads=`1`, batch=`256`, precision=`fp32`, throughput=`1063195.599` samples/s, p50=`0.239` ms
- Lowest batch-1 p50 latency: threads=`1`, precision=`fp32`, p50=`0.015` ms, throughput=`62900.915` samples/s

## Run References

### separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000

- Run dir: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260518_061121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000`
- Model flow: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260518_061121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000/MODEL_FLOW.md`
- Model complexity JSON: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260518_061121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000/model_complexity.json`
- Model complexity Markdown: `/home/liangyi/SRS_AI/Model_AIIC_refactor/experiments_refactored/20260518_061121_default_6port_separator3_v2/separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000/MODEL_COMPLEXITY.md`
- Trainable parameters: `38,208`
- MACs / sample: `37,376`
- FLOPs / sample estimate: `75,800`

## Results

| Run | Backend | Mode | Precision | Batch | Threads | Status | Prep (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Throughput | Skip reason |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 1 | 1 | ok | 5.422087 | 0.015445 | 0.016463449999999998 | 0.02545409999999998 | 62900.914705101655 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 2 | 1 | ok | 4.994672 | 0.0174985 | 0.01820475 | 0.022713979999999995 | 113035.86062678385 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 4 | 1 | ok | 5.017412 | 0.021998 | 0.024046949999999994 | 0.03875130999999997 | 186574.82210090716 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 8 | 1 | ok | 4.796565 | 0.0241695 | 0.02891259999999999 | 0.032217989999999995 | 323729.62425511837 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 16 | 1 | ok | 4.801069 | 0.0316595 | 0.0348897 | 0.04014834 | 511122.9947207384 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 32 | 1 | ok | 4.837415 | 0.045254 | 0.05177525 | 0.06329839999999996 | 695126.8997600944 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 64 | 1 | ok | 4.661452 | 0.074494 | 0.08172805 | 0.08569099999999999 | 855523.4734253021 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 128 | 1 | ok | 4.665482 | 0.1316515 | 0.14391874999999998 | 0.14854598 | 966340.4009557709 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 256 | 1 | ok | 4.76016 | 0.23853600000000003 | 0.2523848 | 0.25839596 | 1063195.5988353689 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 512 | 1 | ok | 4.70581 | 0.5017205 | 0.5172801 | 0.5236111999999999 | 1017099.7511840272 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 1024 | 1 | ok | 4.697018 | 1.1054275 | 1.1394476 | 1.1542725 | 926889.6591808621 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 2048 | 1 | ok | 4.769876 | 2.130439 | 2.1726517999999997 | 2.23248116 | 959780.852038328 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 4096 | 1 | ok | 4.911768 | 4.352748 | 4.3920742 | 4.40427504 | 940593.2293022219 | - |
| `separator3_default_hidden_dim64_mlp_depth3_num_stages2_channel_separator_6port_standard_sampling_rate122880000` | `onnxruntime` | `onnxruntime` | `fp32` | 8192 | 1 | ok | 5.134302 | 8.809207 | 8.88145925 | 8.89952425 | 929183.7086864614 | - |
