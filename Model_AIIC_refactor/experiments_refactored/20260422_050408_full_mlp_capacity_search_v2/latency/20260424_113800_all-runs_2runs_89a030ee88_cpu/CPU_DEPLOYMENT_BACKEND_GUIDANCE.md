# CPU Deployment Backend Guidance

## Scope

This note summarizes a focused CPU deployment-style latency comparison across three runtime paths:

- `pytorch` with `jit`
- `onnxruntime`
- `openvino`

Benchmark scope:

- device: `cpu`
- precision: `fp32`
- threads: `1`
- batch sizes: `1, 2, 4, 8, 16, 32, 64, 128`
- representative small model: `full_mlp_capacity_search_hd32_depth2`
- representative large model: `full_mlp_capacity_search_hd512_depth5`

The purpose of this comparison is not to prove one backend is universally best. The purpose is to identify which backend is most relevant for deployment estimates under different batch regimes.

## Executive Takeaway

The backend choice is strongly batch-dependent and model-size-dependent.

- For low-batch CPU deployment, `onnxruntime` is the strongest default choice.
- For small models, `onnxruntime` remains best across the full tested batch range.
- For larger models, `openvino` starts to become competitive around `batch=8` and clearly wins from about `batch=16` onward.
- `pytorch + jit` is useful as a development reference, but it is not the best latency path in these focused deployment-style comparisons.

Practical guidance:

- If expected production traffic is mostly `batch=1` or other low-batch inference, prioritize `onnxruntime`.
- If expected production traffic includes larger batches on larger models, benchmark `openvino` seriously before finalizing the deployment stack.

## Small Model Result

Model:

- `full_mlp_capacity_search_hd32_depth2`

P50 latency in milliseconds:

| Batch | PyTorch JIT | ONNX Runtime | OpenVINO | Fastest |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.038277 | 0.013817 | 0.087931 | ONNX Runtime |
| 2 | 0.041257 | 0.014616 | 0.142492 | ONNX Runtime |
| 4 | 0.042798 | 0.014507 | 0.095629 | ONNX Runtime |
| 8 | 0.043993 | 0.015417 | 0.099812 | ONNX Runtime |
| 16 | 0.049268 | 0.016749 | 0.113089 | ONNX Runtime |
| 32 | 0.047215 | 0.019346 | 0.138435 | ONNX Runtime |
| 64 | 0.053758 | 0.023395 | 0.111785 | ONNX Runtime |
| 128 | 0.059075 | 0.033757 | 0.115284 | ONNX Runtime |

Interpretation:

- `onnxruntime` wins at every tested batch size.
- `openvino` never catches up on this small model.
- For small-model deployment, there is no evidence here that `openvino` is worth preferring over `onnxruntime`.

## Large Model Result

Model:

- `full_mlp_capacity_search_hd512_depth5`

P50 latency in milliseconds:

| Batch | PyTorch JIT | ONNX Runtime | OpenVINO | Fastest |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.156535 | 0.108534 | 0.135717 | ONNX Runtime |
| 2 | 0.170882 | 0.107011 | 0.133317 | ONNX Runtime |
| 4 | 0.169669 | 0.109319 | 0.163391 | ONNX Runtime |
| 8 | 0.179724 | 0.143913 | 0.143604 | OpenVINO |
| 16 | 0.231667 | 0.165528 | 0.155414 | OpenVINO |
| 32 | 0.340651 | 0.292865 | 0.132299 | OpenVINO |
| 64 | 0.550501 | 0.506474 | 0.193144 | OpenVINO |
| 128 | 0.991357 | 0.986199 | 0.253385 | OpenVINO |

Interpretation:

- `onnxruntime` is best at low batch sizes (`1`, `2`, `4`).
- The crossover happens around `batch=8`.
- From about `batch=16` onward, `openvino` clearly becomes the best backend.
- At very large batch (`128`), `openvino` is dramatically faster than both `onnxruntime` and `pytorch + jit`.

## What This Means For Deployment Planning

This focused result suggests that backend selection should not be made once for all CPU inference scenarios without considering workload shape.

### If deployment is low-batch latency sensitive

Use `onnxruntime` as the primary reference path.

This is especially justified when:

- requests are mostly `batch=1`
- the model is small or medium-sized
- the system is optimized for tail latency rather than throughput aggregation

### If deployment is high-batch CPU throughput oriented

Use `openvino` as a serious candidate for larger models.

This is especially justified when:

- requests are batched before inference
- the deployed model is in the larger part of the tested family
- CPU throughput matters more than single-request latency

### If only one backend can be tested first

Use this priority order:

- low-batch first pass: `onnxruntime`
- large-batch, large-model follow-up: `openvino`

## Backend Role Summary

### PyTorch JIT

- Good for local development and fast validation.
- Useful as a reference baseline.
- Not the best deployment-latency path in these tests.

### ONNX Runtime

- Best general default for low-batch CPU deployment estimates.
- Strong across both small and large models at low batch sizes.
- Also remains competitive at high batch sizes, even when it is not the best.

### OpenVINO

- Not a universal winner.
- Weak on the small model in this comparison.
- Strong on the larger model once batch size becomes sufficiently large.
- Best interpreted as a targeted optimization path for larger-batch CPU deployment, not as the default answer for every workload.

## Recommended Next Step

If future testing is moved to a different machine, the most efficient follow-up is:

- keep `onnxruntime` as the default low-batch deployment reference
- test `openvino` only on larger models and the batch range around the observed crossover region, especially `8, 16, 32, 64, 128`

This keeps the validation focused on the regime where `openvino` has already shown evidence of winning.