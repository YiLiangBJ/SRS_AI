# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `10,512`
- Parameter memory (FP32): `0.040 MiB (42,048 bytes)`
- MACs / sample: `9,984`
- Multiplications / sample: `10,152`
- Additions / sample: `10,295`
- Other scalar ops / sample: `409`
- FLOPs / sample estimate: `20,856`
- Batch scaling: Multiply the per-sample counts by runtime batch size N for a first-order batch estimate.

## Counting Assumptions

- All counts are estimated for one forward pass of one sample; multiply by runtime batch size N for a first-order batch estimate.
- MACs and multiply/add counts are counted from affine layers using scalar CPU-style arithmetic.
- Bias accumulation is counted as additions inside the addition totals.
- Normalization, activations, and residual corrections are included as estimated scalar ops where relevant.
- Tensor reshapes, concatenations, indexing, and memory traffic are not counted as arithmetic FLOPs.
- Latency depends on implementation and hardware; these counts are for fast complexity screening only.

## Operator Breakdown

| Block | Repeat | Parameters | MACs | Multiplies | Adds | Other ops | FLOPs est. | Why |
|---|---|---|---|---|---|---|---|---|
| `input_normalize_restore` | `once per sample` | `0` | `0` | `168` | `23` | `25` | `216` | Per-sample RMS normalization and output rescaling are applied around the network when normalize_energy=true. |
| `real_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `2,400` | `2,304` | `2,304` | `2,304` | `0` | `4,608` | Real branch affine layer maps width 24 to width 16. |
| `real_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `96` | `96` | Real branch ReLU keeps hidden width 16 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `1,632` | `1,536` | `1,536` | `1,536` | `0` | `3,072` | Real branch affine layer maps width 16 to width 16. |
| `real_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `96` | `96` | Real branch ReLU keeps hidden width 16 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `1,224` | `1,152` | `1,152` | `1,152` | `0` | `2,304` | Real branch affine layer maps width 16 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `2,400` | `2,304` | `2,304` | `2,304` | `0` | `4,608` | Imag branch affine layer maps width 24 to width 16. |
| `imag_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `96` | `96` | Imag branch ReLU keeps hidden width 16 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `1,632` | `1,536` | `1,536` | `1,536` | `0` | `3,072` | Imag branch affine layer maps width 16 to width 16. |
| `imag_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `96` | `96` | Imag branch ReLU keeps hidden width 16 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `1,224` | `1,152` | `1,152` | `1,152` | `0` | `2,304` | Imag branch affine layer maps width 16 to width 12. |
| `residual_correction` | `per stage (1 stage executions)` | `0` | `0` | `0` | `288` | `0` | `288` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 10512,
  "trainable_parameters_string": "10,512",
  "parameter_memory_bytes_fp32": 42048,
  "parameter_memory_bytes_fp32_string": "0.040 MiB (42,048 bytes)",
  "macs_per_sample": 9984,
  "macs_per_sample_string": "9,984",
  "multiplications_per_sample": 10152,
  "multiplications_per_sample_string": "10,152",
  "additions_per_sample": 10295,
  "additions_per_sample_string": "10,295",
  "other_scalar_ops_per_sample_estimate": 409,
  "other_scalar_ops_per_sample_estimate_string": "409",
  "flops_per_sample_estimate": 20856,
  "flops_per_sample_estimate_string": "20,856",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
