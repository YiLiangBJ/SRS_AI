# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `510,528`
- Parameter memory (FP32): `1.948 MiB (2,042,112 bytes)`
- MACs / sample: `503,808`
- Multiplications / sample: `503,976`
- Additions / sample: `504,407`
- Other scalar ops / sample: `6,169`
- FLOPs / sample estimate: `1,014,552`
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
| `real_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `38,400` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Real branch affine layer maps width 24 to width 128. |
| `real_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `1,536` | `1,536` | Real branch ReLU keeps hidden width 128 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `198,144` | `196,608` | `196,608` | `196,608` | `0` | `393,216` | Real branch affine layer maps width 128 to width 128. |
| `real_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `1,536` | `1,536` | Real branch ReLU keeps hidden width 128 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Real branch affine layer maps width 128 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `38,400` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Imag branch affine layer maps width 24 to width 128. |
| `imag_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `1,536` | `1,536` | Imag branch ReLU keeps hidden width 128 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `198,144` | `196,608` | `196,608` | `196,608` | `0` | `393,216` | Imag branch affine layer maps width 128 to width 128. |
| `imag_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `1,536` | `1,536` | Imag branch ReLU keeps hidden width 128 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Imag branch affine layer maps width 128 to width 12. |
| `residual_correction` | `per stage (2 stage executions)` | `0` | `0` | `0` | `576` | `0` | `576` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 510528,
  "trainable_parameters_string": "510,528",
  "parameter_memory_bytes_fp32": 2042112,
  "parameter_memory_bytes_fp32_string": "1.948 MiB (2,042,112 bytes)",
  "macs_per_sample": 503808,
  "macs_per_sample_string": "503,808",
  "multiplications_per_sample": 503976,
  "multiplications_per_sample_string": "503,976",
  "additions_per_sample": 504407,
  "additions_per_sample_string": "504,407",
  "other_scalar_ops_per_sample_estimate": 6169,
  "other_scalar_ops_per_sample_estimate_string": "6,169",
  "flops_per_sample_estimate": 1014552,
  "flops_per_sample_estimate_string": "1,014,552",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
