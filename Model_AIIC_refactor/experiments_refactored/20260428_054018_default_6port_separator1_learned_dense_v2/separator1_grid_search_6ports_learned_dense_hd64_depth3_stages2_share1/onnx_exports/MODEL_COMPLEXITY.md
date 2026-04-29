# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `78,624`
- Parameter memory (FP32): `0.300 MiB (314,496 bytes)`
- MACs / sample: `153,600`
- Multiplications / sample: `153,768`
- Additions / sample: `154,199`
- Other scalar ops / sample: `3,097`
- FLOPs / sample estimate: `311,064`
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
| `real_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `19,200` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Real branch affine layer maps width 24 to width 64. |
| `real_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Real branch ReLU keeps hidden width 64 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `49,920` | `49,152` | `49,152` | `49,152` | `0` | `98,304` | Real branch affine layer maps width 64 to width 64. |
| `real_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Real branch ReLU keeps hidden width 64 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `9,360` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Real branch affine layer maps width 64 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `19,200` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Imag branch affine layer maps width 24 to width 64. |
| `imag_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Imag branch ReLU keeps hidden width 64 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `49,920` | `49,152` | `49,152` | `49,152` | `0` | `98,304` | Imag branch affine layer maps width 64 to width 64. |
| `imag_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Imag branch ReLU keeps hidden width 64 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `9,360` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Imag branch affine layer maps width 64 to width 12. |
| `residual_correction` | `per stage (2 stage executions)` | `0` | `0` | `0` | `576` | `0` | `576` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 78624,
  "trainable_parameters_string": "78,624",
  "parameter_memory_bytes_fp32": 314496,
  "parameter_memory_bytes_fp32_string": "0.300 MiB (314,496 bytes)",
  "macs_per_sample": 153600,
  "macs_per_sample_string": "153,600",
  "multiplications_per_sample": 153768,
  "multiplications_per_sample_string": "153,768",
  "additions_per_sample": 154199,
  "additions_per_sample_string": "154,199",
  "other_scalar_ops_per_sample_estimate": 3097,
  "other_scalar_ops_per_sample_estimate_string": "3,097",
  "flops_per_sample_estimate": 311064,
  "flops_per_sample_estimate_string": "311,064",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
