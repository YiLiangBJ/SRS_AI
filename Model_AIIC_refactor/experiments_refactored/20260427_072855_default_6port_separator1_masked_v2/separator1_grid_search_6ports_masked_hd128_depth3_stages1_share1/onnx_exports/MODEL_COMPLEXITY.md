# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `255,120`
- Parameter memory (FP32): `0.973 MiB (1,020,480 bytes)`
- MACs / sample: `251,904`
- Multiplications / sample: `252,072`
- Additions / sample: `252,215`
- Other scalar ops / sample: `3,097`
- FLOPs / sample estimate: `507,384`
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
| `real_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `19,200` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Real branch affine layer maps width 24 to width 128. |
| `real_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Real branch ReLU keeps hidden width 128 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `99,072` | `98,304` | `98,304` | `98,304` | `0` | `196,608` | Real branch affine layer maps width 128 to width 128. |
| `real_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Real branch ReLU keeps hidden width 128 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `9,288` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Real branch affine layer maps width 128 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `19,200` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Imag branch affine layer maps width 24 to width 128. |
| `imag_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Imag branch ReLU keeps hidden width 128 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `99,072` | `98,304` | `98,304` | `98,304` | `0` | `196,608` | Imag branch affine layer maps width 128 to width 128. |
| `imag_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `768` | `768` | Imag branch ReLU keeps hidden width 128 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `9,288` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Imag branch affine layer maps width 128 to width 12. |
| `residual_correction` | `per stage (1 stage executions)` | `0` | `0` | `0` | `288` | `0` | `288` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 255120,
  "trainable_parameters_string": "255,120",
  "parameter_memory_bytes_fp32": 1020480,
  "parameter_memory_bytes_fp32_string": "0.973 MiB (1,020,480 bytes)",
  "macs_per_sample": 251904,
  "macs_per_sample_string": "251,904",
  "multiplications_per_sample": 252072,
  "multiplications_per_sample_string": "252,072",
  "additions_per_sample": 252215,
  "additions_per_sample_string": "252,215",
  "other_scalar_ops_per_sample_estimate": 3097,
  "other_scalar_ops_per_sample_estimate_string": "3,097",
  "flops_per_sample_estimate": 507384,
  "flops_per_sample_estimate_string": "507,384",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
