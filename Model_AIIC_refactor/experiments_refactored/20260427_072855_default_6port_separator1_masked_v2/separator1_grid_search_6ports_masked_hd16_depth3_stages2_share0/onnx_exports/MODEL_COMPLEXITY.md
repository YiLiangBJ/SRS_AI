# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `21,024`
- Parameter memory (FP32): `0.080 MiB (84,096 bytes)`
- MACs / sample: `19,968`
- Multiplications / sample: `20,136`
- Additions / sample: `20,567`
- Other scalar ops / sample: `793`
- FLOPs / sample estimate: `41,496`
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
| `real_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `4,800` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Real branch affine layer maps width 24 to width 16. |
| `real_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `192` | `192` | Real branch ReLU keeps hidden width 16 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `3,264` | `3,072` | `3,072` | `3,072` | `0` | `6,144` | Real branch affine layer maps width 16 to width 16. |
| `real_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `192` | `192` | Real branch ReLU keeps hidden width 16 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `2,448` | `2,304` | `2,304` | `2,304` | `0` | `4,608` | Real branch affine layer maps width 16 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 2 stage executions)` | `4,800` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Imag branch affine layer maps width 24 to width 16. |
| `imag_relu_01` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `192` | `192` | Imag branch ReLU keeps hidden width 16 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 2 stage executions)` | `3,264` | `3,072` | `3,072` | `3,072` | `0` | `6,144` | Imag branch affine layer maps width 16 to width 16. |
| `imag_relu_02` | `per port, per stage (6 ports x 2 stage executions)` | `0` | `0` | `0` | `0` | `192` | `192` | Imag branch ReLU keeps hidden width 16 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 2 stage executions)` | `2,448` | `2,304` | `2,304` | `2,304` | `0` | `4,608` | Imag branch affine layer maps width 16 to width 12. |
| `residual_correction` | `per stage (2 stage executions)` | `0` | `0` | `0` | `576` | `0` | `576` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 21024,
  "trainable_parameters_string": "21,024",
  "parameter_memory_bytes_fp32": 84096,
  "parameter_memory_bytes_fp32_string": "0.080 MiB (84,096 bytes)",
  "macs_per_sample": 19968,
  "macs_per_sample_string": "19,968",
  "multiplications_per_sample": 20136,
  "multiplications_per_sample_string": "20,136",
  "additions_per_sample": 20567,
  "additions_per_sample_string": "20,567",
  "other_scalar_ops_per_sample_estimate": 793,
  "other_scalar_ops_per_sample_estimate_string": "793",
  "flops_per_sample_estimate": 41496,
  "flops_per_sample_estimate_string": "41,496",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
