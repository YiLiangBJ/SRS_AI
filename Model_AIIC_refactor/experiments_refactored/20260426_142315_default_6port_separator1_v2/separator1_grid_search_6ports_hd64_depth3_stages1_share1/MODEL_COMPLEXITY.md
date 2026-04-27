# Model Complexity

- Model type: `separator1`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `78,480`
- Parameter memory (FP32): `0.299 MiB (313,920 bytes)`
- MACs / sample: `76,800`
- Multiplications / sample: `76,968`
- Additions / sample: `77,111`
- Other scalar ops / sample: `1,561`
- FLOPs / sample estimate: `155,640`
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
| `real_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `9,600` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Real branch affine layer maps width 24 to width 64. |
| `real_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `384` | `384` | Real branch ReLU keeps hidden width 64 unchanged. |
| `real_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `24,960` | `24,576` | `24,576` | `24,576` | `0` | `49,152` | Real branch affine layer maps width 64 to width 64. |
| `real_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `384` | `384` | Real branch ReLU keeps hidden width 64 unchanged. |
| `real_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `4,680` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Real branch affine layer maps width 64 to width 12. |
| `imag_linear_01` | `per port, per stage (6 ports x 1 stage executions)` | `9,600` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Imag branch affine layer maps width 24 to width 64. |
| `imag_relu_01` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `384` | `384` | Imag branch ReLU keeps hidden width 64 unchanged. |
| `imag_linear_02` | `per port, per stage (6 ports x 1 stage executions)` | `24,960` | `24,576` | `24,576` | `24,576` | `0` | `49,152` | Imag branch affine layer maps width 64 to width 64. |
| `imag_relu_02` | `per port, per stage (6 ports x 1 stage executions)` | `0` | `0` | `0` | `0` | `384` | `384` | Imag branch ReLU keeps hidden width 64 unchanged. |
| `imag_linear_03` | `per port, per stage (6 ports x 1 stage executions)` | `4,680` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Imag branch affine layer maps width 64 to width 12. |
| `residual_correction` | `per stage (1 stage executions)` | `0` | `0` | `0` | `288` | `0` | `288` | Stage output is summed across ports, subtracted from the mixed input, then broadcast back to every port estimate. |

## Raw Summary

```json
{
  "trainable_parameters": 78480,
  "trainable_parameters_string": "78,480",
  "parameter_memory_bytes_fp32": 313920,
  "parameter_memory_bytes_fp32_string": "0.299 MiB (313,920 bytes)",
  "macs_per_sample": 76800,
  "macs_per_sample_string": "76,800",
  "multiplications_per_sample": 76968,
  "multiplications_per_sample_string": "76,968",
  "additions_per_sample": 77111,
  "additions_per_sample_string": "77,111",
  "other_scalar_ops_per_sample_estimate": 1561,
  "other_scalar_ops_per_sample_estimate_string": "1,561",
  "flops_per_sample_estimate": 155640,
  "flops_per_sample_estimate_string": "155,640",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
