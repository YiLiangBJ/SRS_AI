# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `24,768`
- Parameter memory (FP32): `0.094 MiB (99,072 bytes)`
- MACs / sample: `23,808`
- Multiplications / sample: `23,976`
- Additions / sample: `24,695`
- Other scalar ops / sample: `121`
- FLOPs / sample estimate: `48,792`
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
| `stage_01_hidden_linear_01` | `once per sample` | `800` | `768` | `768` | `768` | `0` | `1,536` | Stage 1 first affine layer maps width 24 to hidden width 32. |
| `stage_01_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after the first hidden affine layer. |
| `stage_01_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 1 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_02_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 first affine layer maps width 144 to hidden width 32. |
| `stage_02_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 2 applies ReLU after the first hidden affine layer. |
| `stage_02_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_02_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 2 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_03_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 3 first affine layer maps width 144 to hidden width 32. |
| `stage_03_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 3 applies ReLU after the first hidden affine layer. |
| `stage_03_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 3 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_03_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 3 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 24768,
  "trainable_parameters_string": "24,768",
  "parameter_memory_bytes_fp32": 99072,
  "parameter_memory_bytes_fp32_string": "0.094 MiB (99,072 bytes)",
  "macs_per_sample": 23808,
  "macs_per_sample_string": "23,808",
  "multiplications_per_sample": 23976,
  "multiplications_per_sample_string": "23,976",
  "additions_per_sample": 24695,
  "additions_per_sample_string": "24,695",
  "other_scalar_ops_per_sample_estimate": 121,
  "other_scalar_ops_per_sample_estimate_string": "121",
  "flops_per_sample_estimate": 48792,
  "flops_per_sample_estimate_string": "48,792",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
