# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `6,608`
- Parameter memory (FP32): `0.025 MiB (26,432 bytes)`
- MACs / sample: `6,400`
- Multiplications / sample: `6,568`
- Additions / sample: `6,423`
- Other scalar ops / sample: `89`
- FLOPs / sample estimate: `13,080`
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
| `joint_linear_01` | `once per sample` | `800` | `768` | `768` | `768` | `0` | `1,536` | Joint affine layer maps width 24 to width 32. |
| `joint_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Joint affine layer maps width 32 to width 32. |
| `joint_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_03` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Joint affine layer maps width 32 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 6608,
  "trainable_parameters_string": "6,608",
  "parameter_memory_bytes_fp32": 26432,
  "parameter_memory_bytes_fp32_string": "0.025 MiB (26,432 bytes)",
  "macs_per_sample": 6400,
  "macs_per_sample_string": "6,400",
  "multiplications_per_sample": 6568,
  "multiplications_per_sample_string": "6,568",
  "additions_per_sample": 6423,
  "additions_per_sample_string": "6,423",
  "other_scalar_ops_per_sample_estimate": 89,
  "other_scalar_ops_per_sample_estimate_string": "89",
  "flops_per_sample_estimate": 13080,
  "flops_per_sample_estimate_string": "13,080",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
