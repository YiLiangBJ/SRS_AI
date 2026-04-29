# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `5,696`
- Parameter memory (FP32): `0.022 MiB (22,784 bytes)`
- MACs / sample: `5,376`
- Multiplications / sample: `5,544`
- Additions / sample: `5,399`
- Other scalar ops / sample: `57`
- FLOPs / sample estimate: `11,000`
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
| `joint_linear_02` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Joint affine layer maps width 32 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 5696,
  "trainable_parameters_string": "5,696",
  "parameter_memory_bytes_fp32": 22784,
  "parameter_memory_bytes_fp32_string": "0.022 MiB (22,784 bytes)",
  "macs_per_sample": 5376,
  "macs_per_sample_string": "5,376",
  "multiplications_per_sample": 5544,
  "multiplications_per_sample_string": "5,544",
  "additions_per_sample": 5399,
  "additions_per_sample_string": "5,399",
  "other_scalar_ops_per_sample_estimate": 57,
  "other_scalar_ops_per_sample_estimate_string": "57",
  "flops_per_sample_estimate": 11000,
  "flops_per_sample_estimate_string": "11,000",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
