# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `19,280`
- Parameter memory (FP32): `0.074 MiB (77,120 bytes)`
- MACs / sample: `18,944`
- Multiplications / sample: `19,112`
- Additions / sample: `18,967`
- Other scalar ops / sample: `217`
- FLOPs / sample estimate: `38,296`
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
| `joint_linear_01` | `once per sample` | `1,600` | `1,536` | `1,536` | `1,536` | `0` | `3,072` | Joint affine layer maps width 24 to width 64. |
| `joint_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_02` | `once per sample` | `4,160` | `4,096` | `4,096` | `4,096` | `0` | `8,192` | Joint affine layer maps width 64 to width 64. |
| `joint_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_03` | `once per sample` | `4,160` | `4,096` | `4,096` | `4,096` | `0` | `8,192` | Joint affine layer maps width 64 to width 64. |
| `joint_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_04` | `once per sample` | `9,360` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Joint affine layer maps width 64 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 19280,
  "trainable_parameters_string": "19,280",
  "parameter_memory_bytes_fp32": 77120,
  "parameter_memory_bytes_fp32_string": "0.074 MiB (77,120 bytes)",
  "macs_per_sample": 18944,
  "macs_per_sample_string": "18,944",
  "multiplications_per_sample": 19112,
  "multiplications_per_sample_string": "19,112",
  "additions_per_sample": 18967,
  "additions_per_sample_string": "18,967",
  "other_scalar_ops_per_sample_estimate": 217,
  "other_scalar_ops_per_sample_estimate_string": "217",
  "flops_per_sample_estimate": 38296,
  "flops_per_sample_estimate_string": "38,296",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
