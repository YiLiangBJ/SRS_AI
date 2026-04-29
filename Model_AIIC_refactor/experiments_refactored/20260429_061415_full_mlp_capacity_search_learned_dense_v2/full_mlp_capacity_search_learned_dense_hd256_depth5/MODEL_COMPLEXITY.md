# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `175,136`
- Parameter memory (FP32): `0.668 MiB (700,544 bytes)`
- MACs / sample: `174,080`
- Multiplications / sample: `174,248`
- Additions / sample: `174,103`
- Other scalar ops / sample: `793`
- FLOPs / sample estimate: `349,144`
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
| `joint_linear_01` | `once per sample` | `6,400` | `6,144` | `6,144` | `6,144` | `0` | `12,288` | Joint affine layer maps width 24 to width 256. |
| `joint_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_02` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Joint affine layer maps width 256 to width 256. |
| `joint_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_03` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Joint affine layer maps width 256 to width 256. |
| `joint_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_04` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Joint affine layer maps width 256 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 175136,
  "trainable_parameters_string": "175,136",
  "parameter_memory_bytes_fp32": 700544,
  "parameter_memory_bytes_fp32_string": "0.668 MiB (700,544 bytes)",
  "macs_per_sample": 174080,
  "macs_per_sample_string": "174,080",
  "multiplications_per_sample": 174248,
  "multiplications_per_sample_string": "174,248",
  "additions_per_sample": 174103,
  "additions_per_sample_string": "174,103",
  "other_scalar_ops_per_sample_estimate": 793,
  "other_scalar_ops_per_sample_estimate_string": "793",
  "flops_per_sample_estimate": 349144,
  "flops_per_sample_estimate_string": "349,144",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
