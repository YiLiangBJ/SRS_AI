# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `43,552`
- Parameter memory (FP32): `0.166 MiB (174,208 bytes)`
- MACs / sample: `43,008`
- Multiplications / sample: `43,176`
- Additions / sample: `43,031`
- Other scalar ops / sample: `281`
- FLOPs / sample estimate: `86,488`
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
| `joint_linear_02` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Joint affine layer maps width 256 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 43552,
  "trainable_parameters_string": "43,552",
  "parameter_memory_bytes_fp32": 174208,
  "parameter_memory_bytes_fp32_string": "0.166 MiB (174,208 bytes)",
  "macs_per_sample": 43008,
  "macs_per_sample_string": "43,008",
  "multiplications_per_sample": 43176,
  "multiplications_per_sample_string": "43,176",
  "additions_per_sample": 43031,
  "additions_per_sample_string": "43,031",
  "other_scalar_ops_per_sample_estimate": 281,
  "other_scalar_ops_per_sample_estimate_string": "281",
  "flops_per_sample_estimate": 86488,
  "flops_per_sample_estimate_string": "86,488",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
