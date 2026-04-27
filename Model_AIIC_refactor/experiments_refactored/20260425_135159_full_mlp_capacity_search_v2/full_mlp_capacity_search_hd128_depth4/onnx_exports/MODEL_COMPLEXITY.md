# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `38,288`
- Parameter memory (FP32): `0.146 MiB (153,152 bytes)`
- MACs / sample: `37,888`
- Multiplications / sample: `38,056`
- Additions / sample: `37,911`
- Other scalar ops / sample: `281`
- FLOPs / sample estimate: `76,248`
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
| `joint_linear_01` | `once per sample` | `3,200` | `3,072` | `3,072` | `3,072` | `0` | `6,144` | Joint affine layer maps width 24 to width 128. |
| `joint_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_02` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Joint affine layer maps width 128 to width 128. |
| `joint_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_03` | `once per sample` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Joint affine layer maps width 128 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 38288,
  "trainable_parameters_string": "38,288",
  "parameter_memory_bytes_fp32": 153152,
  "parameter_memory_bytes_fp32_string": "0.146 MiB (153,152 bytes)",
  "macs_per_sample": 37888,
  "macs_per_sample_string": "37,888",
  "multiplications_per_sample": 38056,
  "multiplications_per_sample_string": "38,056",
  "additions_per_sample": 37911,
  "additions_per_sample_string": "37,911",
  "other_scalar_ops_per_sample_estimate": 281,
  "other_scalar_ops_per_sample_estimate_string": "281",
  "flops_per_sample_estimate": 76248,
  "flops_per_sample_estimate_string": "76,248",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
