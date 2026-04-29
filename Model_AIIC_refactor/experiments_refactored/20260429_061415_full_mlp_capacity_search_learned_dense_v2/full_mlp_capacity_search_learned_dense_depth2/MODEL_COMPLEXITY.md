# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `3,744`
- Parameter memory (FP32): `0.014 MiB (14,976 bytes)`
- MACs / sample: `3,456`
- Multiplications / sample: `3,624`
- Additions / sample: `3,479`
- Other scalar ops / sample: `25`
- FLOPs / sample estimate: `7,128`
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
| `joint_linear_01` | `once per sample` | `3,600` | `3,456` | `3,456` | `3,456` | `0` | `6,912` | Joint affine layer maps width 24 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 3744,
  "trainable_parameters_string": "3,744",
  "parameter_memory_bytes_fp32": 14976,
  "parameter_memory_bytes_fp32_string": "0.014 MiB (14,976 bytes)",
  "macs_per_sample": 3456,
  "macs_per_sample_string": "3,456",
  "multiplications_per_sample": 3624,
  "multiplications_per_sample_string": "3,624",
  "additions_per_sample": 3479,
  "additions_per_sample_string": "3,479",
  "other_scalar_ops_per_sample_estimate": 25,
  "other_scalar_ops_per_sample_estimate_string": "25",
  "flops_per_sample_estimate": 7128,
  "flops_per_sample_estimate_string": "7,128",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
