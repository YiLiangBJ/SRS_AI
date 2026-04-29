# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `54,944`
- Parameter memory (FP32): `0.210 MiB (219,776 bytes)`
- MACs / sample: `54,272`
- Multiplications / sample: `54,440`
- Additions / sample: `54,295`
- Other scalar ops / sample: `409`
- FLOPs / sample estimate: `109,144`
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
| `joint_linear_03` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Joint affine layer maps width 128 to width 128. |
| `joint_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_04` | `once per sample` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Joint affine layer maps width 128 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 54944,
  "trainable_parameters_string": "54,944",
  "parameter_memory_bytes_fp32": 219776,
  "parameter_memory_bytes_fp32_string": "0.210 MiB (219,776 bytes)",
  "macs_per_sample": 54272,
  "macs_per_sample_string": "54,272",
  "multiplications_per_sample": 54440,
  "multiplications_per_sample_string": "54,440",
  "additions_per_sample": 54295,
  "additions_per_sample_string": "54,295",
  "other_scalar_ops_per_sample_estimate": 409,
  "other_scalar_ops_per_sample_estimate_string": "409",
  "flops_per_sample_estimate": 109144,
  "flops_per_sample_estimate_string": "109,144",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
