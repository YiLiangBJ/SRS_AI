# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `59,200`
- Parameter memory (FP32): `0.226 MiB (236,800 bytes)`
- MACs / sample: `58,368`
- Multiplications / sample: `58,536`
- Additions / sample: `58,967`
- Other scalar ops / sample: `281`
- FLOPs / sample estimate: `117,784`
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
| `stage_01_hidden_linear_01` | `once per sample` | `3,200` | `3,072` | `3,072` | `3,072` | `0` | `6,144` | Stage 1 first affine layer maps width 24 to hidden width 128. |
| `stage_01_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after the first hidden affine layer. |
| `stage_01_joint_output` | `once per sample` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Stage 1 output affine layer maps hidden width 128 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_02_hidden_linear_01` | `once per sample` | `18,560` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Stage 2 first affine layer maps width 144 to hidden width 128. |
| `stage_02_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 2 applies ReLU after the first hidden affine layer. |
| `stage_02_joint_output` | `once per sample` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Stage 2 output affine layer maps hidden width 128 to expanded width 144. |
| `stage_02_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 2 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 59200,
  "trainable_parameters_string": "59,200",
  "parameter_memory_bytes_fp32": 236800,
  "parameter_memory_bytes_fp32_string": "0.226 MiB (236,800 bytes)",
  "macs_per_sample": 58368,
  "macs_per_sample_string": "58,368",
  "multiplications_per_sample": 58536,
  "multiplications_per_sample_string": "58,536",
  "additions_per_sample": 58967,
  "additions_per_sample_string": "58,967",
  "other_scalar_ops_per_sample_estimate": 281,
  "other_scalar_ops_per_sample_estimate_string": "281",
  "flops_per_sample_estimate": 117784,
  "flops_per_sample_estimate_string": "117,784",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
