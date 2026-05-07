# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `19,424`
- Parameter memory (FP32): `0.074 MiB (77,696 bytes)`
- MACs / sample: `18,944`
- Multiplications / sample: `19,112`
- Additions / sample: `19,255`
- Other scalar ops / sample: `217`
- FLOPs / sample estimate: `38,584`
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
| `stage_01_hidden_linear_01` | `once per sample` | `1,600` | `1,536` | `1,536` | `1,536` | `0` | `3,072` | Stage 1 first affine layer maps width 24 to hidden width 64. |
| `stage_01_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | Stage 1 applies ReLU after the first hidden affine layer. |
| `stage_01_hidden_linear_02` | `once per sample` | `4,160` | `4,096` | `4,096` | `4,096` | `0` | `8,192` | Stage 1 additional hidden affine layer keeps width 64. |
| `stage_01_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | Stage 1 applies ReLU after hidden affine layer 2. |
| `stage_01_hidden_linear_03` | `once per sample` | `4,160` | `4,096` | `4,096` | `4,096` | `0` | `8,192` | Stage 1 additional hidden affine layer keeps width 64. |
| `stage_01_hidden_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `64` | `64` | Stage 1 applies ReLU after hidden affine layer 3. |
| `stage_01_joint_output` | `once per sample` | `9,360` | `9,216` | `9,216` | `9,216` | `0` | `18,432` | Stage 1 output affine layer maps hidden width 64 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 19424,
  "trainable_parameters_string": "19,424",
  "parameter_memory_bytes_fp32": 77696,
  "parameter_memory_bytes_fp32_string": "0.074 MiB (77,696 bytes)",
  "macs_per_sample": 18944,
  "macs_per_sample_string": "18,944",
  "multiplications_per_sample": 19112,
  "multiplications_per_sample_string": "19,112",
  "additions_per_sample": 19255,
  "additions_per_sample_string": "19,255",
  "other_scalar_ops_per_sample_estimate": 217,
  "other_scalar_ops_per_sample_estimate_string": "217",
  "flops_per_sample_estimate": 38584,
  "flops_per_sample_estimate_string": "38,584",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
