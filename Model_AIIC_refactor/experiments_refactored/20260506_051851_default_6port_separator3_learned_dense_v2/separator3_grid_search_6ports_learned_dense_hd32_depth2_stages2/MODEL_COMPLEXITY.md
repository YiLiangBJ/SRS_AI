# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `15,232`
- Parameter memory (FP32): `0.058 MiB (60,928 bytes)`
- MACs / sample: `14,592`
- Multiplications / sample: `14,760`
- Additions / sample: `15,191`
- Other scalar ops / sample: `89`
- FLOPs / sample estimate: `30,040`
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
| `stage_01_hidden_linear_01` | `once per sample` | `800` | `768` | `768` | `768` | `0` | `1,536` | Stage 1 first affine layer maps width 24 to hidden width 32. |
| `stage_01_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after the first hidden affine layer. |
| `stage_01_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 1 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_02_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 first affine layer maps width 144 to hidden width 32. |
| `stage_02_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 2 applies ReLU after the first hidden affine layer. |
| `stage_02_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_02_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 2 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 15232,
  "trainable_parameters_string": "15,232",
  "parameter_memory_bytes_fp32": 60928,
  "parameter_memory_bytes_fp32_string": "0.058 MiB (60,928 bytes)",
  "macs_per_sample": 14592,
  "macs_per_sample_string": "14,592",
  "multiplications_per_sample": 14760,
  "multiplications_per_sample_string": "14,760",
  "additions_per_sample": 15191,
  "additions_per_sample_string": "15,191",
  "other_scalar_ops_per_sample_estimate": 89,
  "other_scalar_ops_per_sample_estimate_string": "89",
  "flops_per_sample_estimate": 30040,
  "flops_per_sample_estimate_string": "30,040",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
