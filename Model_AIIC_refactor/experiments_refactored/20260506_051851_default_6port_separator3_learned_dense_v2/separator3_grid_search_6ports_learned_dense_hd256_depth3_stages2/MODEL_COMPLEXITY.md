# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `249,408`
- Parameter memory (FP32): `0.951 MiB (997,632 bytes)`
- MACs / sample: `247,808`
- Multiplications / sample: `247,976`
- Additions / sample: `248,407`
- Other scalar ops / sample: `1,049`
- FLOPs / sample estimate: `497,432`
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
| `stage_01_hidden_linear_01` | `once per sample` | `6,400` | `6,144` | `6,144` | `6,144` | `0` | `12,288` | Stage 1 first affine layer maps width 24 to hidden width 256. |
| `stage_01_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 1 applies ReLU after the first hidden affine layer. |
| `stage_01_hidden_linear_02` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Stage 1 additional hidden affine layer keeps width 256. |
| `stage_01_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 1 applies ReLU after hidden affine layer 2. |
| `stage_01_joint_output` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 1 output affine layer maps hidden width 256 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_02_hidden_linear_01` | `once per sample` | `37,120` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 2 first affine layer maps width 144 to hidden width 256. |
| `stage_02_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 2 applies ReLU after the first hidden affine layer. |
| `stage_02_hidden_linear_02` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Stage 2 additional hidden affine layer keeps width 256. |
| `stage_02_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 2 applies ReLU after hidden affine layer 2. |
| `stage_02_joint_output` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 2 output affine layer maps hidden width 256 to expanded width 144. |
| `stage_02_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 2 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 249408,
  "trainable_parameters_string": "249,408",
  "parameter_memory_bytes_fp32": 997632,
  "parameter_memory_bytes_fp32_string": "0.951 MiB (997,632 bytes)",
  "macs_per_sample": 247808,
  "macs_per_sample_string": "247,808",
  "multiplications_per_sample": 247976,
  "multiplications_per_sample_string": "247,976",
  "additions_per_sample": 248407,
  "additions_per_sample_string": "248,407",
  "other_scalar_ops_per_sample_estimate": 1049,
  "other_scalar_ops_per_sample_estimate_string": "1,049",
  "flops_per_sample_estimate": 497432,
  "flops_per_sample_estimate_string": "497,432",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
