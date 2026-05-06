# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `529,536`
- Parameter memory (FP32): `2.020 MiB (2,118,144 bytes)`
- MACs / sample: `526,336`
- Multiplications / sample: `526,504`
- Additions / sample: `527,511`
- Other scalar ops / sample: `2,073`
- FLOPs / sample estimate: `1,056,088`
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
| `stage_03_hidden_linear_01` | `once per sample` | `37,120` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 3 first affine layer maps width 144 to hidden width 256. |
| `stage_03_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 3 applies ReLU after the first hidden affine layer. |
| `stage_03_hidden_linear_02` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Stage 3 additional hidden affine layer keeps width 256. |
| `stage_03_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 3 applies ReLU after hidden affine layer 2. |
| `stage_03_joint_output` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 3 output affine layer maps hidden width 256 to expanded width 144. |
| `stage_03_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 3 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_04_hidden_linear_01` | `once per sample` | `37,120` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 4 first affine layer maps width 144 to hidden width 256. |
| `stage_04_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 4 applies ReLU after the first hidden affine layer. |
| `stage_04_hidden_linear_02` | `once per sample` | `65,792` | `65,536` | `65,536` | `65,536` | `0` | `131,072` | Stage 4 additional hidden affine layer keeps width 256. |
| `stage_04_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `256` | `256` | Stage 4 applies ReLU after hidden affine layer 2. |
| `stage_04_joint_output` | `once per sample` | `37,008` | `36,864` | `36,864` | `36,864` | `0` | `73,728` | Stage 4 output affine layer maps hidden width 256 to expanded width 144. |
| `stage_04_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 4 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 529536,
  "trainable_parameters_string": "529,536",
  "parameter_memory_bytes_fp32": 2118144,
  "parameter_memory_bytes_fp32_string": "2.020 MiB (2,118,144 bytes)",
  "macs_per_sample": 526336,
  "macs_per_sample_string": "526,336",
  "multiplications_per_sample": 526504,
  "multiplications_per_sample_string": "526,504",
  "additions_per_sample": 527511,
  "additions_per_sample_string": "527,511",
  "other_scalar_ops_per_sample_estimate": 2073,
  "other_scalar_ops_per_sample_estimate_string": "2,073",
  "flops_per_sample_estimate": 1056088,
  "flops_per_sample_estimate_string": "1,056,088",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
