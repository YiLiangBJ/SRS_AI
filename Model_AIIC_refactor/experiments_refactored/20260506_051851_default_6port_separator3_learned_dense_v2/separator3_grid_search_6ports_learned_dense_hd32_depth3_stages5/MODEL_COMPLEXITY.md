# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `49,120`
- Parameter memory (FP32): `0.187 MiB (196,480 bytes)`
- MACs / sample: `47,360`
- Multiplications / sample: `47,528`
- Additions / sample: `48,823`
- Other scalar ops / sample: `345`
- FLOPs / sample estimate: `96,696`
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
| `stage_01_hidden_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 1 additional hidden affine layer keeps width 32. |
| `stage_01_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after hidden affine layer 2. |
| `stage_01_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 1 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_02_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 first affine layer maps width 144 to hidden width 32. |
| `stage_02_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 2 applies ReLU after the first hidden affine layer. |
| `stage_02_hidden_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 2 additional hidden affine layer keeps width 32. |
| `stage_02_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 2 applies ReLU after hidden affine layer 2. |
| `stage_02_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 2 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_02_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 2 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_03_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 3 first affine layer maps width 144 to hidden width 32. |
| `stage_03_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 3 applies ReLU after the first hidden affine layer. |
| `stage_03_hidden_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 3 additional hidden affine layer keeps width 32. |
| `stage_03_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 3 applies ReLU after hidden affine layer 2. |
| `stage_03_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 3 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_03_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 3 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_04_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 4 first affine layer maps width 144 to hidden width 32. |
| `stage_04_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 4 applies ReLU after the first hidden affine layer. |
| `stage_04_hidden_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 4 additional hidden affine layer keeps width 32. |
| `stage_04_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 4 applies ReLU after hidden affine layer 2. |
| `stage_04_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 4 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_04_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 4 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `stage_05_hidden_linear_01` | `once per sample` | `4,640` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 5 first affine layer maps width 144 to hidden width 32. |
| `stage_05_hidden_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 5 applies ReLU after the first hidden affine layer. |
| `stage_05_hidden_linear_02` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 5 additional hidden affine layer keeps width 32. |
| `stage_05_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 5 applies ReLU after hidden affine layer 2. |
| `stage_05_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 5 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_05_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 5 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 49120,
  "trainable_parameters_string": "49,120",
  "parameter_memory_bytes_fp32": 196480,
  "parameter_memory_bytes_fp32_string": "0.187 MiB (196,480 bytes)",
  "macs_per_sample": 47360,
  "macs_per_sample_string": "47,360",
  "multiplications_per_sample": 47528,
  "multiplications_per_sample_string": "47,528",
  "additions_per_sample": 48823,
  "additions_per_sample_string": "48,823",
  "other_scalar_ops_per_sample_estimate": 345,
  "other_scalar_ops_per_sample_estimate_string": "345",
  "flops_per_sample_estimate": 96696,
  "flops_per_sample_estimate_string": "96,696",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
