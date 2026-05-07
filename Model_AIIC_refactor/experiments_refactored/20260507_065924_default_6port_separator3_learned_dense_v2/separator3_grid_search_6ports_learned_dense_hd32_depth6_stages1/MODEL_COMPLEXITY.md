# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `9,920`
- Parameter memory (FP32): `0.038 MiB (39,680 bytes)`
- MACs / sample: `9,472`
- Multiplications / sample: `9,640`
- Additions / sample: `9,783`
- Other scalar ops / sample: `185`
- FLOPs / sample estimate: `19,608`
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
| `stage_01_hidden_linear_03` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 1 additional hidden affine layer keeps width 32. |
| `stage_01_hidden_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after hidden affine layer 3. |
| `stage_01_hidden_linear_04` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 1 additional hidden affine layer keeps width 32. |
| `stage_01_hidden_relu_04` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after hidden affine layer 4. |
| `stage_01_hidden_linear_05` | `once per sample` | `1,056` | `1,024` | `1,024` | `1,024` | `0` | `2,048` | Stage 1 additional hidden affine layer keeps width 32. |
| `stage_01_hidden_relu_05` | `once per sample` | `0` | `0` | `0` | `0` | `32` | `32` | Stage 1 applies ReLU after hidden affine layer 5. |
| `stage_01_joint_output` | `once per sample` | `4,752` | `4,608` | `4,608` | `4,608` | `0` | `9,216` | Stage 1 output affine layer maps hidden width 32 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 9920,
  "trainable_parameters_string": "9,920",
  "parameter_memory_bytes_fp32": 39680,
  "parameter_memory_bytes_fp32_string": "0.038 MiB (39,680 bytes)",
  "macs_per_sample": 9472,
  "macs_per_sample_string": "9,472",
  "multiplications_per_sample": 9640,
  "multiplications_per_sample_string": "9,640",
  "additions_per_sample": 9783,
  "additions_per_sample_string": "9,783",
  "other_scalar_ops_per_sample_estimate": 185,
  "other_scalar_ops_per_sample_estimate_string": "185",
  "flops_per_sample_estimate": 19608,
  "flops_per_sample_estimate_string": "19,608",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
