# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `120,992`
- Parameter memory (FP32): `0.462 MiB (483,968 bytes)`
- MACs / sample: `119,808`
- Multiplications / sample: `119,976`
- Additions / sample: `120,119`
- Other scalar ops / sample: `921`
- FLOPs / sample estimate: `241,016`
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
| `stage_01_hidden_linear_02` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 2. |
| `stage_01_hidden_linear_03` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 3. |
| `stage_01_hidden_linear_04` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_04` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 4. |
| `stage_01_hidden_linear_05` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_05` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 5. |
| `stage_01_hidden_linear_06` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_06` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 6. |
| `stage_01_hidden_linear_07` | `once per sample` | `16,512` | `16,384` | `16,384` | `16,384` | `0` | `32,768` | Stage 1 additional hidden affine layer keeps width 128. |
| `stage_01_hidden_relu_07` | `once per sample` | `0` | `0` | `0` | `0` | `128` | `128` | Stage 1 applies ReLU after hidden affine layer 7. |
| `stage_01_joint_output` | `once per sample` | `18,576` | `18,432` | `18,432` | `18,432` | `0` | `36,864` | Stage 1 output affine layer maps hidden width 128 to expanded width 144. |
| `stage_01_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Stage 1 output is summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |

## Raw Summary

```json
{
  "trainable_parameters": 120992,
  "trainable_parameters_string": "120,992",
  "parameter_memory_bytes_fp32": 483968,
  "parameter_memory_bytes_fp32_string": "0.462 MiB (483,968 bytes)",
  "macs_per_sample": 119808,
  "macs_per_sample_string": "119,808",
  "multiplications_per_sample": 119976,
  "multiplications_per_sample_string": "119,976",
  "additions_per_sample": 120119,
  "additions_per_sample_string": "120,119",
  "other_scalar_ops_per_sample_estimate": 921,
  "other_scalar_ops_per_sample_estimate_string": "921",
  "flops_per_sample_estimate": 241016,
  "flops_per_sample_estimate_string": "241,016",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
