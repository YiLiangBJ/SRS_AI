# Model Complexity

- Model type: `separator3`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `24,768`
- Parameter memory (FP32): `0.094 MiB (99,072 bytes)`
- MACs / sample: `24,192`
- Multiplications / sample: `24,360`
- Additions / sample: `24,791`
- Other scalar ops / sample: `25`
- FLOPs / sample estimate: `49,176`
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
| `hidden_linear` | `once per sample` | `3,600` | `3,456` | `3,456` | `3,456` | `0` | `6,912` | Hidden affine layer maps width 24 to expanded width 144. |
| `hidden_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Hidden features are summed across ports, compared with the mixed input, and corrected with a learned dense per-port residual mask. |
| `output_linear` | `once per sample` | `20,880` | `20,736` | `20,736` | `20,736` | `0` | `41,472` | Output affine layer keeps expanded width 144 and predicts the final per-port representation. |
| `output_residual_correction` | `once per sample` | `0` | `0` | `0` | `288` | `0` | `288` | Output features are again corrected with a learned dense per-port residual mask before final output. |

## Raw Summary

```json
{
  "trainable_parameters": 24768,
  "trainable_parameters_string": "24,768",
  "parameter_memory_bytes_fp32": 99072,
  "parameter_memory_bytes_fp32_string": "0.094 MiB (99,072 bytes)",
  "macs_per_sample": 24192,
  "macs_per_sample_string": "24,192",
  "multiplications_per_sample": 24360,
  "multiplications_per_sample_string": "24,360",
  "additions_per_sample": 24791,
  "additions_per_sample_string": "24,791",
  "other_scalar_ops_per_sample_estimate": 25,
  "other_scalar_ops_per_sample_estimate_string": "25",
  "flops_per_sample_estimate": 49176,
  "flops_per_sample_estimate_string": "49,176",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
