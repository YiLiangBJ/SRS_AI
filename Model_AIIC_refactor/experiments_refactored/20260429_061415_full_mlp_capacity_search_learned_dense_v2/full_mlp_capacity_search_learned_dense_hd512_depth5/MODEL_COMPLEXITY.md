# Model Complexity

- Model type: `full_mlp`
- Input shape: `[-1, 24]`
- Output shape: `[-1, 6, 24]`
- Trainable parameters: `612,128`
- Parameter memory (FP32): `2.335 MiB (2,448,512 bytes)`
- MACs / sample: `610,304`
- Multiplications / sample: `610,472`
- Additions / sample: `610,327`
- Other scalar ops / sample: `1,561`
- FLOPs / sample estimate: `1,222,360`
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
| `joint_linear_01` | `once per sample` | `12,800` | `12,288` | `12,288` | `12,288` | `0` | `24,576` | Joint affine layer maps width 24 to width 512. |
| `joint_relu_01` | `once per sample` | `0` | `0` | `0` | `0` | `512` | `512` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_02` | `once per sample` | `262,656` | `262,144` | `262,144` | `262,144` | `0` | `524,288` | Joint affine layer maps width 512 to width 512. |
| `joint_relu_02` | `once per sample` | `0` | `0` | `0` | `0` | `512` | `512` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_03` | `once per sample` | `262,656` | `262,144` | `262,144` | `262,144` | `0` | `524,288` | Joint affine layer maps width 512 to width 512. |
| `joint_relu_03` | `once per sample` | `0` | `0` | `0` | `0` | `512` | `512` | ReLU is applied after every hidden affine layer in the joint MLP. |
| `joint_linear_04` | `once per sample` | `73,872` | `73,728` | `73,728` | `73,728` | `0` | `147,456` | Joint affine layer maps width 512 to width 144. |

## Raw Summary

```json
{
  "trainable_parameters": 612128,
  "trainable_parameters_string": "612,128",
  "parameter_memory_bytes_fp32": 2448512,
  "parameter_memory_bytes_fp32_string": "2.335 MiB (2,448,512 bytes)",
  "macs_per_sample": 610304,
  "macs_per_sample_string": "610,304",
  "multiplications_per_sample": 610472,
  "multiplications_per_sample_string": "610,472",
  "additions_per_sample": 610327,
  "additions_per_sample_string": "610,327",
  "other_scalar_ops_per_sample_estimate": 1561,
  "other_scalar_ops_per_sample_estimate_string": "1,561",
  "flops_per_sample_estimate": 1222360,
  "flops_per_sample_estimate_string": "1,222,360",
  "batch_scaling_rule": "Multiply the per-sample counts by runtime batch size N for a first-order batch estimate."
}
```
