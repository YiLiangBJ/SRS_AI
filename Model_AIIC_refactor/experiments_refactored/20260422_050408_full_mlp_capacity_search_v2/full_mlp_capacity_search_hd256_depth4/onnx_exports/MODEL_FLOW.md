# Model Flow

- Model type: `full_mlp`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `109,200`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

One joint MLP processes the mixed signal once and reshapes the output into all ports.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | Task output is one complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor rank or width. |
| `joint_hidden_1` | `[-1, 256]` | `-` | `6,400` | `6,400` | The first affine layer maps the input width 24 into the configured hidden width 256. |
| `joint_hidden_1_relu` | `[-1, 256]` | `-` | `0` | `0` | ReLU is elementwise, so it keeps the hidden shape unchanged. |
| `joint_hidden_2` | `[-1, 256]` | `-` | `65,792` | `65,792` | This affine layer keeps the same hidden width 256 while adding capacity. |
| `joint_hidden_2_relu` | `[-1, 256]` | `-` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `joint_linear_output` | `[-1, 144]` | `-` | `37,008` | `37,008` | The output affine layer expands hidden width 256 to flat joint output width 144. |
| `reshaped_channels` | `[-1, 6, 24]` | `-` | `0` | `0` | The flat width 144 is partitioned into 6 port blocks of width 24. |
| `separated_channels` | `[-1, 6, 24]` | `-` | `0` | `0` | Rescaling restores amplitude but keeps the separated tensor shape unchanged. |

## Model Spec

```json
{
  "seq_len": 12,
  "pos_values": [
    0,
    2,
    4,
    6,
    8,
    10
  ],
  "normalize_energy": true,
  "hidden_dim": 256,
  "mlp_depth": 4,
  "model_type": "full_mlp",
  "num_ports": 6,
  "num_params": 109200
}
```
