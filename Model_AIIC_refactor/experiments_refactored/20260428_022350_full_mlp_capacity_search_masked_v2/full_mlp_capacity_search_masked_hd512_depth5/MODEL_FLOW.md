# Model Flow

- Model type: `full_mlp`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `611,984`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

One joint MLP processes the mixed signal once and reshapes the output into all ports.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | Task output is one complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor rank or width. |
| `joint_hidden_1` | `[-1, 512]` | `-` | `12,800` | `12,800` | The first affine layer maps the input width 24 into the configured hidden width 512. |
| `joint_hidden_1_relu` | `[-1, 512]` | `-` | `0` | `0` | ReLU is elementwise, so it keeps the hidden shape unchanged. |
| `joint_hidden_2` | `[-1, 512]` | `-` | `262,656` | `262,656` | This affine layer keeps the same hidden width 512 while adding capacity. |
| `joint_hidden_2_relu` | `[-1, 512]` | `-` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `joint_hidden_3` | `[-1, 512]` | `-` | `262,656` | `262,656` | This affine layer keeps the same hidden width 512 while adding capacity. |
| `joint_hidden_3_relu` | `[-1, 512]` | `-` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `joint_linear_output` | `[-1, 144]` | `-` | `73,872` | `73,872` | The output affine layer expands hidden width 512 to flat joint output width 144. |
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
  "hidden_dim": 512,
  "mlp_depth": 5,
  "residual_correction_mode": "masked",
  "model_type": "full_mlp",
  "num_ports": 6,
  "num_params": 611984
}
```
