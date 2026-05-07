# Model Flow

- Model type: `separator3`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `10,976`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

1 joint refinement stages; stage 1 maps 24 to 144, later stages map 144 to 144, and every stage ends with learned-dense residual correction.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | The task provides one mixed complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor width. |
| `stage_1_input` | `[-1, 24]` | `once` | `0` | `0` | The first stage consumes the mixed signal width 24 directly. |
| `stage_1_hidden_1` | `[-1, 32]` | `once` | `800` | `800` | The first affine layer of stage 1 maps width 24 to hidden width 32. |
| `stage_1_hidden_1_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_1_hidden_2` | `[-1, 32]` | `once` | `1,056` | `1,056` | This affine layer keeps hidden width 32 inside stage 1. |
| `stage_1_hidden_2_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `stage_1_hidden_3` | `[-1, 32]` | `once` | `1,056` | `1,056` | This affine layer keeps hidden width 32 inside stage 1. |
| `stage_1_hidden_3_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `stage_1_hidden_4` | `[-1, 32]` | `once` | `1,056` | `1,056` | This affine layer keeps hidden width 32 inside stage 1. |
| `stage_1_hidden_4_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `stage_1_hidden_5` | `[-1, 32]` | `once` | `1,056` | `1,056` | This affine layer keeps hidden width 32 inside stage 1. |
| `stage_1_hidden_5_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `stage_1_hidden_6` | `[-1, 32]` | `once` | `1,056` | `1,056` | This affine layer keeps hidden width 32 inside stage 1. |
| `stage_1_hidden_6_relu` | `[-1, 32]` | `once` | `0` | `0` | ReLU is elementwise, so the hidden width stays the same. |
| `stage_1_joint_output` | `[-1, 144]` | `once` | `4,752` | `4,752` | The final affine layer of stage 1 maps hidden width 32 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_1_port_features` | `[-1, 6, 24]` | `once` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_1_residual_corrected` | `[-1, 6, 24]` | `once` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
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
  "hidden_dim": 32,
  "num_stages": 1,
  "mlp_depth": 7,
  "residual_correction_mode": "learned_dense",
  "model_type": "separator3",
  "num_ports": 6,
  "stage_hidden_dims": [
    32
  ],
  "num_params": 10976
}
```
