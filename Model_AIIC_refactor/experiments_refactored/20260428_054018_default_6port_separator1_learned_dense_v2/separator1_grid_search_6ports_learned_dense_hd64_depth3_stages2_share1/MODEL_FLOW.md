# Model Flow

- Model type: `separator1`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `78,624`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

2 refinement stages; each stage runs one real branch and one imag branch per port, then applies residual correction.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | The task provides one mixed complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor width. |
| `replicated_port_features` | `[-1, 6, 24]` | `-` | `0` | `0` | The separator starts each port estimate from the same mixed input, so a port axis of size 6 is introduced. |
| `port_stage_input` | `[-1, 24]` | `per port, per stage (2 stages total, stage weights shared)` | `0` | `0` | Each port-stage block consumes one real-stacked sequence of width 24. |
| `real_branch_hidden_1` | `[-1, 64]` | `per port, per stage (2 stages total, stage weights shared)` | `1,600` | `9,600` | The first branch affine layer maps width 24 to hidden width 64. |
| `imag_branch_hidden_1` | `[-1, 64]` | `per port, per stage (2 stages total, stage weights shared)` | `1,600` | `9,600` | The first branch affine layer maps width 24 to hidden width 64. |
| `real_branch_output` | `[-1, 12]` | `per port, per stage (2 stages total, stage weights shared)` | `780` | `4,680` | The real branch final affine layer reduces hidden width 64 to one real channel width 12. |
| `imag_branch_output` | `[-1, 12]` | `per port, per stage (2 stages total, stage weights shared)` | `780` | `4,680` | The imaginary branch final affine layer reduces hidden width 64 to one imaginary channel width 12. |
| `port_output` | `[-1, 24]` | `per port, per stage (2 stages total, stage weights shared)` | `0` | `0` | Concatenating one real width-12 output and one imag width-12 output reconstructs one width-24 port tensor. |
| `stacked_stage_output` | `[-1, 6, 24]` | `per stage (2 stages total)` | `0` | `0` | Stacking all 6 ports reintroduces the port axis while keeping each port width at 24. |
| `residual_corrected_output` | `[-1, 6, 24]` | `per stage (2 stages total)` | `0` | `0` | Residual correction adds the same mixed-signal residual back to every port estimate, so the shape is unchanged. |
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
  "hidden_dim": 64,
  "mlp_depth": 3,
  "num_stages": 2,
  "share_weights_across_stages": true,
  "use_hidden_layer_norm": false,
  "use_hidden_relu": true,
  "residual_correction_mode": "learned_dense",
  "model_type": "separator1",
  "num_ports": 6,
  "num_params": 78624
}
```
