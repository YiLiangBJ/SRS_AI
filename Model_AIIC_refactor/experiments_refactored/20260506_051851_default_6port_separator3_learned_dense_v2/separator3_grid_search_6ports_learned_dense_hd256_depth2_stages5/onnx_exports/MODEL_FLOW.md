# Model Flow

- Model type: `separator3`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `340,640`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

5 joint refinement stages; stage 1 maps 24 to 144, later stages map 144 to 144, and every stage ends with learned-dense residual correction.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | The task provides one mixed complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor width. |
| `stage_1_input` | `[-1, 24]` | `once` | `0` | `0` | The first stage consumes the mixed signal width 24 directly. |
| `stage_1_hidden_1` | `[-1, 256]` | `once` | `6,400` | `6,400` | The first affine layer of stage 1 maps width 24 to hidden width 256. |
| `stage_1_hidden_1_relu` | `[-1, 256]` | `once` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_1_joint_output` | `[-1, 144]` | `once` | `37,008` | `37,008` | The final affine layer of stage 1 maps hidden width 256 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_1_port_features` | `[-1, 6, 24]` | `once` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_1_residual_corrected` | `[-1, 6, 24]` | `once` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
| `stage_1_flattened_output` | `[-1, 144]` | `once` | `0` | `0` | The per-port output is flattened back to one joint vector so the next stage can process all ports jointly again. |
| `stage_2_input` | `[-1, 144]` | `stage 2` | `0` | `0` | Stage 2 consumes the flattened joint output of the previous stage, so the width is 144. |
| `stage_2_hidden_1` | `[-1, 256]` | `stage 2` | `37,120` | `37,120` | The first affine layer of stage 2 maps width 144 to hidden width 256. |
| `stage_2_hidden_1_relu` | `[-1, 256]` | `stage 2` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_2_joint_output` | `[-1, 144]` | `stage 2` | `37,008` | `37,008` | The final affine layer of stage 2 maps hidden width 256 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_2_port_features` | `[-1, 6, 24]` | `stage 2` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_2_residual_corrected` | `[-1, 6, 24]` | `stage 2` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
| `stage_2_flattened_output` | `[-1, 144]` | `stage 2` | `0` | `0` | The per-port output is flattened back to one joint vector so the next stage can process all ports jointly again. |
| `stage_3_input` | `[-1, 144]` | `stage 3` | `0` | `0` | Stage 3 consumes the flattened joint output of the previous stage, so the width is 144. |
| `stage_3_hidden_1` | `[-1, 256]` | `stage 3` | `37,120` | `37,120` | The first affine layer of stage 3 maps width 144 to hidden width 256. |
| `stage_3_hidden_1_relu` | `[-1, 256]` | `stage 3` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_3_joint_output` | `[-1, 144]` | `stage 3` | `37,008` | `37,008` | The final affine layer of stage 3 maps hidden width 256 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_3_port_features` | `[-1, 6, 24]` | `stage 3` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_3_residual_corrected` | `[-1, 6, 24]` | `stage 3` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
| `stage_3_flattened_output` | `[-1, 144]` | `stage 3` | `0` | `0` | The per-port output is flattened back to one joint vector so the next stage can process all ports jointly again. |
| `stage_4_input` | `[-1, 144]` | `stage 4` | `0` | `0` | Stage 4 consumes the flattened joint output of the previous stage, so the width is 144. |
| `stage_4_hidden_1` | `[-1, 256]` | `stage 4` | `37,120` | `37,120` | The first affine layer of stage 4 maps width 144 to hidden width 256. |
| `stage_4_hidden_1_relu` | `[-1, 256]` | `stage 4` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_4_joint_output` | `[-1, 144]` | `stage 4` | `37,008` | `37,008` | The final affine layer of stage 4 maps hidden width 256 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_4_port_features` | `[-1, 6, 24]` | `stage 4` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_4_residual_corrected` | `[-1, 6, 24]` | `stage 4` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
| `stage_4_flattened_output` | `[-1, 144]` | `stage 4` | `0` | `0` | The per-port output is flattened back to one joint vector so the next stage can process all ports jointly again. |
| `stage_5_input` | `[-1, 144]` | `stage 5` | `0` | `0` | Stage 5 consumes the flattened joint output of the previous stage, so the width is 144. |
| `stage_5_hidden_1` | `[-1, 256]` | `stage 5` | `37,120` | `37,120` | The first affine layer of stage 5 maps width 144 to hidden width 256. |
| `stage_5_hidden_1_relu` | `[-1, 256]` | `stage 5` | `0` | `0` | ReLU is applied after every hidden linear layer and keeps the hidden shape unchanged. |
| `stage_5_joint_output` | `[-1, 144]` | `stage 5` | `37,008` | `37,008` | The final affine layer of stage 5 maps hidden width 256 to expanded width 144 = num_ports * (2 * seq_len). |
| `stage_5_port_features` | `[-1, 6, 24]` | `stage 5` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `stage_5_residual_corrected` | `[-1, 6, 24]` | `stage 5` | `0` | `0` | Stage output is summed across ports, compared with the mixed input, and corrected with one learned dense per-port residual mask for that stage. |
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
  "num_stages": 5,
  "mlp_depth": 2,
  "residual_correction_mode": "learned_dense",
  "model_type": "separator3",
  "num_ports": 6,
  "stage_hidden_dims": [
    256,
    256,
    256,
    256,
    256
  ],
  "num_params": 340640
}
```
