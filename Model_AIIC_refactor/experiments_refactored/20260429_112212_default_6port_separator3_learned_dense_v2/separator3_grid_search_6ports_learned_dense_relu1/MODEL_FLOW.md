# Model Flow

- Model type: `separator3`
- Input layout: `N x (2*seq_len) real-stacked float32 = [real_part, imag_part]`
- Input shape: `[-1, 24]`
- Output layout: `N x num_ports x (2*seq_len) real-stacked float32`
- Output shape: `[-1, 6, 24]`
- Total trainable parameters: `24,768`
- Dynamic dimensions: Dynamic dimensions are written as -1.

## Flow Summary

One hidden joint linear block and one output joint linear block, each followed by learned-dense residual correction; hidden ReLU is optional and output stays linear.

## Node Shapes

| Step | Shape | Repeat | Params / occurrence | Effective params | Why this shape |
|---|---|---|---|---|---|
| `mixed_signal` | `[-1, 24]` | `-` | `0` | `0` | The task provides one mixed complex sequence flattened into real and imaginary blocks. |
| `normalized_input` | `[-1, 24]` | `-` | `0` | `0` | Normalization rescales values but does not change tensor width. |
| `hidden_linear` | `[-1, 144]` | `-` | `3,600` | `3,600` | The hidden affine layer maps width 24 to expanded width 144 = num_ports * (2 * seq_len). |
| `hidden_relu` | `[-1, 144]` | `-` | `0` | `0` | ReLU is applied only inside the hidden block and does not change tensor width. |
| `hidden_port_features` | `[-1, 6, 24]` | `-` | `0` | `0` | The expanded width 144 is partitioned into 6 port blocks of width 24. |
| `hidden_residual_corrected` | `[-1, 6, 24]` | `-` | `0` | `0` | The hidden block sums across ports, computes the mixed-signal residual, and adds a per-port learned dense weighting of that residual back to each port estimate. |
| `flattened_hidden` | `[-1, 144]` | `-` | `0` | `0` | The per-port hidden representation is flattened so the output linear layer can mix information across all ports jointly. |
| `output_linear` | `[-1, 144]` | `-` | `20,880` | `20,880` | The output affine layer keeps expanded width 144 and produces the final per-port real-stacked blocks. |
| `output_port_features` | `[-1, 6, 24]` | `-` | `0` | `0` | The output width 144 is partitioned back into 6 port blocks of width 24. |
| `output_residual_corrected` | `[-1, 6, 24]` | `-` | `0` | `0` | A second learned-dense residual correction re-enforces that the separated outputs sum back to the mixed input while preserving signed residual contributions. |
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
  "use_hidden_relu": true,
  "residual_correction_mode": "learned_dense",
  "model_type": "separator3",
  "num_ports": 6,
  "num_params": 24768
}
```
