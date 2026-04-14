# Separator1 Matlab Notes

This note is the easiest handoff path if the implementation team mainly cares about separator1.

Separator1 is a real-valued network in the deployment path.

It does not use complex block matrices like separator2.

Instead, each port-stage block contains two ordinary real MLP branches:

- one branch predicts the real part of the separated channel
- one branch predicts the imaginary part of the separated channel

Both branches take the same real-stacked input:

```text
input = [real_part, imag_part]
shape = N x (2*seq_len)
```

## Bundle Field Naming

For separator1, the Matlab bundle stores weights with these field patterns:

- `p01_s01_real_l01_weight`
- `p01_s01_real_l01_bias`
- `p01_s01_imag_l01_weight`
- `p01_s01_imag_l01_bias`

The naming means:

- `p01`: port 1
- `s01`: stage 1
- `real` or `imag`: branch
- `l01`: layer 1 inside that branch MLP

## Tensor Shapes

For separator1, keep these shapes in mind:

- mixed input: `N x (2*seq_len)`
- one branch hidden layer: `N x hidden_dim`
- one branch final layer: `N x seq_len`
- one port output: `N x (2*seq_len)`
- all ports in one stage: `N x num_ports x (2*seq_len)`

For the common 6-port setup in this repo:

- `seq_len = 12`
- input width = `24`
- output width per port = `24`

If `mlp_depth = 3`, each branch has:

- layer 1: `24 -> hidden_dim`
- layer 2: `hidden_dim -> hidden_dim`
- layer 3: `hidden_dim -> 12`

The final separator output for one port is built as:

```text
[real_out, imag_out]
```

This is block-stacked, not interleaved.

## Forward Structure

For one port in one stage, Matlab computes:

```text
real_1 = ReLU(input * W_real_1^T + b_real_1)
real_2 = ReLU(real_1 * W_real_2^T + b_real_2)
real_out = real_2 * W_real_3^T + b_real_3

imag_1 = ReLU(input * W_imag_1^T + b_imag_1)
imag_2 = ReLU(imag_1 * W_imag_2^T + b_imag_2)
imag_out = imag_2 * W_imag_3^T + b_imag_3

port_output = [real_out, imag_out]
```

That means the bundle and Matlab reference both use:

```text
[real_0 ... real_(L-1), imag_0 ... imag_(L-1)]
```

and not:

```text
[real_0, imag_0, real_1, imag_1, ...]
```

Then the stage performs residual refinement:

```text
y_recon = sum(port_output over all ports)
residual = input_mixed - y_recon
refined_port_output = port_output + residual
```

This refined tensor becomes the input to the next stage.

## Reference Matlab Entry Points

Use these files:

- `import_refactor_matlab_bundle.m`
- `predict_refactor_separator1_bundle_explicit.m`
- `run_refactor_separator1_demo.m`

The explicit helper is intentionally easy to read.

It keeps the loops visible:

- stage loop
- port loop
- layer loop
- branch split into real and imag

It also stores debug information in:

- `debug.stage_outputs`
- `debug.stage_port_layer_traces`
- `debug.port_layer_outputs` (backward-compatible alias)

Each `debug.stage_port_layer_traces{stageIdx, portIdx}{layerIdx}` entry stores:

- `real_input`, `imag_input`
- `real_weight`, `imag_weight`
- `real_bias`, `imag_bias`
- `real_affine`, `imag_affine`
- `real_post_activation`, `imag_post_activation`

This makes the `Wx + b` step directly visible during debugging.

## Suggested Re-implementation Order

If another team is rewriting separator1 in Matlab or C, this is the safest order:

1. Load one exported bundle and verify `reference_output` with the provided Matlab explicit helper.
2. Re-implement one branch MLP as repeated `y = xW^T + b`, then ReLU on non-final layers.
3. Re-implement one full port as `real_branch + imag_branch` concatenation.
4. Re-implement one stage as all ports plus residual refinement.
5. Repeat stages and compare against `debug.stage_outputs` and `debug.stage_port_layer_traces`.

## Recommendation

If your Matlab team wants the clearest possible separator1 reference, start from `run_refactor_separator1_demo.m` rather than the generic unified demo.