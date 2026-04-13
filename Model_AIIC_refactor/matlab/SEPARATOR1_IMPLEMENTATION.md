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
- `debug.port_layer_outputs`

`debug.port_layer_outputs{stageIdx, portIdx}{layerIdx, 1}` is the real-branch output.

`debug.port_layer_outputs{stageIdx, portIdx}{layerIdx, 2}` is the imag-branch output.

## Recommendation

If your Matlab team wants the clearest possible separator1 reference, start from `run_refactor_separator1_demo.m` rather than the generic unified demo.