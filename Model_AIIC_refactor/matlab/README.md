# Matlab Integration Guide

This folder provides two complete Matlab handoff paths for trained runs under Model_AIIC_refactor.

Use the ONNX path when you want Matlab to import a deployment-ready graph quickly.

Use the explicit Matlab bundle path when you want to inspect the architecture as clear matrix multiplies, biases, activations, ports, and stages.

If you mainly care about separator1, also read `SEPARATOR1_IMPLEMENTATION.md`.

## No-Path Workflow

If you do not want to add this folder to the Matlab path, use this workflow:

```matlab
cd('.../SRS_AI/Model_AIIC_refactor/matlab')
run('run_refactor_separator1_demo.m')
```

or:

```matlab
cd('.../SRS_AI/Model_AIIC_refactor/matlab')
run('run_refactor_model_demo.m')
```

This is supported directly.

The demo scripts add their own folder to the temporary Matlab path and build `exportDir` from the script location, so they do not depend on the current working directory being the repository root.

## What You Need To Know First

The refactored training pipeline saves one trained run per directory:

```text
<run_dir>/
  model.pth
  config.yaml
  tensorboard/
```

The minimum file you need is model.pth.

`config.yaml` is also useful because it mirrors the resolved `model_spec`, `training_spec`, and `metadata`.

If a run directory does not have `model.pth` but does have `checkpoint_batch_*.pth`, the exporters will automatically use the latest checkpoint file in that run directory.

The exporters also accept both schemas below:

- current refactor schema: `model_spec`, `training_spec`, `metadata`
- older schema still present in historical results: `model_config`, `training_config`, `metadata`

If you only know an experiment directory, not the exact run name, list its runs first:

```bash
python ./Model_AIIC_refactor/export_onnx.py \
  --exp_dir ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name> \
  --list_runs
```

The same run-selection logic also works for the Matlab bundle exporter:

```bash
python ./Model_AIIC_refactor/export_matlab_bundle.py \
  --exp_dir ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name> \
  --list_runs
```

By default, new training outputs are created under:

```text
./Model_AIIC_refactor/experiments_refactored/
```

## Path A: Existing Run To ONNX To Matlab

If you already have a trained run directory, export it from the SRS_AI root:

```bash
python ./Model_AIIC_refactor/export_onnx.py \
  --run_dir ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name> \
  --opset 13 \
  --dynamic_batch \
  --validate
```

This writes:

```text
<run_dir>/
  onnx_exports/
    <run_name>.onnx
    export_manifest.json
```

When `model_spec.normalize_energy=true`, the exported ONNX graph already includes the per-sample RMS normalization and the output rescaling step.

In Matlab:

1. Either enter the `matlab/` folder and run the script directly, or add this folder to the path.
2. Edit `exportDir` in `run_refactor_model_demo.m` or `run_refactor_onnx_demo.m`.
3. Run the script.

The import helpers now resolve export directories robustly across platforms.

That means these all work:

- absolute Windows paths such as `C:\work\SRS_AI\Model_AIIC_refactor\...`
- absolute Linux paths
- repo-root relative paths such as `./Model_AIIC_refactor/experiments_refactored/...`

The most common source of `ManifestNotFound` in Matlab is not Windows path separators.

It is using a relative path from the wrong current working directory.

The demo scripts now build `exportDir` from the script location, so they do not depend on the current working directory.

```matlab
run("./Model_AIIC_refactor/matlab/run_refactor_model_demo.m")
```

The unified function entry is:

```matlab
[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(".../<run_name>/onnx_exports", "onnx", 2);
```

### ONNX Matlab Inputs And Outputs

- Input shape: `N x (2*seq_len)`
- Input layout: `[real_part, imag_part]`
- Output shape: `N x num_ports x (2*seq_len)`
- Output layout: per port, `[real_part, imag_part]`

Example when `seq_len = 12` and `num_ports = 4`:

- Input: `N x 24`
- Output: `N x 4 x 24`

## Path B: Existing Run To Explicit Matlab Bundle

This path does not depend on ONNX.

It exports the trained weights into a Matlab-readable `.mat` file plus a JSON manifest.

This is the recommended path if the Matlab implementation team wants to see the model as explicit matrix multiplications and activations.

### Export From A Trained Run

```bash
python ./Model_AIIC_refactor/export_matlab_bundle.py \
  --run_dir ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name> \
  --batch_size 2
```

This writes:

```text
<run_dir>/
  matlab_exports/
    matlab_model_bundle.mat
    matlab_model_bundle_manifest.json
```

When `model_spec.normalize_energy=true`, the Matlab bundle helpers apply the same per-sample RMS normalization before the explicit layer stack and rescale the separated outputs afterwards, matching the PyTorch model behavior.

### What The Project Saves And What Matlab Reads

The project saves `model.pth` with these key pieces:

- `model_state_dict`: learned weights
- `model_spec`: architecture definition such as `model_type`, `seq_len`, `num_ports`, `num_stages`, `hidden_dim`, `mlp_depth`, `activation_type`
- `training_spec`: training configuration
- `metadata`: run lineage such as experiment name and run name

The Matlab bundle exporter reads those fields and materializes a Matlab-oriented package:

- `matlab_model_bundle_manifest.json`: human-readable architecture and file metadata
- `matlab_model_bundle.mat`: explicit weights plus sample input/output tensors

The MAT file always contains:

- `sample_input`: reference input with shape `N x (2*seq_len)`
- `reference_output`: PyTorch output for that input with shape `N x num_ports x (2*seq_len)`
- `pos_values`: port positions

For `separator2`, the MAT file also contains one fully materialized effective MLP per port and stage:

- `p01_s01_l01_weight_real`
- `p01_s01_l01_weight_imag`
- `p01_s01_l01_bias_real`
- `p01_s01_l01_bias_imag`
- and so on for each port, stage, and layer

For `separator1`, the MAT file contains separate real and imaginary branches:

- `p01_s01_real_l01_weight`
- `p01_s01_real_l01_bias`
- `p01_s01_imag_l01_weight`
- `p01_s01_imag_l01_bias`

Even if training used `share_weights_across_stages=True`, the exporter writes every effective port-stage block explicitly so the Matlab side can loop stage by stage without extra alias logic.

### Run The Explicit Matlab Demo

1. Add this folder to the Matlab path.
2. Edit `exportDir` in `run_refactor_model_demo.m` or `run_refactor_matlab_bundle_demo.m`.
3. Run the script.

If you prefer not to add paths manually, first `cd` into the `matlab/` folder and run the script there.

```matlab
run("./Model_AIIC_refactor/matlab/run_refactor_model_demo.m")
```

The unified function entry is:

```matlab
[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(".../<run_name>/matlab_exports", "bundle", 2);
```

If you want the lower-level explicit API, keep using:

```matlab
bundle = import_refactor_matlab_bundle(".../<run_name>/matlab_exports");
[outputData, debug] = predict_refactor_matlab_bundle(bundle, bundle.weights.sample_input);
```

If you pass a relative path manually, it is resolved in this order:

1. relative to the current Matlab working directory
2. relative to the repository root inferred from the `matlab/` helper folder

`info.debug.stage_outputs` or `debug.stage_outputs` keeps each stage result so you can inspect the refinement process.

## One Unified Matlab Entry

Use `import_refactor_model.m`, `demo_refactor_model_inference.m`, and `run_refactor_model_demo.m` if you want the exact same Matlab-side workflow regardless of whether the backend is ONNX or the explicit Matlab bundle.

- `mode = "auto"`: detect by manifest file
- `mode = "onnx"`: force ONNX import path
- `mode = "bundle"`: force explicit Matlab bundle path

## API-Style Entry Points

If you want to use these helpers as reusable APIs instead of demos, use these four functions directly:

- `import_refactor_model`: import either ONNX or bundle once
- `describe_refactor_model_io`: inspect input/output names, layouts, and dynamic dimensions
- `prepare_refactor_input`: normalize and validate input tensors
- `predict_refactor_model`: run unified inference on an imported handle or export directory

Example:

```matlab
modelHandle = import_refactor_model(".../<run_name>/onnx_exports", "auto");
ioSpec = describe_refactor_model_io(modelHandle, [], true);
inputData = prepare_refactor_input(modelHandle, randn(3, 24, "single"));
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData);
```

The printed shape spec uses `-1` for dynamic dimensions.

Examples:

- dynamic ONNX input: `[-1, 24]`
- dynamic ONNX output: `[-1, 6, 24]`
- explicit bundle output: `[-1, 4, 24]`

This is intended to make batch-dimension mismatches obvious when you move the helpers into another Matlab project.

The API helpers accept:

- an already imported `modelHandle`
- a raw ONNX manifest
- a raw Matlab bundle manifest
- a bundle struct from `import_refactor_matlab_bundle`

If you move the helper folder elsewhere, the recommended usage is still to pass an absolute export path or a path relative to the current Matlab working directory.

## Architecture Visibility In Matlab

The explicit Matlab bundle path is designed to make the architecture obvious.

For `separator2`, Matlab computes each layer as:

```text
y_R = x_R * W_R^T - x_I * W_I^T + b_R
y_I = x_R * W_I^T + x_I * W_R^T + b_I
```

Then it applies the configured activation between layers, such as `relu`, `split_relu`, `mod_relu`, `z_relu`, or `cardioid`.

Each port is processed independently inside each stage, then the model applies residual correction:

```text
y_recon = sum_port(h_port)
residual = y_input - y_recon
h_port = h_port + residual
```

For `separator1`, Matlab shows two ordinary dense MLP branches:

- one branch predicts the real part
- one branch predicts the imaginary part
- each branch is a cascade of `W x + b`, then ReLU, then final projection

## Recommended Choice

If the Matlab team wants deployment with the fewest moving parts, use the ONNX path.

If the Matlab team wants to re-implement the network with clear matrix math and explicit stage logic, use the explicit Matlab bundle path.

For architecture review and handoff, the explicit Matlab bundle path is usually better.

## Separator1 Priority Path

If compatibility and implementation clarity matter more than matching the ONNX deployment graph exactly, prefer separator1 plus the explicit Matlab bundle path.

Recommended files:

- `SEPARATOR1_IMPLEMENTATION.md`
- `predict_refactor_separator1_bundle_explicit.m`
- `run_refactor_separator1_demo.m`