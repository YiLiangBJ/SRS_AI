# Model_AIIC_refactor Unified Guide

This is the single canonical help document for `Model_AIIC_refactor`.

It replaces the old split documentation that used to live in:

- `CONFIG_GUIDE.md`
- `STANDALONE_EVAL_PLOT.md`
- `CHECKPOINT_FORMAT_SPEC.md`
- `matlab/README.md`
- `matlab/SEPARATOR1_IMPLEMENTATION.md`

Generated files such as `TRAINING_REPORT.md` are not part of this guide. They remain per experiment as run artifacts.

## 1. Core Idea

The project is experiment-first.

You do not manually pair model config and training config on the CLI. You launch a named experiment from `configs/experiments.yaml`, and the workflow resolves:

1. model recipes
2. training recipe
3. expanded search-space variants
4. the final executable run plan

## 2. Important Terms

- `experiment`: a named workflow preset from `experiments.yaml`
- `model recipe`: one entry from `model_configs.yaml`
- `training recipe`: one entry from `training_configs.yaml`
- `model label`: the expanded model variant name after search-space resolution
- `training label`: the expanded training variant name after search-space resolution
- `run_name`: the final unique executable run identifier
- `model_spec`: the resolved model schema saved with the run
- `training_spec`: the resolved training schema saved with the run

Use these names consistently in code, reports, checkpoints, and exports.

## 3. Repository Workflow Architecture

The project uses thin CLI entrypoints plus shared workflow modules.

Thin CLIs:

- `train.py`
- `evaluate_models_refactored.py`
- `export_onnx.py`
- `export_matlab_bundle.py`
- `plot.py`

Shared workflow modules:

- `workflows/train_workflow.py`
- `workflows/postprocess_workflow.py`
- `workflows/evaluation_workflow.py`
- `workflows/export_workflow.py`
- `workflows/matlab_export_workflow.py`
- `workflows/plotting_workflow.py`
- `workflows/reporting.py`

This layout keeps research iteration practical:

- CLI usage stays simple
- notebook or benchmark code can call workflow APIs directly
- artifact schemas are shared across training, evaluation, and export
- logic changes happen once in the workflow layer instead of being duplicated in scripts

## 4. Configuration Model

### 4.1 Recommended split

- `configs/model_configs.yaml`: architecture, port layout, ONNX-related model options, and model-side search spaces
- `configs/training_configs.yaml`: optimization, data sampling, loss, validation cadence, scheduler settings, and checkpoint cadence
- `configs/experiments.yaml`: reusable workflow presets binding model recipes to one training recipe

### 4.2 Supported config patterns

Single model config:

```yaml
separator1_default:
  model_type: separator1
  pos_values: [0, 3, 6, 9]
  hidden_dim: 64
  num_stages: 3
```

Fixed params plus search space:

```yaml
separator1_grid_search:
  model_type: separator1
  fixed_params:
    pos_values: [0, 3, 6, 9]
    mlp_depth: 3
  search_space:
    hidden_dim: [32, 64, 128]
    num_stages: [2, 3]
```

Experiment preset:

```yaml
experiments:
  compare_default_models:
    model_configs:
      - separator1_default
      - separator2_default
    training_config: snr_range_0_30_perSample
```

### 4.3 Practical conventions

- Keep `pos_values`, `seq_len`, width/depth, activation, ONNX compatibility, and model-side normalization on the model side.
- Keep SNR policy, TDL policy, loss, LR, validation cadence, scheduler policy, and checkpoint cadence on the training side.
- Put workflow intent in `experiments.yaml`: smoke tests, architecture comparisons, benchmark presets, export candidates, and sweeps.
- If a field is a deliberate scientific sweep, put it in `search_space`.
- If a field is constant, keep it flat or place it in `fixed_params`.
- Prefer narrow sweeps aligned to one question instead of one large unfocused Cartesian product.

### 4.4 Inspect plans before launch

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1 \
  --plan_only \
  --device cpu
```

## 5. Training

### 5.1 Common commands

Train one named experiment:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda
```

Train, then evaluate and plot:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda \
  --eval_after_train \
  --plot_after_eval
```

Benchmark preset with batch-count override:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment perf_quick \
  --num_batches 100 \
  --device cpu
```

### 5.2 Train CLI summary

| Argument | Meaning |
|---|---|
| `--experiment` | Required experiment name from `experiments.yaml` |
| `--batch_size` | Optional override applied after recipe resolution |
| `--num_batches` | Optional override applied after recipe resolution |
| `--device` | `auto`, `cpu`, `cuda`, `cuda:0`, ... |
| `--save_dir` | Parent output directory |
| `--no-amp` | Disable mixed precision |
| `--no-compile` | Disable `torch.compile` |
| `--eval_after_train` | Run evaluation after training |
| `--eval_snr_range` | SNR setting for evaluation |
| `--eval_tdl` | Comma-separated TDL list for evaluation |
| `--eval_num_batches` | Number of evaluation batches |
| `--eval_batch_size` | Evaluation batch size |
| `--plot_after_eval` | Generate plots after evaluation |
| `--export_onnx_after_train` | Export ONNX after training |
| `--onnx_export_selection` | Export `best` or `all` runs |
| `--onnx_output_dir` | Single-run ONNX output override |
| `--onnx_opset` | ONNX opset version |
| `--onnx_batch_size` | Dummy tracing batch size for ONNX export |
| `--onnx_dynamic_batch` | Export ONNX with a dynamic batch axis |
| `--onnx_validate` | Run ONNX checker and ORT validation |
| `--export_matlab_after_train` | Export Matlab bundle after training |
| `--matlab_export_selection` | Export `best` or `all` runs as Matlab bundles |
| `--matlab_output_dir` | Single-run Matlab bundle directory override |
| `--plan_only` | Print the run plan and exit |

### 5.3 Current training behavior worth knowing

- `loss_type=log` now means mean log-NMSE, not raw log-MSE.
- `loss_type=normalized` now means mean per-sample NMSE.
- validation averages multiple batches drawn from the same SNR distribution as training.
- the default LR scheduler is intentionally smoother than before.
- model-side energy normalization is inside the model when `model_spec.normalize_energy=true`.

## 6. Artifact Layout

```text
Model_AIIC_refactor/
  experiments_refactored/
    <timestamp>_<experiment_name>/
      TRAINING_REPORT.md
      <run_name>/
        model.pth
        config.yaml
        tensorboard/
        evaluations/
          <timestamp>/
            evaluation_results.json
            evaluation_results.npy
            plots/
        onnx_exports/
          <run_name>.onnx
          export_manifest.json
        matlab_exports/
          matlab_model_bundle.mat
          matlab_model_bundle_manifest.json
    evaluations/
      <timestamp>_<scope>/
        evaluation_results.json
        evaluation_results.npy
        plots/
```

## 7. Evaluation And Plotting

Evaluation and plotting are independent from training. You can run them later on saved experiment outputs.

Evaluate an existing experiment:

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260409_000000_compare_default_models" \
  --device cuda \
  --snr_range "30:-3:0" \
  --tdl "A-30,B-100,C-300" \
  --num_batches 100 \
  --batch_size 2048
```

Plot later from an experiment or evaluation directory:

```bash
python ./Model_AIIC_refactor/plot.py \
  --input "./Model_AIIC_refactor/experiments_refactored/20260409_000000_compare_default_models"
```

`plot.py` accepts:

- an experiment directory
- an evaluation directory
- an `evaluation_results.json` file directly

## 8. ONNX Export

### 8.1 Export one checkpoint

```bash
python ./Model_AIIC_refactor/export_onnx.py \
  --checkpoint ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>/model.pth \
  --opset 13 \
  --dynamic_batch \
  --validate
```

You can also point to an intermediate checkpoint such as `checkpoint_batch_87000.pth`.

For the single-checkpoint CLI, the default output is written next to the selected checkpoint:

- `model.pth` -> `model.onnx`
- `checkpoint_batch_87000.pth` -> `checkpoint_batch_87000.onnx`

The matching manifest is written alongside it as:

- `model.export_manifest.json`
- `checkpoint_batch_87000.export_manifest.json`

### 8.2 ONNX output layout

Manual single-checkpoint export default:

```text
<run_dir>/
  model.pth
  model.onnx
  model.export_manifest.json
```

or:

```text
<run_dir>/
  checkpoint_batch_87000.pth
  checkpoint_batch_87000.onnx
  checkpoint_batch_87000.export_manifest.json
```

Post-training multi-run export still uses the per-run artifact directory:

```text
<run_dir>/onnx_exports/
  <run_name>.onnx
  export_manifest.json
```

`export_manifest.json` stores resolved model metadata, training metadata, tensor shapes, names, and validation results.

### 8.3 ONNX I/O contract

- input: `N x (2*seq_len)` real-stacked `single`
- output: `N x num_ports x (2*seq_len)` real-stacked `single`

When `model_spec.normalize_energy=true`, the ONNX graph already includes per-sample RMS normalization and output rescaling.

## 9. Matlab Bundle Export

### 9.1 Export one checkpoint

```bash
python ./Model_AIIC_refactor/export_matlab_bundle.py \
  --checkpoint ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>/model.pth
```

The exporter always stores one reference sample. That only affects the bundled `sample_input` and `reference_output`; Matlab inference still accepts arbitrary batch size `N`.

### 9.2 Matlab bundle output layout

```text
<run_dir>/matlab_exports/
  matlab_model_bundle.mat
  matlab_model_bundle_manifest.json
```

### 9.3 What the bundle contains

Always present:

- `sample_input`: `1 x (2*seq_len)`
- `reference_output`: `1 x num_ports x (2*seq_len)`
- `pos_values`

For `separator2`, the bundle also contains fully materialized effective MLP weights per port, stage, and layer:

- `p01_s01_l01_weight_real`
- `p01_s01_l01_weight_imag`
- `p01_s01_l01_bias_real`
- `p01_s01_l01_bias_imag`

For `separator1`, it contains separate real and imaginary branch weights:

- `p01_s01_real_l01_weight`
- `p01_s01_real_l01_bias`
- `p01_s01_imag_l01_weight`
- `p01_s01_imag_l01_bias`

Even when training used `share_weights_across_stages=True`, the exporter writes every effective port-stage block explicitly.

## 10. Matlab Integration

### 10.1 Recommended way to start

If you already have one exported artifact and just want to try it in Matlab, start from exactly one file or export directory and use only this entrypoint first:

- `matlab/run_refactor_model_demo.m`

That script is the recommended quick start.

You should edit only one variable in it:

- `exportPath`

Do not start from the lower-level run scripts unless you specifically need ONNX-only debugging or explicit separator1 traces.

### 10.2 What path can I pass into Matlab?

You can now point Matlab directly to the artifact you actually want to test.

Supported ONNX inputs:

- `<run_dir>/onnx_exports`
- `<run_dir>/checkpoint_batch_100000.onnx`
- `<run_dir>/checkpoint_batch_100000.export_manifest.json`
- `<run_dir>/model.onnx`
- `<run_dir>/model.export_manifest.json`

Supported Matlab bundle inputs:

- `<run_dir>/matlab_exports`
- `<run_dir>/matlab_model_bundle.mat`
- `<run_dir>/matlab_model_bundle_manifest.json`

This means that if you already know which `.onnx` or `.mat` file you want, you do not need to think in terms of “which run directory should I pass”. You can just pass that file directly.

### 10.3 Main Matlab API path

The main Matlab API path is:

- `matlab/import_refactor_model.m`
- `matlab/describe_refactor_model_io.m`
- `matlab/prepare_refactor_input.m`
- `matlab/predict_refactor_model.m`
- `matlab/demo_refactor_model_inference.m`
- `matlab/run_refactor_model_demo.m`

Recommended example with ONNX:

```matlab
[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(".../<run_name>/checkpoint_batch_100000.onnx", "auto", 8);
```

Recommended example with Matlab bundle:

```matlab
[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(".../<run_name>/matlab_model_bundle.mat", "auto", 8);
```

The third argument is always the Matlab-side runtime batch size used to generate test input. It is not tied to export-time batch settings.

### 10.4 What does Matlab use to determine input/output dimensions?

For normal repo exports, Matlab gets the I/O contract from the exported manifest that sits next to the `.onnx` or `.mat` artifact.

That metadata drives:

- input feature width
- output tensor width
- batch-dimension behavior
- input and output layout strings

You therefore do not need to manually construct widths like `24` or `48` in the common workflow. Use:

```matlab
modelHandle = import_refactor_model(exportPath, "auto");
[inputData, ioSpec] = prepare_refactor_input(modelHandle, 8, modelHandle.mode);
[outputData, debug, modelHandle] = predict_refactor_model(modelHandle, inputData, modelHandle.mode);
```

### 10.5 Lower-level bundle usage

```matlab
bundle = import_refactor_matlab_bundle(".../<run_name>/matlab_model_bundle.mat");
inputData = prepare_refactor_input(bundle, 8, "bundle");
[outputData, debug] = predict_refactor_matlab_bundle(bundle, inputData);
```

`prepare_refactor_input` generates `batchSize x (2*seq_len)` input automatically from the imported metadata.

### 10.6 ONNX-specific note

If the ONNX export used fixed batch size instead of dynamic batch, the helper will chunk or pad requests on the Matlab side as needed.

### 10.7 Which Matlab script should I use?

Use this mapping:

- `run_refactor_model_demo.m`: recommended quick start for almost everything
- `run_refactor_onnx_demo.m`: ONNX-only debugging when you know you only want the ONNX backend
- `run_refactor_matlab_bundle_demo.m`: bundle-only debugging when you know you only want explicit Matlab weights
- `run_refactor_separator1_demo.m`: advanced separator1 explicit layer-trace debugging

If you are unsure, use only `run_refactor_model_demo.m`.

### 10.8 Shape conventions in Matlab

- input shape: `N x (2*seq_len)`
- output shape: `N x num_ports x (2*seq_len)`
- real-stacked layout: `[real_part, imag_part]`

The printed shape spec uses `-1` for dynamic dimensions.

Examples:

- dynamic ONNX input: `[-1, 24]`
- dynamic ONNX output: `[-1, 6, 24]`
- bundle output: `[-1, 4, 24]`

## 11. Separator1 Explicit Matlab Notes

If the Matlab implementation team mainly cares about `separator1`, the explicit bundle path is the clearest reference.

`separator1` uses two ordinary real-valued MLP branches per port-stage block:

- one branch predicts the real part
- one branch predicts the imaginary part

Both branches take the same real-stacked input:

```text
input = [real_part, imag_part]
shape = N x (2*seq_len)
```

### 11.1 Separator1 field naming

- `p01_s01_real_l01_weight`
- `p01_s01_real_l01_bias`
- `p01_s01_imag_l01_weight`
- `p01_s01_imag_l01_bias`

Meaning:

- `p01`: port 1
- `s01`: stage 1
- `real` or `imag`: branch
- `l01`: layer 1 inside that branch MLP

### 11.2 Separator1 tensor shapes

- mixed input: `N x (2*seq_len)`
- one branch hidden layer: `N x hidden_dim`
- one branch final layer: `N x seq_len`
- one port output: `N x (2*seq_len)`
- one stage output: `N x num_ports x (2*seq_len)`

For the common 6-port setup in this repo:

- `seq_len = 12`
- input width = `24`
- output width per port = `24`

### 11.3 Separator1 forward structure

For one port in one stage:

```text
real_1 = ReLU(input * W_real_1^T + b_real_1)
real_2 = ReLU(real_1 * W_real_2^T + b_real_2)
real_out = real_2 * W_real_3^T + b_real_3

imag_1 = ReLU(input * W_imag_1^T + b_imag_1)
imag_2 = ReLU(imag_1 * W_imag_2^T + b_imag_2)
imag_out = imag_2 * W_imag_3^T + b_imag_3

port_output = [real_out, imag_out]
```

Residual refinement then applies:

```text
y_recon = sum(port_output over all ports)
residual = input_mixed - y_recon
refined_port_output = port_output + residual
```

### 11.4 Recommended Matlab files for separator1 review

- `matlab/import_refactor_matlab_bundle.m`
- `matlab/predict_refactor_separator1_bundle_explicit.m`
- `matlab/run_refactor_separator1_demo.m`

The explicit helper keeps the following loops visible:

- stage loop
- port loop
- layer loop
- branch split into real and imag

It also records detailed traces in:

- `debug.stage_outputs`
- `debug.stage_port_layer_traces`
- `debug.port_layer_outputs`

## 12. Checkpoint And Config Schema

### 12.1 Standard checkpoint structure

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_info': model.get_model_info(),
    'model_spec': {...},
    'training_spec': {...},
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': [...],
    'val_losses': [...],
    'loss_type': 'nmse',
    'metadata': {...},
    'eval_results': {...},
}
```

Expected schema for new code:

- `model_spec`
- `training_spec`
- `metadata`
- `model_state_dict`

### 12.2 Human-readable companion

Each run directory should contain:

```text
<run_dir>/
  model.pth
  config.yaml
  tensorboard/
```

`config.yaml` mirrors:

```yaml
model_spec:
  ...
training_spec:
  ...
metadata:
  ...
```

### 12.3 Load expectation

New evaluators and exporters load from `model_spec`.

```python
checkpoint = torch.load(model_path, map_location=device)
model_spec = checkpoint['model_spec']
model = create_model(model_name=model_spec['model_type'], config=model_spec)
```

If a historical checkpoint does not contain `model_spec`, treat it as legacy and rely on the compatibility loader in the utilities layer.

## 13. Benchmark Entry Points

```bash
python ./Model_AIIC_refactor/compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python ./Model_AIIC_refactor/compare_optimizations.py --experiment perf_quick --skip_gpu
```

## 14. Policy

- The old `model_config + training_config` CLI pairing is intentionally removed for training.
- `experiments.yaml` is the supported workflow interface for training and benchmark launches.
- For manual export, the project standardizes on single-checkpoint export CLIs.
- This file is the only maintained help-style guide for `Model_AIIC_refactor`.
