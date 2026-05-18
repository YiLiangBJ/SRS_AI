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

You do not manually pair model config and training config on the CLI. You launch a named experiment from `configs/v2/experiments.yaml`, and the workflow resolves:

1. one task recipe
2. one or more model recipes
3. one training-strategy recipe
4. local and experiment-level sweep expansions
5. the final executable run plan

## 2. Important Terms

- `experiment`: a named workflow preset from `configs/v2/experiments.yaml`
- `task recipe`: one entry from `configs/v2/tasks.yaml`
- `model recipe`: one entry from `configs/v2/models.yaml`
- `training strategy recipe`: one entry from `configs/v2/training_strategies.yaml`
- `task label`: the expanded task variant name after sweep resolution
- `model label`: the expanded model variant name after sweep resolution
- `training label`: the expanded training-strategy variant name after sweep resolution
- `run_name`: the final unique executable run identifier
- `component_specs`: the raw task/model/training_strategy payloads saved with the run
- `model_spec`: the task-compiled runtime model schema saved with the run
- `training_spec`: the compiled runtime training schema saved with the run

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

- `configs/v2/tasks.yaml`: data-generation policy, port layout, sequence length, normalization flag, SNR sampling, and TDL selection
- `configs/v2/tasks.yaml`: optional `sampling_rate` can be set to drive high-resolution TDL generation before KTC decimation and `seq_len` block extraction, for example `122880000.0` for `4096 * 30 kHz`
- `configs/v2/models.yaml`: architecture family plus model-side hyperparameter sweeps
- `configs/v2/training_strategies.yaml`: optimizer, loss, validation cadence, scheduler, and checkpoint policy
- `configs/v2/experiments.yaml`: reusable workflow presets binding task + model + training strategy

### 4.2 Supported config patterns

Task recipe:

```yaml
tasks:
  channel_separator_6port_standard:
    type: channel_separator
    params:
      seq_len: 12
      pos_values: [0, 2, 4, 6, 8, 10]
      snr_config:
        type: range
        min: 0
        max: 30
        per_sample: true
        sampling: stratified
        num_bins: 10
      tdl_config: [A-30, B-100, C-300]
      sampling_rate: 122880000.0
```

Model recipe with sweeps:

```yaml
models:
  full_mlp_capacity_search:
    type: full_mlp
    params:
      normalize_energy: true
      hidden_dim: 128
      mlp_depth: 3
    sweeps:
      - target: params.hidden_dim
        alias: hd
        values: [64, 128, 256, 512]
      - target: params.mlp_depth
        alias: depth
        values: [2, 3, 4, 5]
```

Training-strategy recipe:

```yaml
training_strategies:
  supervised_nmse_plateau:
    type: standard_supervised
    params:
      batch_size: 4096
      num_batches: 100000
      optimizer:
        type: adam
        params:
          learning_rate: 0.01
      loss:
        type: nmse
      validation:
        interval: 100
        batches: 4
      early_stop:
        patience: 5
```

Experiment preset:

```yaml
experiments:
  compare_default_models_v2:
    task: channel_separator_4port_standard
    model: [separator1_default, separator2_default]
    training_strategy: supervised_log_plateau
```

### 4.3 Practical conventions

- Keep `seq_len`, `pos_values`, `snr_config`, and `tdl_config` on the task side.
- Keep width/depth/stage count, activation options, `normalize_energy`, and model-family-specific architecture flags on the model side.
- Keep optimizer, loss, validation cadence, scheduler policy, and checkpoint cadence on the training-strategy side.
- Put workflow intent in `configs/v2/experiments.yaml`: smoke tests, architecture comparisons, export candidates, and sweeps.
- If a field is a deliberate scientific sweep, put it in `sweeps`.
- If a model needs task-owned values such as `seq_len` or `pos_values`, let the task adapter inject them through runtime `model_spec` compilation instead of duplicating them in the model recipe.
- Prefer narrow sweeps aligned to one question instead of one large unfocused Cartesian product.

### 4.4 Inspect plans before launch

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment full_mlp_capacity_search_v2 \
  --plan_only \
  --device cpu
```

### 4.5 Built-in experiment presets

- `quick_full_mlp_v2`: one-run 6-port smoke test for the joint full-MLP baseline
- `quick_full_mlp_masked_v2`: one-run 6-port smoke test for the joint full-MLP baseline with masked residual correction
- `quick_full_mlp_two_stage_v2`: one-run 6-port smoke test for staged full-MLP training
- `quick_full_mlp_three_stage_v2`: one-run 6-port smoke test for `nmse -> log -> weighted` staged full-MLP training
- `full_mlp_nmse_v2`: one-run 6-port full-MLP baseline with plain NMSE loss
- `full_mlp_arch_search_v2`: 9-run 6-port width/depth search for full-MLP
- `full_mlp_capacity_search_v2`: default 20-run 6-port hidden-dim/depth search for full-MLP
- `full_mlp_capacity_search_masked_v2`: default 20-run 6-port hidden-dim/depth search for full-MLP with masked residual correction
- `quick_separator1_v2`: one-run 6-port smoke test for separator1
- `quick_separator1_masked_v2`: one-run 6-port smoke test for separator1 with masked residual correction
- `quick_separator3_v2`: one-run 6-port smoke test for multi-stage separator3 with learned dense residual correction
- `default_6port_separator3_learned_dense_v2`: 32-run standard separator3 sweep over hidden_dim, depth, and stage count with learned dense residual correction
- `quick_separator3_stage_templates_v1`: four-run quick smoke test for hand-designed separator3 stage-hidden templates
- `separator3_stage_templates_learned_dense_v1`: four-run standard comparison of hand-designed separator3 stage-hidden templates
- `compare_default_models_v2`: compare full_mlp_default, separator1_default, and separator2_default on the same 6-port task
- `default_6port_separator1_v2`: default 20-run 6-port separator1 sweep over depth, stage count, weight sharing, and hidden dim for depth-3 variants
- `default_6port_separator1_masked_v2`: default 20-run 6-port separator1 sweep with masked residual correction
- `default_6port_separator3_v2`: default 6-port multi-stage separator3 training run with learned dense residual correction
- `separator1_loss_search_v2`: compare supervised loss choices for 6-port separator1_default

## 5. Training

### 5.1 Common commands

Train the default 6-port full-MLP hyperparameter scan:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment full_mlp_capacity_search_v2 \
  --device cuda
```

Train one quick 6-port full-MLP smoke test:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_v2 \
  --device cuda
```

Train one quick 6-port full-MLP smoke test with masked residual correction:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_masked_v2 \
  --device cuda
```

Train one quick 6-port full-MLP smoke test with learned dense residual correction:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_learned_dense_v2 \
  --device cuda
```

Inspect the default 6-port full-MLP search without launching it:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment full_mlp_capacity_search_v2 \
  --plan_only \
  --device cpu
```

Current depth semantics are intentionally aligned across `full_mlp` and `separator1`:

- `mlp_depth=2` means two Linear mappings
- for `full_mlp`, that is `input -> hidden_dim -> output`
- for `separator1`, that is `input -> hidden_dim -> seq_len` inside each real/imag branch of one stage

So `full_mlp` with `mlp_depth=2` still uses `hidden_dim`; it is not a direct `input -> output` model.

Inspect only one resolved run from a larger experiment:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator1_learned_dense_v2 \
  --runs separator1_grid_search_6ports_learned_dense_depth2_stages2_share0 \
  --plan_only \
  --device cpu
```

Inspect one resolved run with temporary overrides applied after experiment resolution:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator1_learned_dense_v2 \
  --runs separator1_grid_search_6ports_learned_dense_depth2_stages2_share0 \
  --model_override mlp_depth=2 \
  --model_override num_stages=2 \
  --training_override batch_size=16 \
  --plan_only \
  --device cpu
```

Inspect an entire resolved experiment after applying experiment-wide overrides through the universal `--override` entrypoint. The override is applied after sweep resolution, duplicate plans that collapse to the same final task/model/training specs are removed automatically, and generated run names are suffixed with override tokens so the saved directories remain distinguishable:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator3_learned_dense_v2 \
  --override model.num_stages=1 \
  --plan_only \
  --device cpu
```

For training, the universal `--override` scopes are:

- `task.<path>=value`
- `model.<path>=value`
- `training.<path>=value`

If you need to replace a component recipe's local sweep range before plan expansion, use:

- `task.sweeps.<alias>.values=...`
- `model.sweeps.<alias>.values=...`
- `training.sweeps.<alias>.values=...`

Here `<alias>` is the sweep alias from YAML such as `depth`, `hd`, or `stages`. The override is applied before sweep expansion, so it changes the generated search space itself rather than merely collapsing already-expanded runs.

Example:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator3_v2 \
  --override task.seq_len=16 \
  --override model.num_stages=1 \
  --override training.batch_size=8 \
  --plan_only \
  --device cpu
```

Example: replace the separator3 depth sweep from `[2,3]` to `[2,3,4,5,6]` for `default_6port_separator3_learned_dense_v2`:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator3_learned_dense_v2 \
  --override model.sweeps.depth.values=[2,3,4,5,6] \
  --device cuda
```

If you want to inspect the generated run plan first:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator3_learned_dense_v2 \
  --override model.sweeps.depth.values=[2,3,4,5,6] \
  --plan_only \
  --device cpu
```

Train, then evaluate and plot a 3-model 6-port comparison:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models_v2 \
  --device cuda \
  --eval_after_train \
  --plot_after_eval
```

Quick CPU benchmark-style 6-port run with batch-count override:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1_v2 \
  --num_batches 100 \
  --device cpu
```

Quick CPU smoke test for the masked-residual separator1 variant:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1_masked_v2 \
  --device cpu
```

Quick CPU smoke test for the learned-dense-residual separator1 variant:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1_learned_dense_v2 \
  --device cpu
```

Quick CPU smoke test for multi-stage separator3:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator3_v2 \
  --device cpu
```

Standard 6-port separator3 sweep over hidden_dim, depth, and stage count:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment default_6port_separator3_learned_dense_v2 \
  --device cpu
```

Quick separator3 stage-template smoke test:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator3_stage_templates_v1 \
  --device cpu
```

Standard separator3 stage-template comparison:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment separator3_stage_templates_learned_dense_v1 \
  --device cpu
```

`separator3` is now a staged joint MLP:

- stage 1 maps `2*seq_len -> hidden_dim -> ... -> num_ports * 2 * seq_len`
- later stages map `num_ports * 2 * seq_len -> hidden_dim -> ... -> num_ports * 2 * seq_len`
- every stage ends with configurable residual correction
- `stage_hidden_dims` can override the hidden width per stage, for example `[128, 64, 64]`

`separator3` supports these residual-correction modes:

- `global`: add the full residual back to every port
- `masked`: add back only the residual taps selected by `pos_values`
- `learned_dense`: learn one static dense residual mask per stage
- `generated_dense`: generate a per-sample dense residual mask from the current stage features and residual while keeping the separator3 backbone unchanged

Recommended starter templates for `stage_hidden_dims`:

- `[128, 64]`
- `[128, 64, 64]`
- `[128, 128, 64]`
- `[64, 64, 64]`

Recommended VS Code debug flow for separator3:

- train all hand-designed templates with `Python: train.py (separator3 stage templates)`
- debug one explicit template with `Python: train single separator3 [128,64,64] debug`
- evaluate one saved run with `Python: evaluate separator3 run`
- benchmark one saved run with `Python: benchmark separator3 run`
- export one saved checkpoint with `Python: export_onnx (separator3)` or `Python: export_matlab_bundle (separator3)`

Resume one model from a previous checkpoint but train with new training parameters:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_v2 \
  --init_checkpoint ./Model_AIIC_refactor/experiments_refactored/<old_experiment>/<run_name>/model.pth \
  --num_batches 200 \
  --device cuda
```

Quick staged-training smoke test:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_two_stage_v2 \
  --num_batches 20 \
  --device cuda
```

Quick three-stage training smoke test:

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_full_mlp_three_stage_v2 \
  --num_batches 20 \
  --device cuda
```

### 5.2 Train CLI summary

| Argument | Meaning |
|---|---|
| `--experiment` | Required experiment name from `configs/v2/experiments.yaml` |
| `--batch_size` | Optional override applied after recipe resolution |
| `--num_batches` | Optional override applied after recipe resolution |
| `--runs` | Keep only selected run names from the resolved experiment plan |
| `--override` | Preferred universal override entrypoint; training supports `task.*`, `model.*`, and `training.*` |
| `--task_override` | Deprecated compatibility alias for task-scoped training override in `key=value` form |
| `--model_override` | Deprecated compatibility alias for model-scoped training override; duplicate collapsed plans are removed automatically |
| `--training_override` | Deprecated compatibility alias for training-scoped override; override tokens are appended to generated run names |
| `--init_checkpoint` | Initialize weights from one existing checkpoint; model spec must match exactly |
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
- when the model recipe sets `normalize_energy=true`, `separator1`, `separator2`, `separator3`, and `full_mlp` all apply per-sample RMS normalization at model input and restore the same RMS on model output.
- that normalization rule is preserved consistently in Python inference, ONNX export, and Matlab bundle inference.
- every trained run now writes `MODEL_FLOW.md` and `model_flow.json`, showing human-readable node-by-node tensor shapes with dynamic dimensions written as `-1`.
- `MODEL_FLOW.md` now also includes parameter counts per learned node and a plain-language explanation of why each node has that shape.
- if `--init_checkpoint` is provided, the training workflow loads exactly one checkpoint and checks that the current compiled `model_spec` matches the checkpoint `model_spec` field-by-field before training starts.
- `multi_stage_supervised` runs several supervised stages sequentially on the same model weights; each stage can use different loss, learning rate, batch count, and other training settings.

### 5.4 Single-Inference Dimension Examples

The examples below use one concrete 6-port configuration:

- `seq_len = 12`
- `num_ports = 6`
- real-stacked input layout `[real_part, imag_part]`
- batch size `N = 8`

That means the mixed input width is always `2 * seq_len = 24`.

Example A: `full_mlp_default`

- input mixed signal: `8 x 24`
- internal RMS normalization: still `8 x 24`
- first linear layer with `hidden_dim=128`: `8 x 128`
- hidden ReLU output: `8 x 128`
- final linear layer to `num_ports * 2 * seq_len = 144`: `8 x 144`
- reshape to separated channels: `8 x 6 x 24`
- output RMS restoration: still `8 x 6 x 24`

Example B: `separator1_default`

- input mixed signal: `8 x 24`
- internal RMS normalization: still `8 x 24`
- initial replicated per-port features: `8 x 6 x 24`
- one port entering one stage: `8 x 24`
- real branch hidden layer with `hidden_dim=64`: `8 x 64`
- imag branch hidden layer with `hidden_dim=64`: `8 x 64`
- real branch output layer: `8 x 12`
- imag branch output layer: `8 x 12`
- concatenate one port output: `8 x 24`
- stack all ports after one stage: `8 x 6 x 24`
- residual correction: still `8 x 6 x 24`
- final output RMS restoration: `8 x 6 x 24`

Example C: `separator2_default`

- input mixed signal: `8 x 24`
- internal RMS normalization: still `8 x 24`
- initial replicated per-port features: `8 x 6 x 24`
- one port entering one stage: `8 x 24`
- first complex-hidden affine block with `hidden_dim=64`: real and imag parts become `8 x 64` each, stored as one real-stacked tensor `8 x 128`
- hidden activation output: still `8 x 128`
- final complex output affine block: real and imag parts become `8 x 12` each, stored as one real-stacked tensor `8 x 24`
- stack all ports after one stage: `8 x 6 x 24`
- residual correction: still `8 x 6 x 24`
- final output RMS restoration: `8 x 6 x 24`

If you switch to complex input form for `separator1` or `full_mlp`, the user-facing input can be `8 x 12` complex and the final output can be `8 x 6 x 12` complex, but internally both models still convert to the same real-stacked width `24` before most of the learned layers run.

## 6. Artifact Layout

One experiment directory can contain many concrete runs.

- the experiment directory groups one launch of one named experiment preset
- each immediate child run directory is one fully resolved task/model/training combination
- the run directory name is `run_name`, and that name is where model/training/task sweep choices are encoded
- `TRAINING_REPORT.md` summarizes all runs produced under that experiment directory

So the current structure does represent hyperparameter combinations, but it represents them as multiple sibling run directories under one experiment directory, not as extra nested folders under one run.

```text
Model_AIIC_refactor/
  experiments_refactored/
    <timestamp>_<experiment_name>/
      TRAINING_REPORT.md
      <run_name>/
        model.pth
        config.yaml
        MODEL_FLOW.md
        model_flow.json
        tensorboard/
          events.out.tfevents...
          loss_curves.jpg
          nmse_curves.jpg
          learning_rate.jpg
          throughput.jpg
          snr.jpg
        stage_artifacts/
          <stage_name>/
            stage_summary.json
        evaluations/
          <timestamp>/
            evaluation_results.json
            evaluation_results.npy
            EVALUATION_SUMMARY.md
            plots/
        onnx_exports/
          <run_name>.onnx
          export_manifest.json
          MODEL_FLOW.md
          model_flow.json
        matlab_exports/
          matlab_model_bundle.mat
          matlab_model_bundle_manifest.json
          MODEL_FLOW.md
          model_flow.json
        latency/
          <timestamp>_<device>/
            latency_results.json
            latency_samples.npz
            hardware_manifest.json
            LATENCY_REPORT.md
            plots/
    evaluations/
      <timestamp>_<scope>/
        evaluation_results.json
        evaluation_results.npy
        EVALUATION_SUMMARY.md
        plots/
```

      The `tensorboard/` directory is now useful even if you never launch TensorBoard itself:

      - `loss_curves.jpg`: train loss and validation loss when available
      - `nmse_curves.jpg`: train and validation NMSE in dB when available
      - `learning_rate.jpg`: learning-rate schedule over training
      - `throughput.jpg`: sampled throughput over training
      - `snr.jpg`: sampled training SNR over batches

      The `latency/` directory is a standalone benchmark artifact root:

      - it is not part of train, eval, or plot postprocessing
      - single-run benchmark results live under each run directory
      - experiment-wide comparison results also live under the experiment directory
      - reports include latency statistics together with references to the run's complexity and flow artifacts

Example for a multi-run experiment:

```text
Model_AIIC_refactor/
  experiments_refactored/
    20260421_130000_full_mlp_capacity_search_v2/
      TRAINING_REPORT.md
      full_mlp_capacity_search_hd64_depth2/
        model.pth
        config.yaml
        ...
      full_mlp_capacity_search_hd64_depth3/
        model.pth
        config.yaml
        ...
      full_mlp_capacity_search_hd128_depth2/
        model.pth
        config.yaml
        ...
      ... 13 more run directories ...
```

In other words:

- experiment level: one training launch, one report, one shared container
- run level: one concrete hyperparameter combination, one checkpoint set, one config, one flow description, and one export location

Even when an experiment expands to only one concrete run, the layout still remains:

```text
<timestamp>_<experiment_name>/
  TRAINING_REPORT.md
  <run_name>/
    model.pth
    config.yaml
    MODEL_FLOW.md
    ...
```

So single-run and multi-run experiments use the same `experiment -> run` directory shape.

If you later want a stronger visual separation, an alternative would be an extra level such as `task_label/model_label/training_label`, but that is not how the current workflow stores artifacts.

## 7. Evaluation And Plotting

Evaluation and plotting are independent from training. You can run them later on saved experiment outputs.

Current evaluation behavior follows the same two-level structure as training:

- if you evaluate one run directory, results are saved only under that run directory
- if you evaluate an experiment directory, every run gets its own evaluation result under its own run directory
- in that experiment case, the experiment directory also gets one aggregate evaluation summary for cross-run comparison
- per-run evaluation artifacts do not overwrite the experiment-level comparison summary
- every evaluation directory now also writes `EVALUATION_SUMMARY.md` for human-readable ranking and quick inspection

Evaluate an existing experiment:

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260421_000000_compare_default_models_v2" \
  --device cuda \
  --snr_range "30:-3:0" \
  --tdl "A-30,B-100,C-300" \
  --num_batches 100 \
  --batch_size 2048
```

Evaluation also supports the same universal `--override` entrypoint for command parameters. This keeps the mental model aligned with training and benchmarking.

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260421_000000_compare_default_models_v2" \
  --override batch_size=4096 \
  --override num_batches=50 \
  --override plot_after_eval=false
```

Plot later from an experiment or evaluation directory:

```bash
python ./Model_AIIC_refactor/plot.py \
  --input "./Model_AIIC_refactor/experiments_refactored/20260421_000000_compare_default_models_v2"
```

`plot.py` accepts:

- an experiment directory
- an evaluation directory
- an `evaluation_results.json` file directly

Current plotting behavior mirrors evaluation:

- if the input is one run directory, `plot.py` finds the latest evaluation under that run and writes plots into that run evaluation's `plots/`
- if the input is one experiment directory, `plot.py` finds the latest aggregate experiment evaluation and also the latest evaluation under each run
- in that experiment case, each run keeps its own model-specific curves under its own run evaluation directory
- the experiment evaluation directory also gets comparison plots spanning multiple runs
- plot legends are placed outside the plotting area as a single column and shrink their font size when needed so dense multi-line comparisons stay readable without stretching the curve area vertically

Example after evaluating an experiment with two runs:

```text
<experiment_dir>/
  TRAINING_REPORT.md
  separator1_default/
    evaluations/
      20260421_124428/
        evaluation_results.json
        evaluation_results.npy
        EVALUATION_SUMMARY.md
        plots/
          nmse_vs_snr_TDL_A_30.png
          nmse_vs_snr_combined.png
  separator2_default/
    evaluations/
      20260421_124428/
        evaluation_results.json
        evaluation_results.npy
        EVALUATION_SUMMARY.md
        plots/
          nmse_vs_snr_TDL_A_30.png
          nmse_vs_snr_combined.png
  evaluations/
    20260421_124428_separator1_default_separator2_default/
      evaluation_results.json
      evaluation_results.npy
      EVALUATION_SUMMARY.md
      plots/
        nmse_vs_snr_TDL_A_30.png
        nmse_vs_snr_combined.png
```

`TRAINING_REPORT.md` now also lists `task`, `model`, and `training` labels per run instead of forcing you to infer everything only from `run_name`.
For multi-stage training, `TRAINING_REPORT.md` also includes a per-stage summary block.

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
  MODEL_FLOW.md
  model_flow.json
```

`export_manifest.json` stores resolved model metadata, training metadata, tensor shapes, names, validation results, and the same model-flow description that is also written as `MODEL_FLOW.md` and `model_flow.json` beside the exported artifact.

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

That reference pair is not just for convenience. It is the formal Python-to-Matlab parity anchor:

- `sample_input` is generated in Python during export
- `reference_output` is generated in Python by running the trained PyTorch model on that same `sample_input`

Before a Matlab bundle is treated as deployment-ready, you should run the Matlab code path on that exact exported `sample_input` and compare the result against `reference_output`.

### 9.2 Matlab bundle output layout

```text
<run_dir>/matlab_exports/
  matlab_model_bundle.mat
  matlab_model_bundle_manifest.json
  MODEL_FLOW.md
  model_flow.json
```

### 9.3 What the bundle contains

Always present:

- `sample_input`: `1 x (2*seq_len)`
- `reference_output`: `1 x num_ports x (2*seq_len)`
- `pos_values`

Supported bundle model types:

- `separator1`
- `separator2`
- `separator3`
- `full_mlp`

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

For `separator3`, it contains every stage's joint MLP weights and the learned-dense residual mask for that stage:

- `stage01_joint_l01_weight`
- `stage01_joint_l01_bias`
- `stage01_joint_l02_weight`
- `stage01_joint_l02_bias`
- `stage01_residual_mask`
- `stage02_joint_l01_weight`
- `stage02_joint_l01_bias`
- `stage02_residual_mask`

For `generated_dense`, the checkpoint stores the joint MLP weights plus the dynamic mask-generator weights for each stage instead of a directly trained static residual mask parameter.

If `use_hidden_layer_norm=true`, hidden layers also include per-branch LayerNorm parameters:

- `p01_s01_real_l01_ln_weight`
- `p01_s01_real_l01_ln_bias`
- `p01_s01_real_l01_ln_eps`
- `p01_s01_imag_l01_ln_weight`
- `p01_s01_imag_l01_ln_bias`
- `p01_s01_imag_l01_ln_eps`

Even when training used `share_weights_across_stages=True`, the exporter writes every effective port-stage block explicitly.

For `full_mlp`, the bundle contains the single joint network weights in execution order:

- `joint_l01_weight`
- `joint_l01_bias`
- `joint_l02_weight`
- `joint_l02_bias`

Matlab bundle inference applies the same per-sample RMS input normalization and output rescaling rule as the Python model when `model_spec.normalize_energy=true`.
The Matlab bundle manifest also embeds the same model-flow description that is saved as `MODEL_FLOW.md` and `model_flow.json` in the bundle directory, so the artifact can be copied to another machine and inspected without the original training workspace.

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

### 10.5.1 Off-the-shelf Matlab component package

Every Matlab bundle export now also creates a versioned component package under:

- `.../<run_name>/matlab_exports/matlab_component/v1_<timestamp>/`

That versioned folder is designed to be copyable into another Matlab project as a self-contained deployment component. It includes:

- `matlab_model_bundle.mat`
- `matlab_model_bundle_manifest.json`
- the required runtime helpers such as `import_refactor_matlab_bundle.m` and `predict_refactor_matlab_bundle.m`
- short deployment-first entrypoints:
  - `init_model.m`
  - `predict_model.m`
  - `split_ports.m`
- short model-specific aliases:
  - `init_<short_tag>.m`
  - `predict_<short_tag>.m`
- optional lower-level helpers:
  - `load_srs_ai_matlab_component.m`
  - `predict_srs_ai_matlab_component.m`
- `demo/` subfolder containing:
  - `demo_quick_start.m`
  - `demo_step_by_step.m`
  - `demo_sim_platform_loop.m`
  - `README_DEMO.md`

Recommended deployment usage after copying that versioned folder to your Matlab project:

```matlab
state = init_model();
outputData = predict_model(state, randn(8, 24, 'single'));
ports = split_ports(outputData);
```

This is intentionally split so you can do file loading and manifest parsing only once, then reuse `state` for every subsequent slot.

For a 6-port separator model, this means:

- input: `N x 24`
- output: `N x 6 x 24`
- `ports{k}`: `N x 24` for the `k`-th port

This component-package path is the recommended Matlab handoff format when a model is ready for downstream integration.

### 10.5.2 Deliver subfolder for minimal handoff

Each versioned Matlab component package now also includes a dedicated minimal handoff folder:

- `.../<run_name>/matlab_exports/matlab_component/v1_<timestamp>/deliver/`

This `deliver/` folder is the one to copy when you want the smallest deployment-ready set. It contains:

- `matlab_model_bundle.mat`
- `matlab_model_bundle_manifest.json`
- the minimum runtime helpers needed by the deployed API
- short deployment entrypoints:
  - `init_model.m`
  - `predict_model.m`
  - `split_ports.m`
- short model-specific aliases:
  - `init_<short_tag>.m`
  - `predict_<short_tag>.m`
- a single deployment demo:
  - `demo_deliver_two_call.m`

Recommended deployed usage from `deliver/`:

```matlab
state = init_model();                % first slot / one-time init
outputData = predict_model(state, x); % later slots reuse state
ports = split_ports(outputData);
```

The `demo_deliver_two_call.m` script demonstrates exactly this pattern:

- first call initializes and caches `state`
- second call checks that `state` already exists and skips reloading the bundle
- final reference check runs the Matlab code path on Python-generated `sample_input` and compares to Python-generated `reference_output`

This parity check should be treated as the required deployment gate between:

1. Matlab bundle export
2. Matlab-side integration / deployment handoff

If you are integrating into a slot-based Matlab simulation platform, prefer copying `deliver/` rather than the larger parent component folder.

Recommended first-use order inside the copied component package:

1. Run `demo/demo_quick_start.m`.
2. Then open `demo/demo_step_by_step.m` and execute it section by section in the Matlab editor.
3. When integrating into your simulation platform, copy the pattern from `demo/demo_sim_platform_loop.m`.
4. If you want a short model-specific API, use `init_<short_tag>.m` and `predict_<short_tag>.m`.

### 10.6 ONNX-specific note

The Matlab helper now uses a version-compatible ONNX import path:

- it first tries `importNetworkFromONNX` without `OutputLayerType`
- if needed, it falls back to `importONNXNetwork`

If the ONNX export used fixed batch size instead of dynamic batch, the helper will chunk or pad requests on the Matlab side as needed.

### 10.7 Which Matlab script should I use?

Use this mapping:

- `run_refactor_model_demo.m`: recommended quick start for almost everything
- `run_refactor_onnx_demo.m`: ONNX-only debugging when you know you only want the ONNX backend
- `run_refactor_matlab_bundle_demo.m`: bundle-only debugging when you know you only want explicit Matlab weights
- `run_refactor_separator1_demo.m`: advanced separator1 explicit layer-trace debugging

If you are unsure, use only `run_refactor_model_demo.m`.

If you are handing a model to another Matlab project and want the smallest copyable deployment unit, prefer the versioned `matlab_component/v1_<timestamp>/` package instead of the raw `matlab_exports/` directory.

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

If hidden LayerNorm is enabled, the same hidden layer also has:

- `p01_s01_real_l01_ln_weight`
- `p01_s01_real_l01_ln_bias`
- `p01_s01_real_l01_ln_eps`
- `p01_s01_imag_l01_ln_weight`
- `p01_s01_imag_l01_ln_bias`
- `p01_s01_imag_l01_ln_eps`

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

For one hidden layer in one port and one stage, the current refactor implementation is:

```text
real_hidden = input * W_real^T + b_real
real_hidden = LayerNorm(real_hidden)        # if use_hidden_layer_norm=true
real_hidden = ReLU(real_hidden)             # if use_hidden_relu=true

imag_hidden = input * W_imag^T + b_imag
imag_hidden = LayerNorm(imag_hidden)        # if use_hidden_layer_norm=true
imag_hidden = ReLU(imag_hidden)             # if use_hidden_relu=true
```

The final output layer is linear only:

```text
real_out = real_hidden * W_real_out^T + b_real_out
imag_out = imag_hidden * W_imag_out^T + b_imag_out

port_output = [real_out, imag_out]
```

### 11.3A Full-MLP residual options

`full_mlp` keeps its original behavior by default with `model_spec.residual_correction_mode = none`.

It also supports `model_spec.residual_correction_mode = masked` after the joint output is reshaped to `N x num_ports x (2*seq_len)`:

- compute `y_recon = sum(port_output over all ports)`
- compute `residual = input_mixed - y_recon`
- for each branch tied to `pos_values = k`, add back only residual indices `k` and `k + seq_len`

It also supports `model_spec.residual_correction_mode = learned_dense`:

- compute `y_recon = sum(port_output over all ports)`
- compute `residual = input_mixed - y_recon`
- learn one dense residual-mixing weight per `(port, tap)`
- apply `refined_port_output = port_output + residual * learned_mask[port, :]`

For `learned_dense`, the residual is no longer restricted to a fixed tap pair from `pos_values`; every port receives a trainable dense weighting over all real/imag taps.

Training now applies a light default regularization to `learned_dense` masks, pulling weights gently toward `1.0` unless the training strategy explicitly overrides `regularization.learned_dense_mask_l2_to_one`.

For current v2 bundles, `model_spec.use_hidden_layer_norm` tells you whether these LayerNorm parameters are expected to exist in the exported bundle.

Residual refinement then applies:

```text
y_recon = sum(port_output over all ports)
residual = input_mixed - y_recon
refined_port_output = port_output + residual
```

Separator1 also supports an optional masked residual mode through `model_spec.residual_correction_mode`:

- `global` (default): broadcast the full residual back to every port
- `masked`: add back only the real/imag tap pair selected by each branch's `pos_values` entry
- `learned_dense`: add back a trainable dense residual weighting per port

For `masked`, no extra tap parameter is needed. The branch tied to `pos_values=k` receives residual only at indices `k` and `k + seq_len`.

For `learned_dense`, separator1 reuses the existing `share_weights_across_stages` rule:

- if `share_weights_across_stages = true`, one dense residual mask is shared by all refinement stages
- if `share_weights_across_stages = false`, each refinement stage learns its own dense residual mask

The same light default regularization toward `1.0` applies during training, so dense masks start from and are weakly biased toward full residual passthrough instead of collapsing immediately to arbitrary values.

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

## 12. Checkpoint And Config Schema

### 12.1 Standard checkpoint structure

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_info': model.get_model_info(),
    'model_spec': {...},
    'training_spec': {...},
    'component_specs': {
        'task': {...},
        'model': {...},
        'training_strategy': {...},
    },
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': [...],
    'val_losses': [...],
    'loss_type': 'nmse',
    'best_val_loss': float | None,
    'best_val_nmse_db': float | None,
    'best_val_batch': int | None,
    'metadata': {...},
    'eval_results': {...},
}
```

  When validation runs during training, `model.pth` now stores the best validation weights restored at the end of training, not simply the final-step weights. Final evaluation in the training workflow uses these same restored best-validation weights.

Expected schema for new code:

- `model_spec`
- `training_spec`
- `metadata`
- `component_specs`
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
component_specs:
  task:
    ...
  model:
    ...
  training_strategy:
    ...
```

### 12.3 Load expectation

New evaluators and exporters load from `model_spec` plus `component_specs`.

```python
checkpoint = torch.load(model_path, map_location=device)
model_spec = checkpoint['model_spec']
component_specs = checkpoint['component_specs']
model = create_model(model_name=model_spec['model_type'], config=model_spec)
```

Current refactor loaders expect `model_spec`, `training_spec`, `metadata`, and `component_specs` to be present. Old pre-v2 checkpoints are not the supported workflow path for evaluation/export in this guide.

### 12.4 Continue training from one checkpoint

The supported resume workflow is intentionally simple:

- you pass exactly one checkpoint file with `--init_checkpoint`
- the current planned run may use different training parameters
- the current planned run must use the same model structure as the checkpoint
- before training starts, the workflow compares the compiled current `model_spec` against the checkpoint `model_spec`
- if anything differs, training stops and the mismatch is reported field-by-field

This makes it safe to do things such as:

- keep the same full-MLP architecture but change learning rate or loss
- keep the same separator1 architecture but switch to a staged training schedule

It does not allow silent architecture drift.

### 12.5 Multi-stage training strategy schema

The current built-in staged strategy type is `multi_stage_supervised`.

Example:

```yaml
training_strategies:
  quick_two_stage_supervised:
    type: multi_stage_supervised
    params:
      stages:
        - name: warmup_nmse
          params:
            batch_size: 32
            num_batches: 60
            optimizer:
              type: adam
              params:
                learning_rate: 0.01
            loss:
              type: nmse
        - name: finetune_log
          params:
            batch_size: 32
            num_batches: 40
            optimizer:
              type: adam
              params:
                learning_rate: 0.003
            loss:
              type: log
```

Runtime behavior:

- one model instance is created
- stage 1 trains it with the first stage spec
- stage 2 continues from the stage 1 weights
- stage 3, if present, continues from the stage 2 weights, and so on
- per-stage checkpoint scratch artifacts live under `stage_artifacts/`
- final `model.pth`, `config.yaml`, `MODEL_FLOW.md`, and exports still live at the run root as usual

## 13. Benchmark Entry Points

### 13.1 Standalone Latency Benchmark

Latency benchmarking is a separate task from training, evaluation, and plotting.

Supported first-version benchmark dimensions:

- device: CPU-first by default; CUDA interface is available but not the current default focus
- execution mode: CPU defaults to `eager`, `jit`, and `compile`; CUDA currently defaults to `eager`
- precision profile: `fp32`, plus device-specific lower-precision profiles when supported
- batch size: default `1,2,4,8,16,32,64,128`
- CPU threads / cores: default `1,2,4,8`

For large batch studies, `benchmark_latency.py` also supports generating batch sizes from antenna-count and RBG-count products:

- `--batch_antennas 8,16,32,64`
- `--batch_rbgs 1,2,4,8,17,34,68`

When both are provided, the CLI expands `antenna_count * rbg_count` products, unions them with any explicit `--batch_sizes`, de-duplicates them, and benchmarks the resulting ascending batch list.

Current default behavior is intentionally CPU-centric:

- `benchmark_latency.py` defaults to `--device cpu`
- `benchmark_latency.py` defaults to `--runtime_backends pytorch`
- `benchmark_latency.py` defaults to CPU execution modes `eager,jit,compile`
- CPU is the primary path for current validation and regression coverage
- CUDA interface is kept available for later expansion, but it is not the first-version default benchmark path

The base `uv sync` environment now includes both `onnxruntime` and `openvino` for CPU deployment benchmarking.

For a complete deployment-oriented setup and reproduction guide covering `uv` environment management, `onnxruntime`, `openvino`, install commands, and focused CPU benchmark recipes, see:

- `Model_AIIC_refactor/CPU_DEPLOYMENT_BENCHMARK_GUIDE.md`

Example: benchmark one run on CPU across the default batch and thread profiles:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu
```

The benchmark CLI also supports the same universal `--override` entrypoint for its own command parameters.

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --override thread_counts=1,2,4 \
  --override measure_iters=100 \
  --override batch_sizes=1,2,4,8,16,32,64,128
```

Example: benchmark one run on CPU with only TorchScript JIT mode:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --execution_modes jit
```

Example: benchmark one run on CPU across eager, JIT, and compile modes:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --execution_modes eager,jit,compile
```

Example: benchmark one run on CPU with ONNX Runtime as a deployment-oriented backend:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends onnxruntime \
  --precision_profiles fp32
```

Notes for ONNX Runtime backend:

- current first-version support is CPU only
- current first-version support is `fp32` only
- the benchmark reuses or creates `run_dir/onnx_exports/export_manifest.json` and the matching `.onnx` export as needed
- results are reported with `runtime_backend=onnxruntime` and `execution_mode=onnxruntime`

Example: benchmark one run on CPU with OpenVINO as a deployment-oriented backend:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends openvino \
  --precision_profiles fp32
```

Notes for OpenVINO backend:

- current first-version support is CPU only
- current first-version support is `fp32` only
- the benchmark reuses or creates `run_dir/onnx_exports/export_manifest.json` and the matching `.onnx` export as needed
- this is a standalone OpenVINO runtime path, not ONNX Runtime with OpenVINO Execution Provider
- results are reported with `runtime_backend=openvino` and `execution_mode=openvino`

Example: benchmark one run on CPU across PyTorch, ONNX Runtime, and OpenVINO in one pass:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends pytorch,onnxruntime,openvino
```

Notes for the combined backend command:

- PyTorch runs the CPU default precision set: `fp32,bf16`
- ONNX Runtime expands only its valid `fp32` combinations during task generation
- OpenVINO expands only its valid `fp32` combinations during task generation
- combinations that are structurally valid but unavailable in the current environment, such as a missing backend package, are still reported as skipped in the latency results rather than crashing the whole benchmark

Example: benchmark a whole experiment on CUDA with selected precision profiles:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>" \
  --device cuda \
  --precision_profiles fp32,fp16,bf16
```

Example: benchmark one run on CPU with an explicit wider batch sweep:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --batch_sizes 1,2,4,8,16,32,64,128
```

Example: benchmark one run on CPU with a batch sweep generated from antenna count times RBG count, while keeping the legacy small-batch anchors:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends onnxruntime \
  --precision_profiles fp32 \
  --thread_counts 1 \
  --batch_sizes 1,2,4,8,16,32,64,128 \
  --batch_antennas 8,16,32,64 \
  --batch_rbgs 1,2,4,8,17,34,68
```

This command is a good first pass for the specific deployment question where batch size represents `antenna_count * rbg_count`, with:

- antenna count in `{8,16,32,64}`
- RBG count up to `68`
- maximum batch size `64 * 68 = 4352`

The generated batch set covers:

- the existing low-batch latency anchors `1..128`
- intermediate deployment-relevant products such as `136,256,272,512,544,1088,2176`
- the full upper bound `4352`

Example: run the same large-batch study over a whole experiment directory instead of a single run:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>" \
  --device cpu \
  --runtime_backends onnxruntime \
  --precision_profiles fp32 \
  --thread_counts 1 \
  --batch_sizes 1,2,4,8,16,32,64,128 \
  --batch_antennas 8,16,32,64 \
  --batch_rbgs 1,2,4,8,17,34,68
```

Example: after identifying representative large-batch points, run a focused thread-scaling follow-up instead of sweeping threads over every batch size:

```bash
python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends onnxruntime \
  --precision_profiles fp32 \
  --thread_counts 1,2,4,8 \
  --batch_sizes 128,256,512,1088,2176,4352
```

Recommended evaluation workflow for the `antenna_count * rbg_count` deployment case:

1. Start with `--thread_counts 1` and the generated batch grid above.
2. Plot or inspect latency growth and throughput gain from `128` up to `4352`.
3. Check whether throughput begins to flatten, which indicates diminishing amortization of fixed overhead.
4. Only after that, run a second pass with selected large batch sizes and `1,2,4,8` threads to see whether higher thread counts help throughput enough to justify the latency penalty.

The benchmark writes run-local results under:

```text
<run_dir>/latency/<timestamp>_<device>/
```

and experiment-level aggregate comparisons under:

```text
<experiment_dir>/latency/<timestamp>_<scope>_<device>/
```

Each latency directory contains:

- `latency_results.json`: structured summary with device, precision, batch, threads, percentile latencies, throughput, and skip reasons
- `latency_results.json`: also records `execution_mode` and graph preparation time per benchmark configuration
- `latency_results.csv`: flattened table for Excel / CSV workflows; one row per benchmark configuration
- `latency_samples.npz`: raw latency samples for each measured configuration
- `hardware_manifest.json`: captured environment and hardware information
- `LATENCY_REPORT.md`: human-readable summary for implementation teams
- `plots/`: static latency and throughput plots

The CSV export is generated automatically during every new latency benchmark run.

Important CSV columns include:

- benchmark dimensions: run, runtime backend, execution mode, precision, batch size, threads
- latency stats: mean, std, min, p50, p90, p95, p99, max
- throughput stats: `throughput_samples_per_sec`, `samples_per_ms`, `throughput_per_thread`
- per-sample view: `p50_latency_us`, `latency_per_sample_us`
- model complexity: `trainable_parameters`, `macs_per_sample`, `flops_per_sample_estimate`
- thread grouping helpers: `thread_group`, `threads_per_physical_core_ratio`, `threads_per_logical_cpu_ratio`
- hardware context: CPU model, CPU capability, mkldnn/oneDNN fields, Python and PyTorch versions

For CPU benchmarks, `LATENCY_REPORT.md` now also includes thread-scaling highlights per run:

- best throughput configuration: the thread count and batch size that reached the highest measured samples/s
- lowest batch-1 p50 latency: the thread count that minimized batch-1 p50 latency

CPU plot outputs now include both run-local and aggregate-friendly views:

- run-local: `p50_latency_vs_batch.jpg`, `throughput_vs_batch.jpg`, `p50_latency_vs_threads.jpg`
- experiment aggregate: `bs1_p50_comparison.jpg`
- experiment aggregate at fixed threads: `p50_latency_vs_batch_threads_<N>.jpg`, `throughput_vs_batch_threads_<N>.jpg`
- multi-series latency plots and re-plots now use a larger qualitative palette, more line styles, and adaptive legend-aware sizing that keeps the legend in a single outside column while shrinking legend font size instead of stretching the plot area

If you want a detailed conceptual explanation of how `eager`, `jit`, and `compile` differ on CPU with oneDNN underneath, see:

- `Model_AIIC_refactor/ONEDNN_CPU_BENCHMARK_TUTORIAL.md`

### 13.2 Backfill CSV For Existing Latency Results

If you already have historical latency directories containing only `latency_results.json`, you can generate `latency_results.csv` later without rerunning the benchmark.

Export one latency directory or one `latency_results.json` file:

```bash
python ./Model_AIIC_refactor/export_latency_csv.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu"
```

Or export one file to an explicit output path:

```bash
python ./Model_AIIC_refactor/export_latency_csv.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu/latency_results.json" \
  --output "./somewhere/latency_table.csv"
```

Recursively backfill every `latency_results.json` under one directory tree:

```bash
python ./Model_AIIC_refactor/export_latency_csv.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>" \
  --recursive
```

### 13.3 Re-Plot Saved Latency Results

You can generate new comparison figures later from an existing latency directory without rerunning the benchmark.

Use:

```bash
python ./Model_AIIC_refactor/plot_latency_benchmark.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu"
```

The input can be either:

- a latency directory containing `latency_results.json`
- a `latency_results.json` file directly

The script generates two subplot-oriented comparison views for each selected thread count:

- `mode_panels_<metric>_threads_<N>.jpg`
  each subplot fixes one execution mode, and compares different models / precision profiles inside that mode
- `model_precision_panels_<metric>_threads_<N>.jpg`
  each subplot fixes one model + precision combination, and compares different execution modes inside that combination

All plot entrypoints in the workflow now keep legends outside the plotting area in a single column and adapt legend font size when many lines are present.

Example: draw throughput instead of p50 latency:

```bash
python ./Model_AIIC_refactor/plot_latency_benchmark.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu" \
  --metric throughput_samples_per_sec
```

Example: only draw a subset of execution modes and one thread count:

```bash
python ./Model_AIIC_refactor/plot_latency_benchmark.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu" \
  --execution_modes eager,jit \
  --thread_counts 1
```

Example: only draw a subset of runs and precision profiles:

```bash
python ./Model_AIIC_refactor/plot_latency_benchmark.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<experiment>/latency/<timestamp>_<scope>_cpu" \
  --runs run_a,run_b \
  --precision_profiles fp32,bf16
```

The new plots are written under a separate `plots_custom/` directory beside the input latency results so they do not overwrite the benchmark's default plots.

### 13.4 Training Perf Utilities

```bash
python ./Model_AIIC_refactor/compare_cpu_gpu.py --experiment quick_separator1_v2 --skip_gpu
python ./Model_AIIC_refactor/compare_optimizations.py --experiment quick_separator1_v2 --skip_gpu
```

## 14. Policy

- The old `model_config + training_config` CLI pairing is intentionally removed for training.
- `configs/v2/experiments.yaml` is the supported workflow interface for training and benchmark launches.
- Task/model/training_strategy components are the supported source of truth for new runs.
- `full_mlp`, `separator1`, and `separator2` all share the same internal normalize-input / restore-output energy contract when `normalize_energy=true`.
- model-flow descriptions are first-class artifacts and are written both into training run directories and export directories.
- experiment-level evaluation always keeps run-local evaluation results separate from the aggregate comparison summary.
- For manual export, the project standardizes on single-checkpoint export CLIs.
- This file is the only maintained help-style guide for `Model_AIIC_refactor`.
