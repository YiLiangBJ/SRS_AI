# Config Guide

## Recommended Split

- `model_configs.yaml`: model architecture, port layout, ONNX mode, and model-side search spaces
- `training_configs.yaml`: optimization, data sampling, stopping, logging, and training-side search spaces
- `experiments.yaml`: reusable experiment presets that bind model recipes to one training recipe
- `train.py`: thin CLI that dispatches into the workflow layer
- `workflows/train_workflow.py`: builds the Cartesian product of model variants and training variants into an executable experiment plan

## Supported Patterns

### Single config

```yaml
separator1_default:
  model_type: separator1
  pos_values: [0, 3, 6, 9]
  hidden_dim: 64
  num_stages: 3
```

### Fixed params plus search space

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

### Experiment preset

```yaml
experiments:
  compare_default_models:
    model_configs:
      - separator1_default
      - separator2_default
    training_config: snr_range_0_30_perSample
```

## Practical Conventions

- Keep `pos_values`, `seq_len`, architecture width/depth, and ONNX-mode compatibility on the model side.
- Keep SNR policy, TDL policy, loss, LR, batch size, validation cadence, and checkpoint cadence on the training side.
- Put workflow intent in `experiments.yaml`: smoke tests, architecture comparisons, benchmarks, paper sweeps, export candidates.
- If a field is part of a deliberate sweep, put it in `search_space`.
- If it is just a constant, keep it flat or place it in `fixed_params`.
- Prefer one search dimension per scientific question instead of one very wide sweep.
- Use `python ./Model_AIIC_refactor/train.py --experiment <name> --plan_only` to inspect the resolved run matrix before launching long jobs.

## Naming

- recipe name: the YAML entry name, such as `separator1_default` or `quick_test`
- label: the resolved variant name after search-space expansion
- run name: the final executable training instance name

Use recipe names for traceability back to config files.

Use labels when comparing variants inside one recipe.

Use run names for filesystem paths, exports, reports, and result aggregation.

## Workflow Architecture

The current refactored project uses a lightweight workflow architecture:

- thin CLIs:
  - `train.py`
  - `evaluate_models_refactored.py`
  - `export_onnx.py`
  - `plot.py`
- shared workflow modules:
  - `workflows/train_workflow.py`
  - `workflows/postprocess_workflow.py`
  - `workflows/evaluation_workflow.py`
  - `workflows/export_workflow.py`
  - `workflows/plotting_workflow.py`
  - `workflows/reporting.py`

This is useful for research because:

- command-line usage stays simple
- notebook scripts and benchmark tools can call workflow APIs directly
- experiment metadata, checkpoints, evaluation, and ONNX exports share one artifact schema
- future changes to training logic happen once in the workflow layer

## Why This Layout Works

- It keeps architecture decisions and training decisions decoupled.
- It makes run names deterministic and readable.
- It lets the code expand each side once, then combine them into one plan.
- It keeps the project scalable without turning it into a heavy framework.

## Migration Rule

Backward-compatible aliases are intentionally removed in new code.

Prefer these terms consistently:

- `experiment`
- `model_recipe_name`
- `training_recipe_name`
- `model_label`
- `training_label`
- `run_name`
- `model_spec`
- `training_spec`
