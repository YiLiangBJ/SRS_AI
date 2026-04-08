# Config Guide

## Recommended Split

- model_configs.yaml: only model architecture, port layout, and model-side search spaces.
- training_configs.yaml: only optimization, data sampling, stopping, logging, and training-side search spaces.
- train.py: builds the Cartesian product of model variants and training variants into one experiment plan.

## Supported Patterns

- Single config:

```yaml
separator1_default:
  model_type: separator1
  pos_values: [0, 3, 6, 9]
  hidden_dim: 64
  num_stages: 3
```

- Fixed params plus search space:

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

## Practical Conventions

- Keep ports and seq_len on the model side. They define what the model is.
- Keep SNR, TDL, loss, LR, batch size, validation, and checkpointing on the training side. They define how the model is trained.
- If a field is part of a deliberate sweep, put it in search_space. If it is just a constant, put it in fixed_params or leave it flat for single configs.
- Prefer one search dimension per question. Example: compare losses in one training config, compare LR in another, instead of one very wide sweep.
- Use train.py --plan_only to inspect the final run matrix before launching long jobs.

## Why This Layout Works

- It keeps architecture decisions and training decisions decoupled.
- It makes run names deterministic and readable.
- It lets the code cache each side's expansion once, then combine them into a single plan.