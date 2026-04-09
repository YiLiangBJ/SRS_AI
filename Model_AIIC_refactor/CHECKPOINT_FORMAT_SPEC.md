# Checkpoint Format Spec

## Principles

1. A checkpoint must contain everything needed to rebuild the model and understand the run.
2. New code uses `model_spec`, `training_spec`, `metadata`, and `eval_results`.
3. The refactored project is experiment-first and stores run lineage explicitly.

## Standard Format

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_info': model.get_model_info(),

    'model_spec': {
        'model_type': 'separator1',
        'hidden_dim': 64,
        'num_stages': 3,
        'mlp_depth': 3,
        'share_weights_across_stages': False,
        'activation_type': 'relu',
        'onnx_mode': False,
        'seq_len': 12,
        'num_ports': 4,
        'pos_values': [0, 3, 6, 9],
        'num_params': 156032,
    },

    'training_spec': {
        'loss_type': 'nmse',
        'learning_rate': 0.01,
        'num_batches': 10000,
        'batch_size': 4096,
        'snr_config': {'type': 'range', 'min': 0, 'max': 30},
        'tdl_config': 'A-30',
        'print_interval': 100,
        'validation_interval': None,
        'early_stop_loss': None,
        'patience': 3,
        'keep_last_n_checkpoints': 2,
        'save_interval': None,
    },

    'optimizer_state_dict': optimizer.state_dict(),
    'losses': [...],
    'val_losses': [...],
    'loss_type': 'nmse',

    'metadata': {
        'experiment_name': 'compare_default_models',
        'model_recipe_name': 'separator1_default',
        'model_label': 'separator1_default_hd64_stages3_depth3',
        'run_name': 'separator1_default_hd64_stages3_depth3',
        'training_recipe_name': 'snr_range_0_30_perSample',
        'training_label': 'snr_range_0_30_perSample',
        'training_duration': 1234.5,
        'timestamp': '2026-04-09T00:00:00',
    },

    'eval_results': {
        'nmse': 0.001234,
        'nmse_db': -29.08,
        'per_port_nmse_db': [-30.1, -28.5, -29.2, -28.8],
    },
}
```

## Human-Readable Companion

Each trained run directory should contain:

```text
<run_dir>/
    model.pth
    config.yaml
    tensorboard/
```

`config.yaml` mirrors the structured checkpoint metadata:

```yaml
model_spec:
  ...
training_spec:
  ...
metadata:
  ...
```

## Naming Rules

Use these terms consistently:

- `experiment_name`: named workflow from `experiments.yaml`
- `model_recipe_name`: model entry from `model_configs.yaml`
- `training_recipe_name`: training entry from `training_configs.yaml`
- `model_label`: expanded model variant name
- `training_label`: expanded training variant name
- `run_name`: final executable run identifier

Do not introduce legacy names such as `config`, `training_config`, `model_config`, or `config_instance_name` in new code.

## Load Expectations

New evaluators, exporters, and workflow helpers load from `model_spec`.

```python
checkpoint = torch.load(model_path, map_location=device)
model_spec = checkpoint['model_spec']
model = create_model(model_name=model_spec['model_type'], config=model_spec)
```

If a checkpoint does not contain `model_spec`, treat it as obsolete and retrain with the current refactored workflow.

## Validation Checklist

```python
assert 'model_spec' in checkpoint
assert 'training_spec' in checkpoint
assert 'metadata' in checkpoint
assert 'model_state_dict' in checkpoint

assert 'model_type' in checkpoint['model_spec']
assert 'pos_values' in checkpoint['model_spec']
assert 'num_ports' in checkpoint['model_spec']
assert len(checkpoint['model_spec']['pos_values']) == checkpoint['model_spec']['num_ports']
```

## Workflow Note

The refactored project now uses thin CLI entrypoints and shared workflow modules. Checkpoint production and loading should go through the workflow/utilities layer rather than script-local ad hoc logic.
