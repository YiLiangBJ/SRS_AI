# Standalone Eval And Plot

## Summary

Evaluation and plotting are fully independent from training.

Training is launched experiment-first with `train.py`, while evaluation and plotting can be run later on saved experiment outputs.

## Supported Workflow

### Step 1: Train a named experiment

```bash
python train.py \
  --experiment compare_default_models \
  --device cuda
```

This creates a timestamped experiment directory under `experiments_refactored/`.

### Step 2: Evaluate later

```bash
python evaluate_models_refactored.py \
  --exp_dir "./experiments_refactored/20260409_000000_compare_default_models" \
  --device cuda \
  --snr_range "30:-3:0" \
  --tdl "A-30,B-100,C-300" \
  --num_batches 100 \
  --batch_size 2048 \
  --output "./experiments_refactored/20260409_000000_compare_default_models/evaluation_results"
```

### Step 3: Plot later

```bash
python plot.py \
  --input "./experiments_refactored/20260409_000000_compare_default_models/evaluation_results/evaluation_results.json" \
  --output "./experiments_refactored/20260409_000000_compare_default_models/plots"
```

## ONNX Export Later

You can also export ONNX after training:

```bash
python export_onnx.py \
  --exp_dir "./experiments_refactored/20260409_000000_compare_default_models" \
  --runs separator2_default_hd64_stages3_depth3 \
  --output "./experiments_refactored/20260409_000000_compare_default_models/onnx_exports" \
  --dynamic_batch \
  --validate
```

## Directory Shape

```text
experiments_refactored/
    <timestamp>_<experiment_name>/
        <run_name>/
            model.pth
            config.yaml
            tensorboard/
        evaluation_results/
            evaluation_results.json
            evaluation_results.npy
        onnx_exports/
            <run_name>/
                <run_name>.onnx
                export_manifest.json
        plots/
            ...png
        TRAINING_REPORT.md
```

## Notes

- Training, evaluation, export, and plotting each have a thin CLI entrypoint.
- The orchestration logic lives in `workflows/`, so scripts stay lightweight.
- Evaluation discovers trained runs inside an experiment directory.
- Plotting works on `evaluation_results.json` and is decoupled from training.
- You can re-run evaluation with different SNR or TDL settings as many times as you want.
- You can re-export ONNX with different dynamic-axis or validation options without retraining.

## Benchmark Presets

```bash
python compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python compare_optimizations.py --experiment perf_quick --skip_gpu
```

## Policy

The old `model_config + training_config` training CLI is removed. `experiments.yaml` is the supported interface for training and benchmark launches.
