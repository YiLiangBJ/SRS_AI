# Standalone Eval And Plot

## Summary

Evaluation and plotting are fully independent from training.

Training is launched experiment-first with `train.py`, while evaluation and plotting can be run later on saved experiment outputs.

## Supported Workflow

### Step 1: Train a named experiment

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda
```

This creates a timestamped experiment directory under `experiments_refactored/`.

### Step 2: Evaluate later

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260409_000000_compare_default_models" \
  --device cuda \
  --snr_range "30:-3:0" \
  --tdl "A-30,B-100,C-300" \
  --num_batches 100 \
  --batch_size 2048
```

This writes one evaluation run under:

```text
single run: Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>/evaluations/<timestamp>/
multiple runs: Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/evaluations/<timestamp>_<scope>/
```

so different SNR / TDL / run selections do not overwrite each other.

### Step 3: Plot later

```bash
python ./Model_AIIC_refactor/plot.py \
  --input "./Model_AIIC_refactor/experiments_refactored/20260409_000000_compare_default_models"
```

`plot.py` now accepts any of these as `--input`:

- experiment directory: automatically picks the latest evaluation run
- evaluation directory: reads its `evaluation_results.json`
- `evaluation_results.json` directly

By default plots are written to `<evaluation_dir>/plots`.

## ONNX Export Later

You can also export ONNX after training:

```bash
python ./Model_AIIC_refactor/export_onnx.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260409_000000_compare_default_models" \
  --runs separator2_default_hd64_stages3_depth3 \
  --dynamic_batch \
  --validate
```

If you migrated an old multi-run evaluation into per-run folders and want each copied JSON to keep only one run, use:

```bash
python ./Model_AIIC_refactor/split_evaluation_results.py \
  --input "./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>/evaluations/<eval_dir>" \
  --run_name "<run_name>"
```

## Directory Shape

```text
experiments_refactored/
    <timestamp>_<experiment_name>/
        <run_name>/
            model.pth
            config.yaml
            tensorboard/
      evaluations/
        <timestamp>/
          evaluation_results.json
          evaluation_results.npy
          plots/
            ...png
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
          ...png
    TRAINING_REPORT.md
```

## Notes

- Training, evaluation, export, and plotting each have a thin CLI entrypoint.
- The orchestration logic lives in `workflows/`, so scripts stay lightweight.
- Evaluation discovers trained runs inside an experiment directory.
- Plotting works on one evaluation run directory and is decoupled from training.
- You can re-run evaluation with different SNR, TDL, or run selections as many times as you want without overwriting previous results.
- You can re-export ONNX with different dynamic-axis or validation options without retraining.

## Benchmark Presets

```bash
python ./Model_AIIC_refactor/compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python ./Model_AIIC_refactor/compare_optimizations.py --experiment perf_quick --skip_gpu
```

## Policy

The old `model_config + training_config` training CLI is removed. `experiments.yaml` is the supported interface for training and benchmark launches.
