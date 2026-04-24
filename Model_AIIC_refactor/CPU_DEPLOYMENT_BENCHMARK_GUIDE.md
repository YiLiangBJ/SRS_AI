# CPU Deployment Benchmark Guide

## Purpose

This guide documents the current CPU deployment-oriented benchmark paths implemented in `Model_AIIC_refactor`.

It is intended to be complete enough that a fresh machine can be prepared and used to reproduce the following CPU benchmark backends:

- `pytorch` with `jit`
- `onnxruntime`
- `openvino`

The focus is deployment-style latency benchmarking, not training-time validation.

## What Is Implemented

The latency benchmark entrypoint is:

- `Model_AIIC_refactor/benchmark_latency.py`

The current runtime backends are:

- `pytorch`
- `onnxruntime`
- `openvino`

Current first-version scope:

- CPU only for `onnxruntime`
- CPU only for `openvino`
- `fp32` only for `onnxruntime`
- `fp32` only for `openvino`
- `pytorch` keeps the existing CPU modes `eager`, `jit`, `compile`
- deployment-oriented comparisons so far primarily use `pytorch + jit`

## Backend Meaning

### `pytorch`

- Native PyTorch execution path.
- For CPU latency work, the most relevant mode is usually `jit`.
- Best used as a development/reference baseline.

### `onnxruntime`

- ONNX Runtime as the execution engine.
- Current implementation uses `CPUExecutionProvider`.
- This is the preferred low-batch deployment reference path based on the benchmark results gathered so far.

### `openvino`

- Standalone OpenVINO runtime path.
- This is **not** ONNX Runtime with an OpenVINO Execution Provider.
- Current implementation reads the exported ONNX model and compiles it directly with OpenVINO for CPU.
- Based on current results, this path is mainly interesting for larger models at larger batch sizes.

## Environment Setup With `uv`

This repository already uses `uv` to manage the Python environment.

### Recommended fresh-machine setup

From the repository root:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync
```

Notes:

- `uv sync` installs the dependencies declared in `pyproject.toml`.
- `onnxruntime` is already declared in the project dependencies, so it should be available after `uv sync`.
- `openvino` is not currently part of the base project dependencies and must be installed separately when needed.

### Install OpenVINO with `uv`

For the current environment, prefer `uv pip` instead of plain `pip`:

```bash
uv pip install openvino
```

This avoids relying on a standalone `pip` binary and works well with a `uv`-managed environment.

### If ONNX Runtime is missing

Normally `uv sync` should install it. If not, install it explicitly with:

```bash
uv pip install onnxruntime
```

### Verify installed runtime packages

```bash
uv run python - <<'PY'
import importlib.util
import importlib.metadata

for name in ['onnxruntime', 'openvino']:
    installed = importlib.util.find_spec(name) is not None
    print(name, 'installed=', installed)
    if installed:
        print(name, 'version=', importlib.metadata.version(name))
PY
```

## How The Benchmark Uses ONNX

Both deployment-oriented backends reuse ONNX export artifacts.

The benchmark will reuse or create:

- `<run_dir>/onnx_exports/export_manifest.json`
- `<run_dir>/onnx_exports/<run_name>.onnx`

This means:

- you do not need to pre-export ONNX manually for the default workflow
- the benchmark can automatically create missing ONNX artifacts when needed

## Core Commands

All commands below assume you are in the repository root and are using `uv run`.

### 1. PyTorch JIT CPU benchmark

```bash
uv run python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends pytorch \
  --execution_modes jit \
  --precision_profiles fp32
```

### 2. ONNX Runtime CPU benchmark

```bash
uv run python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends onnxruntime \
  --precision_profiles fp32
```

### 3. OpenVINO CPU benchmark

```bash
uv run python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dir "./Model_AIIC_refactor/experiments_refactored/<experiment>/<run_name>" \
  --device cpu \
  --runtime_backends openvino \
  --precision_profiles fp32
```

### 4. Side-by-side low-batch deployment comparison

This reproduces the focused comparison between PyTorch JIT and ONNX Runtime:

```bash
uv run python ./Model_AIIC_refactor/benchmark_latency.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2" \
  --device cpu \
  --runtime_backends pytorch,onnxruntime \
  --execution_modes jit \
  --precision_profiles fp32 \
  --batch_sizes 1,128 \
  --thread_counts 1 \
  --warmup_iters 3 \
  --measure_iters 5
```

### 5. Focused three-backend comparison on representative models

```bash
uv run python ./Model_AIIC_refactor/benchmark_latency.py \
  --run_dirs "./Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd32_depth2,./Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/full_mlp_capacity_search_hd512_depth5" \
  --device cpu \
  --runtime_backends pytorch,onnxruntime,openvino \
  --execution_modes jit \
  --precision_profiles fp32 \
  --batch_sizes 1,2,4,8,16,32,64,128 \
  --thread_counts 1 \
  --warmup_iters 3 \
  --measure_iters 5
```

## Result Output Layout

Single-run outputs are written to:

```text
<run_dir>/latency/<timestamp>_<device>/
```

Aggregate outputs are written to:

```text
<experiment_dir>/latency/<timestamp>_<scope>_<device>/
```

Each latency directory contains:

- `latency_results.json`
- `latency_results.csv`
- `latency_samples.npz`
- `hardware_manifest.json`
- `LATENCY_REPORT.md`
- `plots/`

Important fields now include:

- `runtime_backend`
- `execution_mode`
- `requested_precision_profile`
- `graph_prep_time_ms`
- `p50_latency_ms`
- `throughput_samples_per_sec`
- `onnxruntime_version`
- `onnxruntime_providers`
- `openvino_version`
- `openvino_device`

## Current Engineering Interpretation

The currently gathered CPU benchmark results support the following practical guidance.

### Low batch deployment

- Prioritize `onnxruntime`.
- Especially true for `batch=1` and other low-batch settings.

### Small models

- `onnxruntime` is currently the strongest choice across the tested batch range.
- `openvino` has not shown evidence of being worth defaulting to for small models.

### Larger models and larger batches

- `openvino` becomes more interesting as batch size increases.
- In the current focused tests, `openvino` begins to catch up around batch `8` and can clearly win from about batch `16` onward on larger models.

### PyTorch JIT

- Useful as a familiar baseline and development reference.
- Not the best deployment-latency path in the current CPU comparisons.

## Existing Result Notes

Relevant written summaries generated during this work:

- `Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/latency/20260424_113800_all-runs_2runs_89a030ee88_cpu/CPU_DEPLOYMENT_BACKEND_GUIDANCE.md`
- `Model_AIIC_refactor/experiments_refactored/20260422_050408_full_mlp_capacity_search_v2/latency/20260424_113800_all-runs_2runs_89a030ee88_cpu/CPU_DEPLOYMENT_BACKEND_GUIDANCE_ZH.md`

These summarize the currently observed backend-selection pattern for representative small and large models.

## Limitations

Current implementation limits to remember:

- `onnxruntime` backend: CPU only, `fp32` only
- `openvino` backend: CPU only, `fp32` only
- no GPU deployment-oriented benchmark path yet for these runtime backends
- current deployment comparisons are strongest for single-thread CPU studies; further multi-thread evaluation should be done explicitly when needed

## Recommended Reproduction Strategy On Another Machine

If you want the fastest path to reproduce the deployment conclusions on another machine:

1. Create the environment with `uv venv` and `uv sync`.
2. Install `openvino` with `uv pip install openvino`.
3. Verify `onnxruntime` and `openvino` imports with `uv run python`.
4. First run the focused two-backend comparison: `pytorch + jit` vs `onnxruntime`.
5. Then run the small-model / large-model focused three-backend comparison.
6. Only expand to more models or more threads after the small focused results are stable.

This keeps the reproduction effort low while preserving the most important deployment decisions.