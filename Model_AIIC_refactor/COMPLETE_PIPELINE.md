# Complete Pipeline

This project now uses an experiment-first workflow.

## Core Idea

You do not launch training by manually pairing model and training configs on the CLI.
You launch a named experiment from experiments.yaml, and the code resolves:

1. model recipes
2. training recipe
3. expanded variants
4. final run plan

## Common Commands

### Preview a plan

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1 \
  --plan_only \
  --device cpu
```

### Train a named experiment

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda
```

### Train, evaluate, and plot

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda \
  --eval_after_train \
  --plot_after_eval
```

### Override batch count for a benchmark preset

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment perf_quick \
  --num_batches 100 \
  --device cpu
```

## Output Layout

```text
Model_AIIC_refactor/
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
      TRAINING_REPORT.md
```

## Runtime Terms

- experiment: a workflow preset from experiments.yaml
- model recipe: one model entry from model_configs.yaml
- training recipe: one training entry from training_configs.yaml
- model label: expanded model variant name
- training label: expanded training variant name
- run_name: final unique executable run identifier

## Train CLI

| Argument | Meaning |
|---|---|
| --experiment | Required experiment name from experiments.yaml |
| --batch_size | Optional override applied after recipe resolution |
| --num_batches | Optional override applied after recipe resolution |
| --device | auto, cpu, cuda, cuda:0, ... |
| --save_dir | Parent output directory |
| --no-amp | Disable mixed precision |
| --no-compile | Disable torch.compile |
| --eval_after_train | Run evaluation after training |
| --eval_snr_range | SNR setting for evaluation, supports range or comma-separated values |
| --eval_tdl | TDL list for evaluation |
| --eval_num_batches | Number of evaluation batches |
| --eval_batch_size | Evaluation batch size |
| --plot_after_eval | Generate plots after evaluation |
| --export_onnx_after_train | Export trained runs to ONNX after training |
| --onnx_export_selection | Export `best` or `all` runs |
| --onnx_output_dir | Single-run ONNX artifact directory override |
| --onnx_opset | ONNX opset version |
| --onnx_batch_size | Dummy batch size used for tracing |
| --onnx_dynamic_batch | Export a dynamic batch dimension |
| --onnx_validate | Run ONNX checker and ORT smoke validation |
| --export_matlab_after_train | Export trained runs to Matlab explicit-weight bundles after training |
| --matlab_export_selection | Export `best` or `all` runs as Matlab bundles |
| --matlab_output_dir | Single-run Matlab bundle directory override |
| --plan_only | Print the run plan and exit |

## Benchmark CLI

Benchmark scripts also use experiment-first entry points:

```bash
python ./Model_AIIC_refactor/compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python ./Model_AIIC_refactor/compare_optimizations.py --experiment perf_quick --skip_gpu
```

## Typical Flow

1. Define recipes in model_configs.yaml and training_configs.yaml.
2. Define a workflow preset in experiments.yaml.
3. Preview with --plan_only.
4. Train the experiment.
5. Optionally evaluate, plot, and export ONNX artifacts.

## Example Summary Output

```text
Training Summary

Total runs trained: 1
Start time: 2026-04-08 14:00:00
End time: 2026-04-08 14:05:30
Total duration: 0.09 hours (330.0s)

1. separator1_default_hd64_stages2_depth3:
   Final loss: 0.123456
   Min loss: 0.123456
   Eval NMSE: -5.23 dB
   Parameters: 156,032
   Duration: 330.0s

Best run: separator1_default_hd64_stages2_depth3
```

## Policy

The old model_config plus training_config CLI pairing is intentionally removed. experiments.yaml is the supported workflow entry point.

## ONNX Export for Matlab

### Export the best trained run after training

```bash
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
  --device cuda \
  --export_onnx_after_train \
  --onnx_export_selection best \
  --onnx_validate
```

### Export runs later from an experiment directory

```bash
python ./Model_AIIC_refactor/export_onnx.py \
  --exp_dir Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name> \
  --runs separator2_default_hd64_stages3_depth3 \
  --opset 13 \
  --dynamic_batch \
  --validate
```

### Export output layout

```text
<run_dir>/onnx_exports/
  <run_name>.onnx
  export_manifest.json
```

`export_manifest.json` stores the resolved model spec, training metadata, input/output names, dummy tensor shapes, and validation summary. This is the handoff file for Matlab or deployment tooling.

### Matlab import helper

Use `matlab/import_refactor_onnx.m`:

```matlab
net = import_refactor_onnx(".../<run_name>/onnx_exports");
```

For a complete import plus inference example, use `matlab/demo_refactor_onnx_inference.m`:

```matlab
[net, inputData, outputData, manifest] = demo_refactor_onnx_inference(".../<run_name>/onnx_exports", 2);
```

For a script-style demo, edit and run `matlab/run_refactor_onnx_demo.m`.

For one unified Matlab entry that works for both ONNX and non-ONNX bundle imports, use `matlab/import_refactor_model.m`, `matlab/demo_refactor_model_inference.m`, and `matlab/run_refactor_model_demo.m`.

### Direct Matlab bundle export without ONNX

If the Matlab side needs explicit matrices and activations rather than an ONNX graph, export a Matlab bundle from an existing trained run:

```bash
python ./Model_AIIC_refactor/export_matlab_bundle.py \
  --run_dir ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>
```

This writes:

```text
<run_dir>/matlab_exports/
  matlab_model_bundle.mat
  matlab_model_bundle_manifest.json
```

In Matlab, use:

```matlab
[modelHandle, inputData, outputData, info] = demo_refactor_model_inference(".../<run_name>/matlab_exports", "bundle", 2);
```

For a script-style demo, edit and run `matlab/run_refactor_model_demo.m` or `matlab/run_refactor_matlab_bundle_demo.m`.

For the full handoff guide, including how to start from a run directory when you only know where the model is stored, see `matlab/README.md`.

The exported ONNX model uses:

- input: `N x (2*seq_len)` real-stacked `single`
- output: `N x num_ports x (2*seq_len)` real-stacked `single`

This matches the refactored project’s main real-stacked inference path and avoids custom complex-tensor handling during export.
 
## Architecture Note

The project now uses a lightweight workflow architecture:

- thin CLI entrypoints:
  - `train.py`
  - `evaluate_models_refactored.py`
  - `export_onnx.py`
  - `plot.py`
- shared orchestration modules:
  - `workflows/train_workflow.py`
  - `workflows/postprocess_workflow.py`
  - `workflows/evaluation_workflow.py`
  - `workflows/export_workflow.py`
  - `workflows/plotting_workflow.py`
  - `workflows/reporting.py`

This is a practical research-friendly structure:

- scripts stay easy to use from the command line
- notebook or benchmark code can call workflow APIs directly
- training, evaluation, export, and plotting share one artifact schema
- future experiment logic changes happen in one workflow layer instead of being duplicated across scripts

  TDL: C-300
    C-300: 100%|███████████████████| 11/11 [00:05<00:00,  2.01it/s]
    ✓ 完成 C-300

✓ 模型 separator1_default_hd64_stages2_depth3 评估完成

✓ Evaluation completed!
  Results saved to: experiments_refactored/evaluation_results
```

---

### 绘图阶段

```
================================================================================
📈 Generating Plots
================================================================================

  ✓ Generated: nmse_vs_snr_TDL_A_30.png
  ✓ Generated: nmse_vs_snr_TDL_B_100.png
  ✓ Generated: nmse_vs_snr_TDL_C_300.png
  ✓ Generated: nmse_vs_snr_combined.png

✓ Plots generated!
  Saved to: experiments_refactored/plots
```

---

### 最终总结

```
================================================================================
🎉 Complete Pipeline Finished!
================================================================================
  Training:   experiments_refactored/separator1_default_training
  Evaluation: experiments_refactored/separator1_default_training/evaluation_results
  Plots:      experiments_refactored/separator1_default_training/plots
================================================================================
```

---

## 🔧 独立使用各模块

### 只评估（不训练）

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
    --exp_dir "./Model_AIIC_refactor/experiments_refactored/separator1_default_training" \
  --device cuda
```

如果一次实验里训练了很多 run，只评估其中一部分可以直接加：

```bash
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/separator1_default_training" \
    --runs "separator1_default_hd64_stages2_depth3,separator2_default_hd64_stages3_depth3" \
    --device cuda
```

这样只会评估你点名的 run，不会把这个实验目录下所有超参数组合都跑一遍。

默认会写到：

```text
单 run: ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/<run_name>/evaluations/<timestamp>/
多 run: ./Model_AIIC_refactor/experiments_refactored/<timestamp>_<experiment_name>/evaluations/<timestamp>_<scope>/
```

这样重复评估不同模型组合、不同 SNR 范围、不同 TDL 配置时不会互相覆盖。

### 只绘图（不训练/评估）

```bash
python ./Model_AIIC_refactor/plot.py \
  --input "./Model_AIIC_refactor/experiments_refactored/separator1_default_training"
```

`plot.py` 的 `--input` 支持三种形式：

- 实验目录：自动选择最新一次评估目录
- 评估目录：读取其中的 `evaluation_results.json`
- 结果 JSON 文件本身

默认绘图输出到该次评估目录下的 `plots/`。

---

## 📈 典型工作流

### 工作流1：快速实验

```bash
# 训练 + 快速评估
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1 \
    --num_batches 1000 \
    --device cuda \
    --eval_after_train \
    --eval_snr_range "20:-5:0" \
    --eval_num_batches 50 \
    --plot_after_eval
```

**时间**：~5 分钟
**用途**：快速验证想法

---

### 工作流2：完整训练

```bash
# 完整训练 + 完整评估
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
    --device cuda \
    --eval_after_train \
    --eval_snr_range "30:-3:0" \
    --eval_num_batches 100 \
    --plot_after_eval
```

**时间**：~30 分钟
**用途**：正式实验

---

### 工作流3：Grid Search

```bash
# 网格搜索 + 评估 + 对比图
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
    --device cuda \
    --eval_after_train \
    --eval_num_batches 200 \
    --plot_after_eval
```

**时间**：~2-4 小时
**用途**：找到最佳配置

---

## 💡 最佳实践

### 1. GPU训练 + 完整流程

```bash
# 推荐：GPU + 自动优化 + 完整流程
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**优点**：
- ✅ 自动启用 torch.compile（训练加速）
- ✅ 评估禁用编译（避免首次开销）
- ✅ 一次运行获得所有结果

---

### 2. 调试模式

```bash
# CPU + 不编译 + 小数据集
python ./Model_AIIC_refactor/train.py \
  --experiment quick_separator1 \
    --num_batches 100 \
    --device cpu \
    --no-compile \
    --eval_after_train \
    --eval_num_batches 10
```

**优点**：
- ✅ 快速验证代码逻辑
- ✅ 不需要GPU
- ✅ 错误信息清晰

---

### 3. 后期补充评估/绘图

```bash
# 如果忘了加 --eval_after_train，后期补充：

# 1. 评估已训练的模型
python ./Model_AIIC_refactor/evaluate_models_refactored.py \
  --exp_dir "./Model_AIIC_refactor/experiments_refactored/20260409_033734_default_6port_separator1" \
    --device cuda

# 2. 生成图表
python ./Model_AIIC_refactor/plot.py \
  --input "./Model_AIIC_refactor/experiments_refactored/20260409_033734_default_6port_separator1/evaluations/20260409_063011_4-runs__migrated_from_experiment"
```

---

## ⚙️ 技术细节

### 实现方式

1. **train.py** 添加参数控制
   - `--eval_after_train`: 触发评估
   - `--plot_after_eval`: 触发绘图

2. **evaluate_models_refactored.py** 添加程序化接口
   - `evaluate_models_programmatic()`: 供其他脚本调用
   - 保持 `main()` 用于命令行调用

3. **plot.py** 简化绘图接口
   - `generate_plots_programmatic()`: 自动生成所有图表
   - 非交互式后端（Agg）

### 数据流

```
train.py
  ↓ (训练)
  → 保存模型到 save_dir/
  ↓ (if --eval_after_train)
  → evaluate_models_programmatic()
      → 评估所有模型
      → 单 run 保存到 run_dir/evaluations/<timestamp>/
      → 多 run 保存到 save_dir/evaluations/<timestamp>_<scope>/
  ↓ (if --plot_after_eval)
  → generate_plots_programmatic()
      → 读取该次评估目录下的 evaluation_results.json
      → 在对应评估目录下生成 plots/
  ↓
✓ 完成！
```

---

## ✅ 优势总结

### vs Bash 脚本

| 方面 | 方案A (train.py参数) | Bash脚本 |
|------|---------------------|----------|
| **易用性** | ✅ 一个命令 | ❌ 需要维护脚本 |
| **跨平台** | ✅ 纯Python | ❌ 需要bash+bat |
| **参数传递** | ✅ 统一参数 | ❌ 需要硬编码 |
| **错误处理** | ✅ 清晰 | ❌ 分散 |
| **调试** | ✅ 单进程 | ❌ 多进程 |
| **灵活性** | ✅ 高（可选启用） | ⚠️ 中 |

### 关键优势

1. ✅ **一个命令完成所有**
2. ✅ **参数统一管理**
3. ✅ **模块化设计**（可独立使用）
4. ✅ **错误处理完善**
5. ✅ **跨平台**（纯Python）
6. ✅ **易于维护**

---

## 🎉 开始使用

```bash
# 最简单的完整流程
python ./Model_AIIC_refactor/train.py \
  --experiment compare_default_models \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**一行命令，完成所有工作！** 🚀
