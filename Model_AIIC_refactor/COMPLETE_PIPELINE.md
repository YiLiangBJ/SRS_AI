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
python train.py \
  --experiment quick_separator1 \
  --plan_only \
  --device cpu
```

### Train a named experiment

```bash
python train.py \
  --experiment compare_default_models \
  --device cuda
```

### Train, evaluate, and plot

```bash
python train.py \
  --experiment compare_default_models \
  --device cuda \
  --eval_after_train \
  --plot_after_eval
```

### Override batch count for a benchmark preset

```bash
python train.py \
  --experiment perf_quick \
  --num_batches 100 \
  --device cpu
```

## Output Layout

```text
experiments_refactored/
  <timestamp>_<experiment_name>/
    <run_name>/
      model.pth
      config.yaml
      tensorboard/
    TRAINING_REPORT.md
    evaluation_results/
    plots/
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
| --eval_snr_range | SNR range for evaluation |
| --eval_tdl | TDL list for evaluation |
| --eval_num_batches | Number of evaluation batches |
| --eval_batch_size | Evaluation batch size |
| --plot_after_eval | Generate plots after evaluation |
| --export_onnx_after_train | Export trained runs to ONNX after training |
| --onnx_export_selection | Export `best` or `all` runs |
| --onnx_output_dir | ONNX artifact directory |
| --onnx_opset | ONNX opset version |
| --onnx_batch_size | Dummy batch size used for tracing |
| --onnx_dynamic_batch | Export a dynamic batch dimension |
| --onnx_validate | Run ONNX checker and ORT smoke validation |
| --plan_only | Print the run plan and exit |

## Benchmark CLI

Benchmark scripts also use experiment-first entry points:

```bash
python compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python compare_optimizations.py --experiment perf_quick --skip_gpu
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
python train.py \
  --experiment compare_default_models \
  --device cuda \
  --export_onnx_after_train \
  --onnx_export_selection best \
  --onnx_validate
```

### Export runs later from an experiment directory

```bash
python export_onnx.py \
  --exp_dir experiments_refactored/<timestamp>_<experiment_name> \
  --runs separator2_default_hd64_stages3_depth3 \
  --output onnx_exports \
  --opset 13 \
  --dynamic_batch \
  --validate
```

### Export output layout

```text
onnx_exports/
  <run_name>/
    <run_name>.onnx
    export_manifest.json
```

`export_manifest.json` stores the resolved model spec, training metadata, input/output names, dummy tensor shapes, and validation summary. This is the handoff file for Matlab or deployment tooling.

### Matlab import helper

Use `matlab/import_refactor_onnx.m`:

```matlab
net = import_refactor_onnx("onnx_exports/my_run");
```

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
python evaluate_models_refactored.py \
    --exp_dir "./experiments_refactored/separator1_default_training" \
    --device cuda \
    --output "./my_evaluation_results"
```

### 只绘图（不训练/评估）

```bash
python plot.py \
    --input "./evaluation_results/evaluation_results.json" \
    --output "./my_plots"
```

---

## 📈 典型工作流

### 工作流1：快速实验

```bash
# 训练 + 快速评估
python train.py \
    --model_config separator1_small \
    --training_config quick_test \
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
python train.py \
    --model_config separator1_default \
    --training_config default \
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
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
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
python train.py \
    --model_config separator1_default \
    --training_config default \
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
python train.py \
    --model_config separator1_small \
    --training_config quick_test \
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
python evaluate_models_refactored.py \
    --exp_dir "./experiments_refactored/my_experiment" \
    --device cuda

# 2. 生成图表
python plot.py \
    --input "./experiments_refactored/my_experiment/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/my_experiment/plots"
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
      → 保存结果到 save_dir/evaluation_results/
  ↓ (if --plot_after_eval)
  → generate_plots_programmatic()
      → 读取评估结果
      → 生成图表到 save_dir/plots/
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
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**一行命令，完成所有工作！** 🚀
