# Standalone Eval And Plot
## Summary

Evaluation and plotting can still be run independently.
What changed is only the training entry point: training is experiment-first.

## Supported Workflow
### Step 1: Train a named experiment

```bash
python train.py \
    --experiment compare_default_models \
    --device cuda
```

This creates a timestamped experiment directory under experiments_refactored.
### Step 2: Evaluate later

```bash
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20260408_120000_compare_default_models" \
    --device cuda \
    --snr_range "30:-3:0" \
    --tdl "A-30,B-100,C-300" \
    --num_batches 100 \
    --batch_size 2048 \
    --output "./experiments_refactored/20260408_120000_compare_default_models/evaluation_results"
```
### Step 3: Plot later

```bash
python plot.py \
    --input "./experiments_refactored/20260408_120000_compare_default_models/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20260408_120000_compare_default_models/plots"
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
        plots/
            ...png
    TRAINING_REPORT.md
```

## Notes
- Training is launched by experiment name only.
- Evaluation works on an experiment output directory and discovers trained runs inside it.
- Plotting works on evaluation_results.json and is fully decoupled from training.
- You can re-run evaluation with different SNR or TDL settings as many times as you want.

## Typical Use Cases
### Train now, evaluate later

```bash
python train.py --experiment compare_default_models --device cuda
python evaluate_models.py --exp_dir "./experiments_refactored/<timestamp>_compare_default_models" --device cuda
```
### Train once, evaluate multiple ways

```bash
python evaluate_models.py \
    --exp_dir "./experiments_refactored/<timestamp>_compare_default_models" \
    --snr_range "30:-3:0" \
    --output "./experiments_refactored/<timestamp>_compare_default_models/eval_full"

python evaluate_models.py \
    --exp_dir "./experiments_refactored/<timestamp>_compare_default_models" \
    --snr_range "20:-2:0" \
    --output "./experiments_refactored/<timestamp>_compare_default_models/eval_fast"
```
### Benchmark presets

```bash
python compare_cpu_gpu.py --experiment perf_quick --skip_gpu
python compare_optimizations.py --experiment perf_quick --skip_gpu
```
## Policy

The old model_config plus training_config training CLI is removed. For training and benchmark launches, experiments.yaml is the supported interface.
# ✅ 独立使用 Eval 和 Plot

## 🎯 确认：完全可以独立使用！

### 三种工作流

#### 1. **一键完成**（推荐）⭐
```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**输出**：训练 + 评估 + 绘图，一次完成

---

#### 2. **只训练，稍后评估和绘图**（灵活）⭐⭐

**步骤1：训练**
```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda
# ✅ 只训练，不评估
```

**输出**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
└── separator1_hd64_stages2_depth3/
    ├── model.pth          ← 训练好的模型
    ├── config.yaml
    └── training_history.png
```

---

**步骤2：稍后评估（独立运行）**
```bash
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda \
    --snr_range "30:-3:0" \
    --tdl "A-30,B-100,C-300" \
    --num_batches 100 \
    --batch_size 2048 \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results"
```

**输出**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
├── separator1_hd64_stages2_depth3/
│   └── model.pth
└── evaluation_results/        ← ✅ 新增
    ├── evaluation_results.json
    └── evaluation_results.npy
```

---

**步骤3：稍后绘图（独立运行）**
```bash
python plot.py \
    --input "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/plots"
```

**输出**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
├── separator1_hd64_stages2_depth3/
├── evaluation_results/
└── plots/                     ← ✅ 新增
    ├── nmse_vs_snr_TDL_A_30.png
    ├── nmse_vs_snr_TDL_B_100.png
    ├── nmse_vs_snr_TDL_C_300.png
    └── nmse_vs_snr_combined.png
```

---

#### 3. **完全手动控制**

你可以在**任何时候**、**任何顺序**运行这些脚本：

```bash
# 今天训练
python train.py ...

# 明天评估
python evaluate_models.py --exp_dir ...

# 后天绘图
python plot.py --input ...

# 下周重新评估（不同参数）
python evaluate_models.py --exp_dir ... --snr_range "20:-2:0"

# 再次绘图
python plot.py --input ...
```

**✅ 完全独立，互不依赖！**

---

## 📋 详细使用示例

### 场景1：训练后立即评估

```bash
# 1. 训练（不加 eval 参数）
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda

# 输出：
# 🚀 Experiment: 20251212_103045_separator1_default_default
# ...
# ✓ Model saved to: ./experiments_refactored/20251212_103045_separator1_default_default/separator1_hd64_stages2_depth3

# 2. 立即评估（使用上面的目录）
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda

# 输出：
# Using device: cuda
# ...
# ✓ 保存 JSON: ./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results/evaluation_results.json

# 3. 立即绘图
python plot.py \
    --input "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/plots"

# 输出：
# ✓ Generated: nmse_vs_snr_TDL_A_30.png
# ✓ Generated: nmse_vs_snr_TDL_B_100.png
# ...
```

---

### 场景2：训练后几天再评估

```bash
# 周一训练
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda
# 创建：20251209_103045_separator1_default_default/

# 周五评估（几天后）
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251209_103045_separator1_default_default" \
    --device cuda \
    --snr_range "30:-3:0" \
    --tdl "A-30,B-100,C-300"

# ✅ 没问题！模型已保存，可以随时评估
```

---

### 场景3：Grid Search 后批量评估

```bash
# 1. Grid Search 训练
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
    --device cuda

# 输出：18个模型
# 创建：20251212_103045_separator1_grid_search_default/
#   ├── separator1_hd32_stages2_depth3/
#   ├── separator1_hd64_stages2_depth3/
#   └── ... (18 models)

# 2. 稍后评估所有模型
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_grid_search_default" \
    --device cuda \
    --num_batches 100

# ✅ 自动发现并评估所有18个模型

# 3. 绘制对比图
python plot.py \
    --input "./experiments_refactored/20251212_103045_separator1_grid_search_default/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20251212_103045_separator1_grid_search_default/plots"

# ✅ 生成所有模型的对比图
```

---

### 场景4：重新评估（不同参数）

```bash
# 第一次评估（完整）
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda \
    --snr_range "30:-3:0" \
    --tdl "A-30,B-100,C-300" \
    --num_batches 100 \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results_full"

# 第二次评估（快速，不同参数）
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda \
    --snr_range "20:-5:0" \
    --tdl "A-30" \
    --num_batches 50 \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results_quick"

# ✅ 两次评估结果独立保存
```

---

### 场景5：对比不同实验

```bash
# 实验1评估
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda \
    --output "./comparison/exp1_results"

# 实验2评估
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_105123_separator1_small_quick" \
    --device cuda \
    --output "./comparison/exp2_results"

# 分别绘图对比
python plot.py --input "./comparison/exp1_results/evaluation_results.json" --output "./comparison/exp1_plots"
python plot.py --input "./comparison/exp2_results/evaluation_results.json" --output "./comparison/exp2_plots"
```

---

## 🔧 命令行参数

### evaluate_models.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--exp_dir` | ✅ | - | 训练实验目录 |
| `--device` | ❌ | auto | 设备 (auto/cpu/cuda) |
| `--snr_range` | ❌ | "30:-3:0" | SNR范围 |
| `--tdl` | ❌ | "A-30,B-100,C-300" | TDL配置 |
| `--num_batches` | ❌ | 100 | 评估批次数 |
| `--batch_size` | ❌ | 2048 | 批大小 |
| `--output` | ❌ | ./evaluation_results | 结果保存目录 |
| `--models` | ❌ | None | 指定模型（None=全部） |
| `--use_amp` | ❌ | False | 启用混合精度 |
| `--no-compile` | ❌ | False | 禁用编译 |

---

### plot.py

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | ✅ | - | evaluation_results.json路径 |
| `--output` | ❌ | ./plots | 图表保存目录 |

---

## ✅ 优势

### 独立使用的优势

1. **灵活性** ⭐⭐⭐
   - 训练和评估分离
   - 可以随时重新评估
   - 可以尝试不同评估参数

2. **资源优化** ⭐⭐
   - 训练用GPU
   - 评估可以稍后用空闲GPU
   - 或用CPU慢慢评估

3. **并行化** ⭐⭐⭐
   - 一个GPU训练
   - 另一个GPU评估旧模型
   - 提高资源利用率

4. **调试友好** ⭐⭐
   - 训练出错不影响已有结果
   - 评估出错可以重跑
   - 绘图出错可以重新生成

---

## 🎯 推荐工作流

### 快速实验（推荐一键）

```bash
python train.py \
    --model_config separator1_small \
    --training_config quick_test \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**时间**：~10分钟，全部完成

---

### 正式实验（推荐分离）

```bash
# 1. 训练（长时间，可以放后台）
nohup python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda > train.log 2>&1 &

# 2. 等训练完成，稍后评估
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_default_default" \
    --device cuda

# 3. 立即绘图
python plot.py \
    --input "./experiments_refactored/20251212_103045_separator1_default_default/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20251212_103045_separator1_default_default/plots"
```

**优点**：
- ✅ 训练可以后台运行
- ✅ 评估可以选择空闲时间
- ✅ 出错可以单独重跑

---

### Grid Search（推荐分离）

```bash
# 1. 训练多个模型（长时间）
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
    --device cuda

# 2. 稍后批量评估
python evaluate_models.py \
    --exp_dir "./experiments_refactored/20251212_103045_separator1_grid_search_default" \
    --device cuda \
    --num_batches 200  # 更多批次，更准确

# 3. 生成对比图
python plot.py \
    --input "./experiments_refactored/20251212_103045_separator1_grid_search_default/evaluation_results/evaluation_results.json" \
    --output "./experiments_refactored/20251212_103045_separator1_grid_search_default/plots"
```

---

## 💡 最佳实践

### 1. 训练时保存完整配置

```bash
python train.py ... > train_log.txt
# 保存命令和输出，方便后续追溯
```

---

### 2. 评估时使用描述性输出目录

```bash
python evaluate_models.py \
    --exp_dir "..." \
    --output "./results/eval_full_snr30-0_tdl_all"
```

---

### 3. 批量处理

```bash
# 评估多个实验
for exp_dir in experiments_refactored/202512*/; do
    python evaluate_models.py --exp_dir "$exp_dir" --device cuda
done
```

---

## ✅ 总结

### 回答你的问题

**Q: 如果train的时候没加eval和plot的参数，我后面也可以用eval和plot，单独用吧？**

**A: 完全可以！✅✅✅**

1. ✅ `evaluate_models.py` 可以独立运行
2. ✅ `plot.py` 可以独立运行
3. ✅ 可以在任何时候运行
4. ✅ 可以运行多次（不同参数）
5. ✅ 完全不依赖训练过程

**只要模型文件存在，就可以随时评估和绘图！**

---

### 三种模式对比

| 模式 | 命令 | 优点 | 适用场景 |
|------|------|------|---------|
| **一键** | `train.py --eval_after_train --plot_after_eval` | 简单快速 | 小实验 |
| **分离** | `train.py` → `evaluate_models.py` → `plot.py` | 灵活可控 | 正式实验 ⭐ |
| **混合** | `train.py --eval_after_train` → `plot.py` | 部分自动 | 中等实验 |

---

**推荐**：正式实验用**分离模式**，快速测试用**一键模式**！
