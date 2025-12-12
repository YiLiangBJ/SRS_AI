# 🚀 一键完整流程：Train → Eval → Plot

## ✅ 已实现功能

### 方案A：在 train.py 中添加参数

通过命令行参数控制完整的训练、评估、绘图流程，一个命令完成所有工作！

---

## 🎯 使用方法

### 1. 基础用法：只训练

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda
```

**输出**：
- 训练好的模型保存在 `./experiments_refactored/...`
- 训练报告 `TRAINING_REPORT.md`

---

### 2. 训练 + 评估

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train
```

**输出**：
- 训练好的模型
- 评估结果 `evaluation_results/`
  - `evaluation_results.json`
  - `evaluation_results.npy`

**默认评估参数**：
- SNR 范围：30:-3:0 dB
- TDL 配置：A-30, B-100, C-300
- 每个SNR点：100 batches × 2048 samples

---

### 3. 训练 + 评估 + 绘图 ⭐ 推荐

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**输出**：
```
./experiments_refactored/
├── separator1_default_xxx/
│   ├── model.pth
│   └── training_history.png
├── evaluation_results/
│   ├── evaluation_results.json
│   └── evaluation_results.npy
├── plots/
│   ├── nmse_vs_snr_TDL_A_30.png
│   ├── nmse_vs_snr_TDL_B_100.png
│   ├── nmse_vs_snr_TDL_C_300.png
│   └── nmse_vs_snr_combined.png
└── TRAINING_REPORT.md
```

**完整流程**：
1. ✅ 训练模型
2. ✅ 自动评估
3. ✅ 自动生成图表

---

### 4. 自定义评估参数

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --eval_snr_range "20:-2:0" \
    --eval_tdl "A-30,C-300" \
    --eval_num_batches 200 \
    --eval_batch_size 4096 \
    --plot_after_eval
```

**自定义参数**：
- `--eval_snr_range`: SNR范围（默认 "30:-3:0"）
- `--eval_tdl`: TDL配置（默认 "A-30,B-100,C-300"）
- `--eval_num_batches`: 批次数（默认 100）
- `--eval_batch_size`: 批大小（默认 2048）

---

### 5. Grid Search + 完整流程

```bash
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**自动完成**：
1. ✅ 训练所有配置（18个模型）
2. ✅ 评估所有模型
3. ✅ 生成对比图表

---

## 📊 命令行参数完整列表

### 训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_config` | str | separator1_default | 模型配置名称 |
| `--training_config` | str | default | 训练配置名称 |
| `--batch_size` | int | None | 覆盖batch size |
| `--num_batches` | int | None | 覆盖batch数量 |
| `--device` | str | auto | 设备 (auto/cpu/cuda) |
| `--save_dir` | str | ./experiments_refactored | 保存目录 |
| `--no-amp` | flag | False | 禁用混合精度 |
| `--no-compile` | flag | False | 禁用模型编译 |

### 评估参数 ⭐ NEW

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--eval_after_train` | flag | False | 训练后自动评估 |
| `--eval_snr_range` | str | "30:-3:0" | 评估SNR范围 |
| `--eval_tdl` | str | "A-30,B-100,C-300" | 评估TDL配置 |
| `--eval_num_batches` | int | 100 | 评估批次数 |
| `--eval_batch_size` | int | 2048 | 评估批大小 |
| `--plot_after_eval` | flag | False | 评估后自动绘图 |

---

## 🎨 输出示例

### 训练完成后

```
================================================================================
Training Summary
================================================================================

Total configurations trained: 1
Start time: 2025-12-12 10:00:00
End time: 2025-12-12 10:05:30
Total duration: 0.09 hours (330.0s)

1. separator1_default_hd64_stages2_depth3:
   Final loss: 0.123456
   Min loss: 0.123456
   Eval NMSE: -5.23 dB
   Parameters: 156,032
   Duration: 330.0s

🏆 Best configuration: separator1_default_hd64_stages2_depth3
   NMSE: -5.23 dB

✓ Training report saved: experiments_refactored/TRAINING_REPORT.md

✓ All training completed!
```

---

### 评估阶段

```
================================================================================
📊 Post-Training Evaluation
================================================================================

Using device: cuda
  GPU: NVIDIA RTX 4090
  CUDA version: 12.1

SNR 范围: [30.0, 27.0, 24.0, ..., 3.0, 0.0]
TDL 配置: ['A-30', 'B-100', 'C-300']

================================================================================
评估模型: separator1_default_hd64_stages2_depth3
================================================================================
✓ 模型加载成功 (device: cuda)
  配置: separator1, stages=2, share_weights=False
  参数数量: 156,032
  端口位置: [0, 3, 6, 9]

  TDL: A-30
    A-30: 100%|████████████████████| 11/11 [00:05<00:00,  2.05it/s]
    ✓ 完成 A-30

  TDL: B-100
    B-100: 100%|███████████████████| 11/11 [00:05<00:00,  2.03it/s]
    ✓ 完成 B-100

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
python evaluate_models.py \
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
python evaluate_models.py \
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

2. **evaluate_models.py** 添加程序化接口
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
