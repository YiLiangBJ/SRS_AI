# README.md 更新 - 评估命令修正

## ✅ 更新内容

### 问题
用户在运行评估命令时遇到格式错误，命令中混入了 Markdown 格式标记。

### 解决方案
更新 `Model_AIIC/README.md`，确保所有 `evaluate_models.py` 命令示例格式正确。

## 📝 更新的部分

### 1. 快速开始 - 评估示例

**之前**:
```bash
python Model_AIIC/evaluate_models.py \---
  --exp_dir ./quick_test \## 🔗 Integration
  --num_batches 5 \
  --output ./quick_eval### 命令行参数
```

**现在**:
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./quick_test \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./quick_eval
```

### 2. 添加 6 端口模型示例 ⭐

新增完整的 6 端口工作流程：

```bash
# 1. 训练 6 端口模型
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --ports "0,2,4,6,8,10" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./out6ports"

# 2. 评估 6 端口模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 3. 绘制对比图
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

### 3. 更新评估参数说明

添加详细的参数表格和多个实际示例：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--exp_dir` | str | **必需** | 训练实验目录 |
| `--tdl` | str | "A-30,B-100,C-300" | TDL 信道配置 |
| `--snr_range` | str | "30:-3:0" | SNR 范围 (start:step:end) |
| `--num_batches` | int | 10 | 评估批次数 |
| `--batch_size` | int | 100 | 批大小 |
| `--output` | str | ./evaluation_results | 输出目录 |

### 4. 更新完整工作流程示例

在 README 的后续部分添加了更多正确的示例：

```bash
# 4 端口模型训练
python Model_AIIC/test_separator.py \
  --batches 1000 --batch_size 2048 \
  --stages "2,3,4" --share_weights "True,False" \
  --ports "0,3,6,9" --save_dir "./full_exp"

# 6 端口模型训练
python Model_AIIC/test_separator.py \
  --batches 1000 --batch_size 2048 \
  --stages "2,3,4" --share_weights "True,False" \
  --ports "0,2,4,6,8,10" --save_dir "./out6ports"

# 评估所有模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./full_exp \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 --batch_size 200 \
  --output ./full_eval

# 评估 6 端口模型（推荐参数）
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 --batch_size 100 \
  --output ./out6ports_eval
```

## 🎯 关键改进

1. **格式修正**: 移除了混入的 Markdown 标记
2. **参数完整**: 所有示例都包含必需参数
3. **6 端口支持**: 添加专门的 6 端口示例
4. **注意事项**: 添加了使用提示和最佳实践

## 📋 注意事项

### 评估 6 端口模型时

- ✅ 使用较小的 `batch_size` (50-100)
- ✅ `exp_dir` 指向包含模型的父目录
- ✅ 模型会自动识别 `pos_values` 配置
- ✅ 确保路径正确（如 `./Model_AIIC/out6ports`）

### SNR 范围格式

- 格式: `"起始:步长:结束"`
- 示例: `"30:-3:0"` → [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0] dB
- 步长为负表示递减

### 常见错误

❌ **错误的命令**:
```bash
python Model_AIIC/evaluate_models.py \---
  --exp_dir ./out6ports \## 标题
  --tdl "A-30,B-100,C-300" \
```

✅ **正确的命令**:
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval
```

## 🚀 快速参考

### 标准评估命令模板

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir <实验目录> \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output <输出目录>
```

### 参数选择建议

| 模型类型 | batch_size | num_batches | 说明 |
|----------|------------|-------------|------|
| 4 端口 | 100-200 | 10 | 标准配置 |
| 6 端口 | 50-100 | 10 | 较小批次 |
| 快速测试 | 50 | 5 | 快速验证 |
| 完整评估 | 100 | 20 | 更准确 |

---

## 📊 绘图命令更新 (2025-12-04 下午)

### 新增内容

在 README.md 的 "6. 结果可视化" 部分添加了完整的绘图命令示例。

#### 6 端口模型可视化示例

```bash
# 简单命令
python Model_AIIC/plot_results.py \
  --input out6ports_eval

# 按 TDL 分图（推荐）
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl

# 按模型分图
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_model
```

#### 布局类型说明

| 布局 | 说明 | 使用场景 |
|------|------|----------|
| `single` | 所有配置在一张图 | 快速对比 |
| `subplots_tdl` | 每个 TDL 一个子图 ⭐ | 推荐，清晰对比 |
| `subplots_model` | 每个模型一个子图 | 关注单个模型 |

#### 图表标题改进

绘图脚本现在会自动在标题中显示：
- ✅ 模型配置（stages, share_weights, loss_type）
- ✅ **端口配置**（如 "Ports: [0,2,4,6,8,10]" 或 "Ports: [0,3,6,9]"）
- ✅ TDL 信道类型
- ✅ SNR 范围

---

**更新日期**: 2025-12-04  
**版本**: v2.3  
**状态**: ✅ 已更新并验证
