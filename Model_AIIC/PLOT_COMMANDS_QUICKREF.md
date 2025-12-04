# 绘图命令更新 - 快速参考

## ✅ 已添加到 README.md

### 6 端口模型可视化完整流程

```bash
# 1. 评估 6 端口模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 2. 绘图 - 简单命令
python Model_AIIC/plot_results.py \
  --input out6ports_eval

# 3. 绘图 - 按 TDL 分图（推荐）⭐
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl

# 4. 绘图 - 按模型分图
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_model
```

## 📊 布局类型对比

| 命令 | 布局 | 特点 | 推荐度 |
|------|------|------|--------|
| `--layout single` | 单图 | 所有配置在一起 | ⭐⭐ |
| `--layout subplots_tdl` | TDL分图 | 每个TDL独立 | ⭐⭐⭐ |
| `--layout subplots_model` | 模型分图 | 每个模型独立 | ⭐⭐ |

## 🎯 快速命令

### 最简单（使用默认布局）

```bash
python Model_AIIC/plot_results.py --input out6ports_eval
```

### 推荐使用（TDL 分图）

```bash
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

## 📍 在 README.md 中的位置

- **章节**: "6. 结果可视化"
- **小节**: "6.3 常用示例"
- **搜索关键词**: "6 端口模型可视化"

## 🔍 图表标题信息

绘图脚本会自动显示：

✅ 模型配置（stages=3, share=False, loss=normalized）  
✅ **端口配置**（Ports: [0,2,4,6,8,10] 或 [0,3,6,9]）  
✅ TDL 类型（A-30, B-100, C-300）  
✅ SNR 范围

## 📁 输出文件

```
out6ports_eval/
├── nmse_comparison_subplots_tdl.png  # TDL 分图
├── nmse_comparison_subplots_tdl.pdf  # PDF 版本
├── nmse_comparison_single.png        # 单图
└── evaluation_results.json           # 原始数据
```

---

**更新**: 2025-12-04  
**版本**: v2.3
