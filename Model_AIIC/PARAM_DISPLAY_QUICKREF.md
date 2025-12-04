# 参数量显示功能 - 快速参考

## ✅ 新功能

图例和标题中自动显示模型参数量！

## 📊 效果对比

### 之前
```
stages=2_share=False - A-30
stages=3_share=False - B-100
```

### 现在 ⭐
```
stages=2_share=False (104.6K) - A-30
stages=3_share=False (157.0K) - B-100
```

## 🚀 使用方法

### 1. 评估（自动读取/计算参数量）

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval
```

### 2. 绘图（自动显示参数量）

```bash
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

## 📋 格式规则

| 参数量 | 显示格式 |
|--------|----------|
| 500 | `500` |
| 1,500 | `1.5K` |
| 104,640 | `104.6K` |
| 1,234,567 | `1.2M` |
| 未知 | 不显示 |

## 🎯 实际示例

### 4 端口模型
```
stages=2_share=False (104.6K)
stages=3_share=True (78.3K)
stages=4_share=False (209.3K)
```

### 6 端口模型
```
stages=2_share=False (234.5K)
stages=3_share=True (140.2K)
stages=4_share=False (468.9K)
```

## 💡 优势

1. **快速对比复杂度** - 一眼看出参数量差异
2. **权重共享效果** - 清晰显示参数减少
3. **性价比分析** - 结合 NMSE 和参数量选择最优模型
4. **自动兼容** - 新旧模型都支持

## 📝 技术细节

- **来源**: 从 checkpoint 的 `hyperparameters['num_params']` 读取
- **备用**: 如果未保存，自动计算 `sum(p.numel())`
- **兼容**: 旧模型自动计算，新模型直接读取

---

**更新**: 2025-12-04 | **版本**: v2.4
