# 超参数配置指南

本指南说明如何使用新增的 `hidden_dim` 和 `num_sub_stages` 超参数来控制模型容量和性能。

---

## 📋 新增超参数

### 1. `hidden_dim` (默认: 64)
- **含义**: MLP 隐藏层的维度（宽度）
- **范围**: 建议 32-256
- **影响**: 控制网络的**宽度**，越大表达能力越强，但参数量增加

### 2. `num_sub_stages` (默认: 2)
- **含义**: 每个 MLP 中隐藏层的数量（深度）
- **范围**: 建议 1-6
- **影响**: 控制网络的**深度**，越大表达能力越强，但训练可能更困难

---

## 🎯 参数量对比

### Model_AIIC_onnx (num_stages=3, share_weights=False)

| 配置 | hidden_dim | num_sub_stages | 参数量 | 特点 |
|------|-----------|---------------|--------|------|
| **窄而浅** | 32 | 1 | 19,488 | 最少参数，最快训练 |
| **默认** | 64 | 2 | 138,528 | 平衡性能和速度 |
| **宽而中等** | 128 | 2 | 473,376 | 高表达能力 |
| **中等而深** | 64 | 4 | 338,208 | 深层特征提取 |
| **窄而深** | 32 | 6 | 146,208 | 参数效率高 |

### Model_AIIC (复数版，参数量更多)

| 配置 | hidden_dim | num_sub_stages | 参数量 | vs ONNX |
|------|-----------|---------------|--------|---------|
| **窄而浅** | 32 | 1 | 57,120 | 2.93x |
| **默认** | 64 | 2 | 156,960 | 1.13x |
| **宽而中等** | 128 | 2 | 540,000 | 1.14x |
| **中等而深** | 64 | 4 | 356,640 | 1.05x |

---

## 🚀 使用方法

### 1. 训练时指定超参数

```bash
# 单一配置
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --hidden_dim "64" \
  --num_sub_stages "2" \
  --stages "3"

# 网格搜索（多个值）
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --hidden_dim "32,64,128" \
  --num_sub_stages "1,2,3" \
  --stages "2,3"

# 这会训练 3×3×2 = 18 个模型组合
```

### 2. 评估时自动加载

```bash
# evaluate_models.py 会自动从保存的 config 中读取
python Model_AIIC_onnx/evaluate_models.py \
  --exp_dir "./experiments/20251209_143045_batch10000_bs2048_ports4_snr0-30" \
  --snr_range "30:-3:0"
```

### 3. Python 代码中使用

```python
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

# 创建模型
model = ResidualRefinementSeparatorReal(
    seq_len=12,
    num_ports=4,
    hidden_dim=128,        # ⭐ 更宽的网络
    num_stages=3,
    num_sub_stages=4,      # ⭐ 更深的网络
    share_weights_across_stages=False,
    activation_type='relu'
)

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

## 💡 配置建议

### 场景 1: 快速原型验证
```bash
--hidden_dim "32" --num_sub_stages "1" --batches 1000
```
- 参数量: ~20k
- 训练时间: 最快
- 用途: 验证数据流程、调试代码

### 场景 2: 标准训练（推荐）
```bash
--hidden_dim "64" --num_sub_stages "2" --batches 50000
```
- 参数量: ~139k
- 训练时间: 中等
- 用途: 大多数场景的默认配置

### 场景 3: 追求最佳性能
```bash
--hidden_dim "128" --num_sub_stages "3" --batches 100000
```
- 参数量: ~673k
- 训练时间: 较长
- 用途: 最终模型，追求最低 NMSE

### 场景 4: 参数受限场景
```bash
--hidden_dim "32" --num_sub_stages "4" --batches 50000
```
- 参数量: ~219k
- 训练时间: 中等
- 用途: 嵌入式部署，追求参数效率

---

## 🔬 实验建议

### 实验 1: 宽度 vs 深度
```bash
# 测试不同的宽度
python Model_AIIC_onnx/test_separator.py \
  --hidden_dim "32,64,128,256" \
  --num_sub_stages "2" \
  --stages "3" \
  --batches 10000

# 测试不同的深度
python Model_AIIC_onnx/test_separator.py \
  --hidden_dim "64" \
  --num_sub_stages "1,2,3,4,5" \
  --stages "3" \
  --batches 10000
```

### 实验 2: 参数量匹配
```bash
# 让 ONNX 版本参数量接近 AIIC 版本 (157k)
python Model_AIIC_onnx/test_separator.py \
  --hidden_dim "92" \
  --num_sub_stages "2" \
  --stages "3" \
  --batches 50000

# 对比性能差异
```

### 实验 3: 全网格搜索
```bash
# 警告: 这会训练 4×4×2 = 32 个模型
python Model_AIIC_onnx/test_separator.py \
  --hidden_dim "32,64,96,128" \
  --num_sub_stages "1,2,3,4" \
  --stages "2,3" \
  --batches 10000 \
  --save_dir "./grid_search_results"
```

---

## 📊 性能预期

### SNR = 0-30 dB, TDL-A

| 配置 | 参数量 | NMSE @ 20dB | 训练时间 (10k batches) |
|------|--------|-------------|----------------------|
| hd=32, sub=1 | 19k | ~-12 dB | ~5 分钟 (CPU) |
| hd=64, sub=2 | 139k | ~-18 dB | ~8 分钟 (CPU) |
| hd=128, sub=3 | 673k | ~-22 dB | ~15 分钟 (CPU) |

*实际性能取决于具体数据和训练策略*

---

## ⚠️ 注意事项

1. **参数量增长**
   - `hidden_dim` 增加 → 参数量**平方级增长**
   - `num_sub_stages` 增加 → 参数量**线性增长**

2. **训练时间**
   - GPU: 参数量增加对训练时间影响较小
   - CPU: 参数量翻倍 → 训练时间约增加 50-80%

3. **过拟合风险**
   - 参数量 > 500k 时，建议增加训练数据量
   - 小数据集（< 10k batches）建议使用 hd=32-64

4. **内存占用**
   - GPU: 参数量 < 1M 时通常无问题
   - CPU: 大 batch_size 时注意内存

---

## 🎯 最佳实践

1. **从默认配置开始**
   ```bash
   --hidden_dim "64" --num_sub_stages "2"
   ```

2. **先调宽度，再调深度**
   - 宽度对性能影响更直接
   - 深度需要更多训练才能收敛

3. **使用学习率衰减**
   - 深层网络（num_sub_stages > 3）建议用 lr scheduler

4. **保存所有配置**
   - 每次实验的 config 都会自动保存到 `model.pth` 和 `metrics.json`
   - 便于后续复现和对比

---

## 📝 相关文件

- **模型定义**: 
  - `Model_AIIC/channel_separator.py`
  - `Model_AIIC_onnx/channel_separator.py`
  - `Model_AIIC_onnx/complex_layers.py`

- **训练脚本**: 
  - `Model_AIIC_onnx/test_separator.py`

- **评估脚本**: 
  - `Model_AIIC_onnx/evaluate_models.py`

- **结果可视化**: 
  - `Model_AIIC_onnx/plot_results.py`

---

## 🔗 下一步

1. 阅读 `README.md` 了解完整的训练流程
2. 运行 `python Model_AIIC_onnx/test_separator.py --help` 查看所有参数
3. 查看 `experiments/` 目录中保存的 `training_report.md` 了解训练详情

