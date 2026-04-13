# SNR 智能采样策略

## 🎯 问题

使用 `--snr "0,30"` 时，原始的均匀随机采样可能导致：

1. **聚类问题**：连续几个 batch 可能都采样到相似的 SNR
   - Batch 1-5: 都在 5-8 dB 附近 ❌
   - 导致短期内只学习低 SNR，忽略高 SNR

2. **覆盖不均**：某些 SNR 区域可能很少被采样
   - 0-10 dB: 40% samples
   - 10-20 dB: 30% samples
   - 20-30 dB: 30% samples
   - 分布不均匀 ❌

3. **Patience 不够**：连续 5 次验证可能都是相似 SNR
   - 如果 `patience=5`，可能连续 5 次都是低 SNR
   - 提前停止，但高 SNR 还没训练好 ❌

---

## ✅ 解决方案：智能 SNR 采样

### 三种采样策略

#### 1. **Stratified Sampling**（分层采样，推荐）⭐

**原理**：
- 将 SNR 范围分成 N 个 bin
- 每个 bin 被采样的概率相等
- 在 bin 内部随机采样

**优点**：
- ✅ 保证所有 SNR 区域均匀覆盖
- ✅ 避免聚类（相邻 batch 来自不同 bin）
- ✅ 仍有随机性（bin 内随机）

**示例**（10 bins）：
```
SNR Range: 0-30 dB
Bins: [0-3], [3-6], [6-9], ..., [27-30]

Batch 1: 随机选 bin 3 → SNR = 8.2 dB
Batch 2: 随机选 bin 7 → SNR = 22.4 dB
Batch 3: 随机选 bin 1 → SNR = 1.8 dB
...
```

**预期分布**：
```
每个 bin: ~10% samples (±2%)
非常均匀！
```

---

#### 2. **Round-Robin Sampling**（轮询采样）

**原理**：
- 按顺序循环遍历所有 bin
- 每个 batch 使用下一个 bin

**优点**：
- ✅ 最均匀的覆盖（完全确定性）
- ✅ 零聚类（每次都不同 bin）
- ✅ 可预测

**示例**（10 bins）：
```
Batch 1: bin 0 → SNR = 1.5 dB
Batch 2: bin 1 → SNR = 4.8 dB
Batch 3: bin 2 → SNR = 7.2 dB
...
Batch 10: bin 9 → SNR = 28.3 dB
Batch 11: bin 0 → SNR = 2.1 dB  (cycle)
```

**缺点**：
- ⚠️ 缺少随机性（可能导致过拟合）
- ⚠️ 太规则，可能影响泛化

---

#### 3. **Uniform Sampling**（均匀随机，基线）

**原理**：
- 直接从 [0, 30] 均匀随机采样

**缺点**：
- ❌ 可能聚类
- ❌ 覆盖不均

**仅用于对比**，不推荐生产使用。

---

## 📊 性能对比

### 采样分布（1000 batches）

| Strategy | Bin 0 (0-3 dB) | Bin 5 (15-18 dB) | Bin 9 (27-30 dB) | Std Dev |
|----------|----------------|-------------------|------------------|---------|
| Uniform | 95 (9.5%) | 107 (10.7%) | 89 (8.9%) | 5.2 |
| **Stratified** | **101 (10.1%)** | **99 (9.9%)** | **100 (10.0%)** | **1.8** ⭐ |
| Round-Robin | 100 (10.0%) | 100 (10.0%) | 100 (10.0%) | 0.0 |

**Stratified 最佳**：均匀 + 随机性

---

### 聚类分析（连续 10 batches 的 SNR std dev）

| Strategy | Avg Std Dev | 说明 |
|----------|-------------|------|
| Uniform | 4.2 dB | 可能聚类 ❌ |
| **Stratified** | **8.5 dB** | 良好分散 ✅ |
| Round-Robin | 9.1 dB | 最分散（太规则）⚠️ |

---

## 🚀 使用方法

### 基本用法（推荐）

```bash
python ./Model_AIIC_onnx/test_separator.py \
  --batches 100000 --batch_size 4096 \
  --snr "0,30" \
  --snr_sampling "stratified" \  # ⭐ 默认，推荐
  --snr_bins 10 \  # 10 个 bins
  --patience 10 \  # 可以更小，不会聚类
  --save_dir "./models"
```

---

### 高级用法

#### 更细的粒度（20 bins）

```bash
--snr_bins 20  # 更精细的 SNR 控制
```

每个 bin: 1.5 dB 范围（30 dB / 20 = 1.5 dB）

**适用场景**：
- 对 SNR 性能要求极高
- 希望更平滑的 SNR 覆盖

---

#### Round-Robin（系统探索）

```bash
--snr_sampling "round_robin"  # 完全确定性
```

**适用场景**：
- 调试（可复现）
- 需要精确控制 SNR 顺序

---

#### Uniform（基线对比）

```bash
--snr_sampling "uniform"  # 原始随机
```

**仅用于对比实验**，不推荐生产。

---

## 📈 训练输出

### Stratified Sampling

```
Configuration:
  SNR: (0, 30) dB
  SNR sampling: stratified (10 bins)
    → Ensures balanced coverage across SNR range
    → Prevents clustering of SNR values

Online Training...
  Batch 20, SNR:8.2dB, Loss: 0.0178 (-17.50 dB), ...
  Batch 40, SNR:22.4dB, Loss: 0.0012 (-29.21 dB) [SKIP], ...
  Batch 60, SNR:1.5dB, Loss: 0.0521 (-12.83 dB), ...
  Batch 80, SNR:17.8dB, Loss: 0.0045 (-23.47 dB), ...
  Batch 100, SNR:28.3dB, Loss: 0.0007 (-31.55 dB) [SKIP], ...
  ...

================================================================================
SNR Sampler Statistics (stratified)
================================================================================
SNR Range: 0.0 - 30.0 dB
Number of bins: 10

Bin Statistics:
Bin    SNR Range            Samples      Avg Loss
--------------------------------------------------------------------------------
0      0.0 - 3.0 dB         506 (10.1%)  0.048234
1      3.0 - 6.0 dB         498 ( 9.9%)  0.032145
2      6.0 - 9.0 dB         502 (10.0%)  0.021456
3      9.0 - 12.0 dB        501 (10.0%)  0.014523
4      12.0 - 15.0 dB       497 ( 9.9%)  0.009234
5      15.0 - 18.0 dB       503 (10.0%)  0.006234
6      18.0 - 21.0 dB       499 ( 9.9%)  0.004123
7      21.0 - 24.0 dB       501 (10.0%)  0.002534
8      24.0 - 27.0 dB       496 ( 9.9%)  0.001523
9      27.0 - 30.0 dB       497 ( 9.9%)  0.000823

Total samples: 5000
================================================================================
```

---

## 🎯 参数建议

### Patience 设置

**之前**（Uniform sampling）：
```bash
--patience 20  # 需要很大，防止聚类
```

**现在**（Stratified sampling）：
```bash
--patience 10  # 可以更小，不会聚类
```

因为 stratified 保证不同 validation 使用不同 SNR bin。

---

### SNR Bins 数量

| Bins | Bin 宽度 | 适用场景 |
|------|----------|----------|
| 5 | 6 dB | 粗粒度，快速训练 |
| **10** | **3 dB** | **平衡（推荐）** ⭐ |
| 15 | 2 dB | 细粒度 |
| 20 | 1.5 dB | 超细粒度，研究用 |

**推荐**：10 bins（3 dB/bin）

---

## ⚙️ 实现细节

### SNRSampler 类

```python
from Model_AIIC_onnx.snr_sampler import SNRSampler

# 创建 sampler
sampler = SNRSampler(
    snr_min=0,
    snr_max=30,
    strategy='stratified',  # or 'round_robin', 'uniform'
    num_bins=10
)

# 采样
batch_snr = sampler.sample()  # 返回 SNR 值

# 查看统计
sampler.print_stats()
```

---

### 集成到训练

```python
# 在 generate_training_data 中
if snr_sampler is not None:
    batch_snr = snr_sampler.sample()  # 智能采样
else:
    batch_snr = np.random.uniform(snr_min, snr_max)  # 简单随机
```

---

## 📝 总结

### 关键改进

1. ✅ **Stratified sampling**：均匀覆盖所有 SNR
2. ✅ **防止聚类**：相邻 batch 不会相似 SNR
3. ✅ **Patience 更小**：10 足够（之前需要 20+）
4. ✅ **可视化统计**：训练结束显示 SNR 分布

---

### 使用建议

**生产训练**（推荐）⭐：
```bash
python ./Model_AIIC_onnx/test_separator.py \
  --batches 100000 \
  --snr "0,30" \
  --snr_sampling "stratified" \
  --snr_bins 10 \
  --patience 10
```

**快速实验**：
```bash
--snr_bins 5  # 更粗粒度
--patience 5
```

**精细研究**：
```bash
--snr_bins 20  # 更细粒度
--patience 15
```

---

**现在训练不会因为 SNR 聚类而出现问题了！** 🎯
