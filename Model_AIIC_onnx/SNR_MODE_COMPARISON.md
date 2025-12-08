# SNR 模式对比：Per-Batch vs Per-Sample

## 🎯 两种 SNR 模式

### 模式 1：Per-Batch SNR（默认，推荐）

**特点**：每个 batch 的所有 sample 使用**相同 SNR**

```python
# Batch 1: 所有 4096 个 sample 都是 SNR=5dB
# Batch 2: 所有 4096 个 sample 都是 SNR=22dB
# Batch 3: 所有 4096 个 sample 都是 SNR=13dB
```

**梯度行为**：
```
∇L_batch1 = 纯粹的 5dB 梯度
∇L_batch2 = 纯粹的 22dB 梯度
→ 模型学会不同 SNR 下的不同策略
```

**适用场景**：
- ✅ 需要模型在**不同 SNR 下有不同行为**
- ✅ 低 SNR：激进去噪，高 SNR：精细分离
- ✅ 想要**精确控制 SNR 覆盖**（通过 stratified sampling）

**命令**：
```bash
# 默认（不需要加参数）
python Model_AIIC_onnx/test_separator.py \
  --batches 100000 \
  --batch_size 4096 \
  --snr "0,30" \
  --snr_sampling "stratified" \
  --snr_bins 30
```

---

### 模式 2：Per-Sample SNR（旧行为）

**特点**：每个 batch 内的每个 sample 使用**不同 SNR**

```python
# Batch 1: 4096 个 sample，SNR = [3dB, 28dB, 12dB, 18dB, ...]
# → 平均 SNR ≈ 15dB
# Batch 2: 4096 个 sample，SNR = [7dB, 21dB, 5dB, 29dB, ...]
# → 平均 SNR ≈ 15dB
```

**梯度行为**：
```
∇L_batch1 = (∇L_3dB + ∇L_28dB + ∇L_12dB + ...) / 4096
          ≈ ∇L_15dB  ← 平均梯度
∇L_batch2 ≈ ∇L_15dB
→ 几乎所有梯度都针对平均 SNR
```

**适用场景**：
- ✅ 想要模型学习**SNR 不变特征**
- ✅ 追求在**平均 SNR** 下的最佳性能
- ✅ 训练更稳定（梯度方差小）
- ⚠️ 但可能无法充分学习极端 SNR（0dB, 30dB）

**命令**：
```bash
# 需要加 --snr_per_sample 参数
python Model_AIIC_onnx/test_separator.py \
  --batches 100000 \
  --batch_size 4096 \
  --snr "0,30" \
  --snr_per_sample  # ⭐ 启用 per-sample 模式
```

---

## 📊 详细对比

| 特性 | Per-Batch SNR | Per-Sample SNR |
|------|--------------|----------------|
| **每个 batch 的 SNR** | 相同 | 不同（平均值） |
| **梯度针对** | 特定 SNR 点 | 平均 SNR |
| **SNR 覆盖** | 精确控制 | 隐式覆盖 |
| **学习内容** | 不同 SNR 的不同策略 | SNR 不变特征 |
| **极端 SNR 性能** | 好 | 可能较差 |
| **平均 SNR 性能** | 好 | 可能更好 |
| **训练稳定性** | 中等 | 更稳定 |
| **推荐场景** | 大范围 SNR 适配 | 特定 SNR 优化 |

---

## 🔬 实验对比

### 实验设置
```bash
# 共同设置
--batches 10000
--batch_size 2048
--snr "0,30"
--stages "2"
--activation_type "relu"
--early_stop "0.0001"
```

### 结果预测

**Per-Batch SNR**：
```
SNR=0dB:  NMSE = 0.05 (-13 dB)  ← 学会激进去噪
SNR=15dB: NMSE = 0.002 (-27 dB)
SNR=30dB: NMSE = 0.0001 (-40 dB) ← 学会精细分离
```

**Per-Sample SNR**：
```
SNR=0dB:  NMSE = 0.08 (-11 dB)  ← 较差（未充分学习）
SNR=15dB: NMSE = 0.0015 (-28 dB) ← 更好（主要优化目标）
SNR=30dB: NMSE = 0.0002 (-37 dB) ← 较差（未充分学习）
```

---

## 💡 推荐使用

### 推荐：Per-Batch SNR（默认）⭐⭐⭐⭐⭐

**原因**：
1. ✅ 明确的 SNR 覆盖控制
2. ✅ 每个 SNR 点都得到充分训练
3. ✅ 结合 stratified sampling 效果更好
4. ✅ 适合大范围 SNR（0-30 dB）

**使用**：
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 100000 \
  --batch_size 2048 \
  --snr "0,30" \
  --snr_sampling "stratified" \
  --snr_bins 30 \
  --stages "2" \
  --save_dir "./models"
```

---

### 可选：Per-Sample SNR（如果需要）⭐⭐⭐

**适用场景**：
- 你只关心**平均性能**（如 15 dB）
- 极端 SNR 性能不重要
- 想要更稳定的训练

**使用**：
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 50000 \  # 可以少一些（不需要覆盖所有 SNR）
  --batch_size 4096 \  # 可以大一些（更稳定）
  --snr "0,30" \
  --snr_per_sample \  # ⭐ 启用
  --stages "2" \
  --save_dir "./models"
```

---

## 🎯 如何选择？

### 问题 1：你的目标是什么？

**A. 全范围 SNR 性能** → Per-Batch SNR ⭐⭐⭐⭐⭐
- 0-30 dB 都要好
- 低 SNR 和高 SNR 都重要

**B. 特定 SNR 性能** → Per-Sample SNR ⭐⭐⭐
- 只关心平均 SNR（如 15 dB）
- 极端 SNR 不重要

---

### 问题 2：你有多少训练时间？

**A. 充足（> 10 小时）** → Per-Batch SNR ⭐⭐⭐⭐⭐
- 可以训练更多 batches（100K-1M）
- 充分覆盖所有 SNR

**B. 有限（< 5 小时）** → Per-Sample SNR ⭐⭐⭐
- 更快收敛到平均性能
- 需要的 batches 更少（50K）

---

### 问题 3：你观察到什么问题？

**A. 低 SNR 性能差** → Per-Batch SNR + 增加 batches
- 确保低 SNR 得到足够训练

**B. 训练不稳定** → Per-Sample SNR
- 更稳定的梯度

**C. 高 SNR 性能差** → Per-Batch SNR + 增加 batches
- 确保高 SNR 得到足够训练

---

## 📋 总结

| 模式 | 默认 | 推荐场景 | 优点 | 缺点 | 推荐度 |
|------|------|---------|------|------|--------|
| **Per-Batch** | ✅ 是 | 大范围 SNR | 精确控制，全覆盖 | 需要更多 batches | ⭐⭐⭐⭐⭐ |
| **Per-Sample** | ❌ 否 | 平均 SNR | 更稳定，快收敛 | 极端 SNR 较差 | ⭐⭐⭐ |

**最终建议**：
1. **默认使用 Per-Batch**（不加 `--snr_per_sample`）
2. **如果效果不好**，尝试 Per-Sample（加 `--snr_per_sample`）
3. **对比两者**，选择效果更好的

🎯
