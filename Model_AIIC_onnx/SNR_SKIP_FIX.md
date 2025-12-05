# SNR-Aware Backward Skip 修正说明

## 🐛 发现的问题

### 问题

**错误实现**：
```python
loss = calculate_loss(h_pred, h_targets, batch_snr, loss_type)
skip_backward = loss.item() < snr_noise_floor_linear  # ❌ 错误！
```

**为什么错误？**

不同的 `loss_type` 会变换 NMSE：

1. **`nmse`**：`loss = nmse`（没问题）
2. **`normalized`**：`loss = nmse / snr_linear`（归一化，扭曲了 NMSE）
3. **`log`**：`loss = -10*log10(nmse)`（对数空间，完全不同尺度）
4. **`weighted`**：`loss = nmse * weight(snr)`（加权，扭曲了 NMSE）

**后果**：
- 使用 `log` loss：`loss` 是负数（dB），而 `snr_noise_floor_linear` 是小于 1 的正数 → **永远不会 skip** ❌
- 使用 `normalized` loss：`loss` 被 SNR 归一化，与 `snr_noise_floor_linear` 不可比 ❌
- 使用 `weighted` loss：`loss` 被加权，无法正确判断是否收敛 ❌

---

## ✅ 正确实现

### 原则

**用原始 NMSE 判断是否 skip，而非变换后的 loss**

```python
# 1. 先计算原始 NMSE（用于判断 skip）
mse = (h_pred - h_targets).abs().pow(2).mean()
signal_power = h_targets.abs().pow(2).mean()
nmse = mse / (signal_power + 1e-10)  # 原始 NMSE

# 2. 再计算优化用的 loss（可能是变换的）
loss = calculate_loss(h_pred, h_targets, batch_snr, loss_type)

# 3. 用 NMSE 判断是否跳过
snr_noise_floor_linear = 1.0 / (10 ** ((batch_snr + 5) / 10))
skip_backward = nmse.item() < snr_noise_floor_linear  # ✅ 正确！

# 4. 用 loss 优化（如果不跳过）
if not skip_backward:
    loss.backward()  # 用变换后的 loss 优化
    optimizer.step()
```

---

## 📊 对比

### 示例：SNR = 20 dB, NMSE = 0.005

| Loss Type | NMSE | Loss 值 | SNR Floor | Skip (错误) | Skip (正确) |
|-----------|------|---------|-----------|-------------|-------------|
| `nmse` | 0.005 | 0.005 | 0.00316 | ❌ No | ✅ No |
| `normalized` | 0.005 | 0.00005 | 0.00316 | ✅ Yes (错!) | ✅ No |
| `log` | 0.005 | -23 dB | 0.00316 | ❌ No (错!) | ✅ No |
| `weighted` | 0.005 | 0.05 | 0.00316 | ❌ No | ✅ No |

**正确实现**：所有情况都用 NMSE = 0.005 与 0.00316 比较，结果一致 ✅

---

## 🎯 训练输出变化

### 之前（错误）

```
Batch 20, SNR:20.0dB, Loss: -23.45 dB, Throughput: 42000 samples/s
                            ↑ 显示的是 loss（log 空间）
```

无法看出实际的 NMSE 是多少。

---

### 现在（正确）

```
Batch 20, SNR:20.0dB, NMSE: 0.0050 (-23.01 dB), Loss(log): -23.45, Throughput: 42000 samples/s
                      ↑ 显示原始 NMSE          ↑ 显示优化用的 loss
```

**对于 `nmse` loss type**（最简单）：
```
Batch 20, SNR:20.0dB, NMSE: 0.0050 (-23.01 dB), Throughput: 42000 samples/s
```
（不显示 loss，因为 loss == NMSE）

---

## 📈 TensorBoard 改进

### 之前

只记录 `Loss/train`（可能是变换后的）

### 现在

同时记录：
- `NMSE/train`：原始 NMSE（线性）⭐
- `NMSE/train_db`：原始 NMSE（dB）⭐
- `Loss/train`：优化用的 loss
- `Loss/train_db`：优化用的 loss（dB）
- `SNR/batch_snr`：当前 batch 的 SNR
- `Skip/backward_skipped`：是否跳过反向传播（1=skip, 0=not skip）

**好处**：
- ✅ 可以直接看 NMSE 趋势（不受 loss type 影响）
- ✅ 可以对比不同 loss type 的效果
- ✅ 可以看 skip 率（高 SNR 应该 skip 更多）

---

## 🔍 验证

### 测试场景：SNR = 20 dB, 不同 loss types

假设模型已收敛：NMSE = 0.001（-30 dB）

**SNR 噪声底限**（+5 dB margin）：
```
SNR = 20 dB → Floor = 1/(10^2.5) = 0.00316 (-25 dB)
```

| Loss Type | NMSE | Loss | Skip (之前) | Skip (现在) | 正确性 |
|-----------|------|------|-------------|-------------|--------|
| `nmse` | 0.001 | 0.001 | ✅ Yes | ✅ Yes | ✅ |
| `normalized` | 0.001 | 0.00001 | ✅ Yes | ✅ Yes | ✅ |
| `log` | 0.001 | -30 dB | ❌ No | ✅ Yes | ✅ 修正！|
| `weighted` | 0.001 | 0.01 | ❌ No | ✅ Yes | ✅ 修正！|

**现在所有 loss type 都能正确 skip！**

---

## 💡 为什么要分离 NMSE 和 loss？

### NMSE 的作用
- 衡量实际性能（信号质量）
- 与 SNR 直接相关
- 用于判断是否收敛

### Loss 的作用
- 指导优化方向
- 可以变换以改善收敛
- 用于反向传播

### 两者应该独立
```python
# NMSE：判断 "做得怎么样"
if nmse < threshold:
    skip  # 已经足够好了

# Loss：指导 "怎么做得更好"
loss.backward()  # 用变换后的 loss 优化梯度
```

---

## 📝 总结

### 关键修改

1. ✅ **分离计算**：先算 NMSE，再算 loss
2. ✅ **用 NMSE 判断 skip**：`skip = nmse < floor`
3. ✅ **用 loss 优化**：`loss.backward()`
4. ✅ **显示两者**：打印 NMSE 和 loss（如果不同）
5. ✅ **记录两者**：TensorBoard 同时记录

### 性能影响

- 计算开销：**可忽略**（只是多一次 MSE 计算）
- 正确性：**显著提升**（修正了 skip 逻辑）
- 可调试性：**大幅提升**（可以看到 NMSE 和 loss）

---

**现在 SNR-Aware Skip 在所有 loss type 下都能正确工作了！** ✅
