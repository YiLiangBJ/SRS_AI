# MATLAB 完全兼容改造计划

## 🎯 目标
移除所有 MATLAB 不支持的算子，同时保证**数学完全等价**。

---

## 📊 当前问题分析

### MATLAB 报告的不支持算子

| 算子 | 数量 | 来源 | 是否必须移除 |
|------|------|------|-------------|
| `Slice` | 80 | 动态切片、`torch.stack` | ✅ 是 |
| `Unsqueeze` | 12 | `.unsqueeze()` | ✅ 是 |
| `ReduceMean` | 1 | `.mean()` | ⚠️ 需研究 |
| `ReduceSum` | 4 | `.sum()` | ⚠️ 需研究 |
| `Pow` | 2 | `x.pow(2)` | ⚠️ 需研究 |
| `Sqrt` | 1 | `.sqrt()` | ⚠️ 需研究 |
| `Expand` | 1 | 残留 | ✅ 是 |
| `Tile` | 1 | `repeat` 生成 | ⚠️ 可能无法避免 |
| `Gather` | 8 | `torch.stack` 内部 | ✅ 是 |
| `ConstantOfShape` | 1 | 初始化 | ✅ 是 |
| `Div/MatMul/Sub` | 75 | 限制：需要一个常量 | ⚠️ 需检查 |

---

## 🔧 改造计划（严格按你的要求）

### 第一步：移除所有动态切片 ✅

**当前代码位置**：
1. 能量归一化：`torch.chunk(y_stacked, 2, dim=-1)` - 已经是 chunk，但仍生成 Split
2. 残差计算：`torch.chunk(features, 2, dim=-1)` - 已经是 chunk

**问题**：`torch.chunk` 虽然是"常量"分割，但 ONNX 中仍然生成 `Split` 算子，MATLAB 不支持。

**解决方案**：
- 改为在模型输入时就分离实部和虚部
- 输入格式从 `(B, L*2)` 改为两个独立输入 `(B, L)` 和 `(B, L)`
- 或者在模型内部手动分离，避免使用 chunk/split

**等价性保证**：数学完全等价，只是改变了数据表示方式。

**⚠️ 请示**：
- **方案 A**：输入改为两个张量 `y_real`, `y_imag`，模型内部不需要切片
- **方案 B**：保持单输入 `y_stacked`，但用索引手动分离（可能仍有 Slice）

**我的建议**：方案 A，因为更彻底且符合你"移除所有动态切片"的要求。

---

### 第二步：移除所有 unsqueeze ✅

**当前使用 unsqueeze 的地方**：
1. 特征初始化：`y_normalized.unsqueeze(1).repeat(1, P, 1)` → `(B, P, L*2)`
2. 残差添加：`residual.unsqueeze(1)` → `(B, 1, L*2)`
3. 能量恢复：`y_energy.unsqueeze(1)` → `(B, 1, 1)`

**解决方案**：
- 初始就使用 3D 张量 `(B, P, L*2)`
- 不要从 2D 升到 3D，而是一直保持 3D

**具体实现**：
```python
# 不要这样（当前）
features = y_normalized.unsqueeze(1).repeat(1, P, 1)  # ❌ unsqueeze

# 改为这样
# 在初始化时就创建正确形状
features = y_normalized[:, None, :].expand(B, P, L*2).contiguous()
# 或者直接 reshape
features = y_normalized.view(B, 1, L*2).expand(B, P, L*2).contiguous()
```

**等价性保证**：完全等价，只是避免了 unsqueeze 算子。

**⚠️ 但是问题**：`expand` 可能仍然生成 ONNX 的 Expand 算子。

**更彻底的方案**：
```python
# 完全避免 unsqueeze 和 expand
features = torch.zeros(B, P, L*2, device=y_normalized.device, dtype=y_normalized.dtype)
for p in range(P):
    features[:, p, :] = y_normalized
```

**⚠️ 请示**：这个方案会引入显式循环，虽然数学等价，但代码风格不同。你接受吗？

---

### 第三步：移除能量归一化（放到模型外）✅

**当前代码**（在 forward 内）：
```python
if self.normalize_energy:
    y_R, y_I = torch.chunk(y_stacked, 2, dim=-1)
    y_mag_sq = y_R.pow(2) + y_I.pow(2)
    y_energy = y_mag_sq.mean(dim=-1, keepdim=True).sqrt()
    y_normalized = y_stacked / y_energy
```

**改造方案**：
```python
# 模型只接受已归一化的输入
def forward(self, y_normalized):
    # 直接使用输入，不做归一化
    features = ...
    return features  # 返回未恢复能量的结果
```

**外部使用**（PyTorch）：
```python
# 在模型外做归一化
y_R, y_I = torch.chunk(y_stacked, 2, dim=-1)
y_energy = (y_R**2 + y_I**2).mean(dim=-1, keepdim=True).sqrt()
y_normalized = y_stacked / y_energy

# 推理
h_normalized = model(y_normalized)

# 恢复能量
h = h_normalized * y_energy.unsqueeze(1)
```

**MATLAB 中**：
```matlab
% 归一化
y_energy = sqrt(mean(y_complex.^2));
y_normalized = y_stacked / y_energy;

% 推理
h_normalized = predict(net, y_normalized);

% 恢复
h = h_normalized * y_energy;
```

**等价性保证**：完全等价，只是把归一化移到模型外。

**优点**：
- 移除 `Pow`, `ReduceMean`, `Sqrt`, `Div` 算子
- 模型更简洁
- 符合你的要求

**⚠️ 影响**：
- 导出的 ONNX 模型输入必须是已归一化的
- 需要更新所有使用代码
- 需要重新训练吗？**不需要**，权重完全兼容

**⚠️ 请示**：这个改动会影响模型接口，你确认接受吗？

---

### 第四步：研究 ReduceMean/ReduceSum ⚠️

**当前使用**：
1. `ReduceMean(1次)`：能量归一化中的 `.mean()`
2. `ReduceSum(4次)`：残差计算中的 `.sum(dim=1)`

**第三步完成后**：
- `ReduceMean` 会被移除（能量归一化在外部）
- `ReduceSum` 还剩下残差计算中的使用

**ReduceSum 的位置**：
```python
y_recon_R = features_R.sum(dim=1)  # (B, P, L) -> (B, L)
y_recon_I = features_I.sum(dim=1)  # (B, P, L) -> (B, L)
```

**MATLAB 不支持的原因**：可能是多维度的 ReduceSum。

**解决方案 A**：手动循环求和
```python
# 不用 sum(dim=1)
y_recon_R = features_R[:, 0, :]  # 先取第一个端口
for p in range(1, P):
    y_recon_R = y_recon_R + features_R[:, p, :]
```

**解决方案 B**：用矩阵乘法替代求和
```python
# sum(dim=1) 等价于与全1向量相乘
ones = torch.ones(1, P, 1)  # (1, P, 1)
y_recon_R = (features_R * ones).sum(dim=1)  # 可能还是不行

# 或者用 matmul
# features_R: (B, P, L)
# ones: (P, 1)
# result: (B, L)
ones = torch.ones(P, 1, device=features_R.device)
y_recon_R = torch.matmul(features_R.transpose(1, 2), ones).squeeze(-1)
```

**⚠️ 问题**：这些方案可能仍然生成不支持的算子。

**最彻底方案**：完全展开循环
```python
# 完全避免 sum/reduce 算子
y_recon_R = features_R[:, 0, :]
for p in range(1, self.num_ports):
    y_recon_R = y_recon_R + features_R[:, p, :]
```

**等价性保证**：数学完全等价。

**⚠️ 请示**：
1. 你接受显式循环吗？
2. 需要我先测试哪个方案能避免 ReduceSum？

---

## 🎯 改造后的模型接口

### 当前接口
```python
# 输入：(B, L*2) - [real; imag]
# 输出：(B, P, L*2) - 已恢复能量
model(y_stacked) -> h_stacked
```

### 改造后接口（提议）

**方案 A**：双输入
```python
# 输入：两个 (B, L) 张量
# 输出：两个 (B, P, L) 张量 - 未恢复能量
model(y_real, y_imag) -> (h_real, h_imag)
```

**方案 B**：单输入（保守）
```python
# 输入：(B, L*2) - [real; imag] 已归一化
# 输出：(B, P, L*2) - 未恢复能量
model(y_normalized) -> h_normalized
```

**⚠️ 请示**：你倾向哪个方案？

---

## 📋 改造步骤总结

基于你的四步要求，我建议的顺序：

### ✅ 第一阶段：移除能量归一化（最安全）
1. 将 `normalize_energy` 相关代码移到模型外
2. 验证等价性
3. 重新导出 ONNX，检查算子列表

**预期效果**：移除 `Pow`, `ReduceMean`, `Sqrt`, `Div`

### ✅ 第二阶段：移除 unsqueeze（需要你确认方案）
1. 改造特征初始化逻辑
2. 避免 2D→3D 的升维操作
3. 验证等价性

**预期效果**：移除 `Unsqueeze`, `Expand`

### ✅ 第三阶段：移除动态切片（最复杂）
1. 改为双输入或手动分离
2. 避免所有 chunk/split 操作
3. 验证等价性

**预期效果**：移除 `Slice`, `Split`, `Gather`, `ConstantOfShape`

### ✅ 第四阶段：移除 ReduceSum
1. 改为显式循环或矩阵乘法
2. 验证等价性
3. 最终测试

**预期效果**：移除 `ReduceSum`

---

## ⚠️ 现在需要你的决策

**问题 1**：第三步（移除能量归一化），你确认接受吗？
- 这会改变模型接口
- 但数学完全等价
- 可以立即开始

**问题 2**：输入格式倾向？
- 方案 A：双输入 `(y_real, y_imag)`
- 方案 B：单输入 `y_stacked`（已归一化）

**问题 3**：是否接受显式循环？
- 用于替代 `sum(dim=1)`
- 数学等价但代码不优雅

**问题 4**：是否需要保持训练兼容？
- 如果是，需要保留原版作为训练版本
- 创建专门的"MATLAB导出版本"

---

**我的建议**：
1. 先做第一阶段（移除能量归一化）- 最安全
2. 导出测试，看看还剩什么问题
3. 再决定后续步骤

**请你明确告诉我**：
1. 是否同意第一阶段改造？
2. 选择方案 A 还是 B？
3. 是否接受显式循环？

我会严格按照你的指示进行，**不会擅自改动**。
