# 参数量差异分析：为什么不是 2 倍关系？

## 🔍 问题

```bash
# Model_AIIC (复数版)
python -c "from Model_AIIC.channel_separator import ResidualRefinementSeparator; 
a = ResidualRefinementSeparator(num_ports=4, num_stages=2, hidden_dim=64, 
num_sub_stages=2, share_weights_across_stages=True); 
print(sum(p.numel() for p in a.parameters()))"
# 输出: 52,320

# Model_AIIC_onnx (实数版)
python -c "from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal; 
a = ResidualRefinementSeparatorReal(num_ports=4, num_stages=2, hidden_dim=64, 
num_sub_stages=2, share_weights_across_stages=True); 
print(sum(p.numel() for p in a.parameters()))"
# 输出: 46,176

# 比例: 52320 / 46176 = 1.133x （而不是 2.0x）
```

**为什么不是 2 倍关系？**

---

## 📊 详细参数构成分析

### Model_AIIC (复数版) - 每个 Port 的 MLP

**结构**：实部和虚部用**两个独立的 MLP**

```
ComplexMLP:
  ├── mlp_real (独立参数)
  │   ├── Linear(24 -> 64):  weight(64,24) + bias(64) = 1,600
  │   ├── Linear(64 -> 64):  weight(64,64) + bias(64) = 4,160
  │   └── Linear(64 -> 12):  weight(12,64) + bias(12) = 780
  └── mlp_imag (独立参数)
      ├── Linear(24 -> 64):  weight(64,24) + bias(64) = 1,600
      ├── Linear(64 -> 64):  weight(64,64) + bias(64) = 4,160
      └── Linear(64 -> 12):  weight(12,64) + bias(64) = 780

每个 MLP: (1,600 + 4,160 + 780) × 2 = 13,080 params
4 个 ports: 13,080 × 4 = 52,320 params
```

**关键点**：
- 输入层接收 `[real, imag]` 拼接，维度 = **24** (12×2)
- 输出层只输出实部或虚部，维度 = **12**
- 实部和虚部的 MLP **完全独立**

---

### Model_AIIC_onnx (实数版) - 每个 Port 的 MLP

**结构**：实部和虚部用**一个共享的 ComplexLinearReal**

```
ComplexMLPReal (使用 ComplexLinearReal):
  ├── fc1: ComplexLinearReal(12 -> 64)
  │   ├── weight_real(64,12) = 768
  │   ├── weight_imag(64,12) = 768
  │   ├── bias_real(64) = 64
  │   └── bias_imag(64) = 64
  │   Total: 1,664
  │
  ├── hidden[0]: ComplexLinearReal(64 -> 64)
  │   ├── weight_real(64,64) = 4,096
  │   ├── weight_imag(64,64) = 4,096
  │   ├── bias_real(64) = 64
  │   └── bias_imag(64) = 64
  │   Total: 8,320
  │
  └── fc_out: ComplexLinearReal(64 -> 12)
      ├── weight_real(12,64) = 768
      ├── weight_imag(12,64) = 768
      ├── bias_real(12) = 12
      └── bias_imag(12) = 12
      Total: 1,560

每个 MLP: 1,664 + 8,320 + 1,560 = 11,544 params
4 个 ports: 11,544 × 4 = 46,176 params
```

**关键点**：
- 输入直接是复数形式（实数张量表示），维度 = **12**（不是 24）
- 输出也是复数形式，维度 = **12**
- 用 ComplexLinearReal 实现复数线性变换

---

## 🔬 为什么不是 2 倍关系？

### 原因：输入/输出维度不同

#### Model_AIIC (复数版)
```python
# 每个独立的 MLP
Input:  [real, imag] 拼接 -> 维度 24
Output: real 或 imag      -> 维度 12

# 输入层参数
mlp_real: Linear(24 -> 64) = 64×24 + 64 = 1,600
mlp_imag: Linear(24 -> 64) = 64×24 + 64 = 1,600
Total: 3,200

# 输出层参数
mlp_real: Linear(64 -> 12) = 12×64 + 12 = 780
mlp_imag: Linear(64 -> 12) = 12×64 + 12 = 780
Total: 1,560
```

#### Model_AIIC_onnx (实数版)
```python
# ComplexLinearReal
Input:  复数 -> 维度 12 (但有 real 和 imag 两个权重矩阵)
Output: 复数 -> 维度 12

# 输入层参数
fc1: ComplexLinearReal(12 -> 64)
  = (64×12 + 64×12 + 64 + 64) = 1,664

# 输出层参数
fc_out: ComplexLinearReal(64 -> 12)
  = (12×64 + 12×64 + 12 + 12) = 1,560
```

---

## 📈 逐层对比

| 层 | Model_AIIC (复数版) | Model_AIIC_onnx (实数版) | 比例 |
|----|---------------------|--------------------------|------|
| **输入层** | 3,200 (两个 24→64) | 1,664 (一个复数 12→64) | 1.92x |
| **隐藏层** | 8,320 (两个 64→64) | 8,320 (一个复数 64→64) | 1.00x |
| **输出层** | 1,560 (两个 64→12) | 1,560 (一个复数 64→12) | 1.00x |
| **总计** | 13,080 | 11,544 | 1.133x |

**关键发现**：
- **隐藏层和输出层**：参数量完全相同（因为维度相同）
- **输入层**：AIIC 版本多了 1,536 个参数（因为输入维度是 24 而不是 12）

---

## 🧮 数学解释

### AIIC 版本输入层
```
两个独立 MLP，每个接收拼接的 [real, imag]:
  mlp_real: (24 × 64 + 64) = 1,600
  mlp_imag: (24 × 64 + 64) = 1,600
  Total: 3,200
```

### ONNX 版本输入层
```
ComplexLinearReal 直接处理复数:
  weight_real: 12 × 64 = 768
  weight_imag: 12 × 64 = 768
  bias_real: 64
  bias_imag: 64
  Total: 1,664
```

**差异来源**：
```
AIIC 多出的参数 = 3,200 - 1,664 = 1,536

这是因为:
  AIIC 输入: 2 × (24 × 64) = 3,072
  ONNX 输入: 2 × (12 × 64) = 1,536
  差异: 3,072 - 1,536 = 1,536 (每个 port)
```

---

## 💡 结论

### 为什么不是 2 倍关系？

**答案**：虽然 AIIC 版本用了两个独立的 MLP，但：

1. **AIIC 版本的输入维度是 ONNX 的 2 倍** (24 vs 12)
   - 因为它把 real 和 imag **拼接**成一个实数向量
   - 而 ONNX 版本直接用复数表示（分开的 real 和 imag 权重）

2. **隐藏层和输出层参数量相同**
   - 两者的隐藏层都是 64→64 的复数线性变换
   - 输出层都是 64→12 的复数线性变换

3. **实际比例 = 1.133x**
   - 主要差异在输入层（3,200 vs 1,664）
   - 每个 port 多了 1,536 个参数

### 参数效率对比

| 版本 | 每 Port 参数量 | 输入表示 | 参数效率 |
|------|---------------|---------|---------|
| **AIIC** | 13,080 | 拼接 [R, I] → 24维 | 较低（输入冗余） |
| **ONNX** | 11,544 | 复数 → 12维 | 较高（紧凑表示） |

**推荐**：ONNX 版本更参数高效，因为它用复数线性层直接处理复数，避免了输入拼接的冗余。

---

## 🔢 完整计算验证

### AIIC 版本 (share_weights=True, 4 ports)
```
每个 port 的 ComplexMLP:
  mlp_real:
    - Linear(24->64): 24×64 + 64 = 1,600
    - Linear(64->64): 64×64 + 64 = 4,160
    - Linear(64->12): 64×12 + 12 = 780
    Subtotal: 6,540
  
  mlp_imag:
    - Linear(24->64): 24×64 + 64 = 1,600
    - Linear(64->64): 64×64 + 64 = 4,160
    - Linear(64->12): 64×12 + 12 = 780
    Subtotal: 6,540
  
  Per port: 6,540 × 2 = 13,080
  4 ports: 13,080 × 4 = 52,320 ✅
```

### ONNX 版本 (share_weights=True, 4 ports)
```
每个 port 的 ComplexMLPReal:
  fc1 (ComplexLinearReal 12->64):
    - weight_real(64,12) + weight_imag(64,12) = 1,536
    - bias_real(64) + bias_imag(64) = 128
    Subtotal: 1,664
  
  hidden[0] (ComplexLinearReal 64->64):
    - weight_real(64,64) + weight_imag(64,64) = 8,192
    - bias_real(64) + bias_imag(64) = 128
    Subtotal: 8,320
  
  fc_out (ComplexLinearReal 64->12):
    - weight_real(12,64) + weight_imag(12,64) = 1,536
    - bias_real(12) + bias_imag(12) = 24
    Subtotal: 1,560
  
  Per port: 1,664 + 8,320 + 1,560 = 11,544
  4 ports: 11,544 × 4 = 46,176 ✅
```

**比例验证**: 52,320 / 46,176 = **1.133x** ✅

