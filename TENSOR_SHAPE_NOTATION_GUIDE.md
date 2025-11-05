# 📐 张量形状标注说明

## 🎯 新增功能

模型结构分析工具现在会显示：
1. **每个模块的输入输出形状**
2. **每个参数张量的维度含义**
3. **维度符号说明**

## 📊 输出示例

### 示例1：卷积层 (Conv1d)

```
├─ conv_real: Conv1d

    【张量形状 Tensor Shapes】
      Input:  (B, 2, L)          ← 输入形状
      Output: (B, 8, L')         ← 输出形状
      说明: B=batch, L=length    ← 维度含义
    
    【参数统计】
      • weight: (8, 2, 3)  # (out_channels, in_channels, kernel_size) = 48 (✓可训练)
                ↑  ↑  ↑
                │  │  └─ kernel_size=3
                │  └───── in_channels=2
                └──────── out_channels=8
```

**解读**：
- **Input**: `(B, 2, L)` = (批次大小, 2个输入通道, 序列长度L)
- **Output**: `(B, 8, L')` = (批次大小, 8个输出通道, 新的序列长度L')
- **weight**: `(8, 2, 3)` = (8个输出通道, 2个输入通道, 3的卷积核大小)

### 示例2：批归一化层 (BatchNorm1d)

```
├─ bn_real: BatchNorm1d

    【张量形状 Tensor Shapes】
      Input:  (B, 8, L)
      Output: (B, 8, L)          ← BatchNorm不改变形状
      说明: B=batch, L=length
    
    【参数统计】
      • weight: (8,)  # (num_features,) = 8 (✓可训练)
      • bias:   (8,)  # (num_features,) = 8 (✓可训练)
    
    【缓冲区】(非可训练)
      • running_mean: (8,) = 8
      • running_var:  (8,) = 8
```

**解读**：
- **Input/Output**: `(B, 8, L)` = (批次, 8个特征, 长度) - 形状不变
- **weight/bias**: `(8,)` = 每个特征一个参数
- **running_mean/var**: 存储训练时的统计信息

### 示例3：激活函数 (ComplexModReLU)

```
├─ activation1: ComplexModReLU

    【张量形状 Tensor Shapes】
      Input:  (*, ...)
      Output: (*, ...)
      说明: shape unchanged      ← 元素级操作，形状不变
    
    【参数统计】
      • bias: (8,) = 8 (✓可训练)
```

**解读**：
- `(*, ...)` 表示任意形状
- 元素级操作保持输入输出形状一致

## 📐 维度符号对照表

| 符号 | 含义 | 说明 |
|------|------|------|
| **B** | Batch | 批次大小，通常是第一个维度 |
| **C** | Channels | 通道数（卷积、BatchNorm等） |
| **L** | Length | 序列长度（1D卷积） |
| **L'** | Length (new) | 卷积/池化后的新序列长度 |
| **H** | Height | 图像高度（2D卷积） |
| **W** | Width | 图像宽度（2D卷积） |
| **H'** | Height (new) | 卷积/池化后的新高度 |
| **W'** | Width (new) | 卷积/池化后的新宽度 |
| **F** | Features | 特征维度（全连接层） |
| **E** | Embedding | 嵌入维度 |
| *** | Wildcard | 任意形状 |

## 🔍 参数维度标注

### Conv1d 参数

```python
weight: (out_channels, in_channels, kernel_size)
bias:   (out_channels,)

示例：weight: (8, 2, 3)  # 8个输出通道，2个输入通道，卷积核大小3
```

### Conv2d 参数

```python
weight: (out_ch, in_ch, kH, kW)
bias:   (out_channels,)

示例：weight: (64, 3, 7, 7)  # 64个输出通道，3个输入通道(RGB)，7x7卷积核
```

### Linear 参数

```python
weight: (out_features, in_features)
bias:   (out_features,)

示例：weight: (10, 128)  # 128维输入 → 10维输出
```

### BatchNorm 参数

```python
weight: (num_features,)  # γ (scale)
bias:   (num_features,)  # β (shift)

示例：weight: (64,)  # 对64个特征进行归一化
```

### Embedding 参数

```python
weight: (num_embeddings, embedding_dim)

示例：weight: (10000, 300)  # 10000个词，每个300维
```

## 📊 完整示例：ComplexConv1d

```
├─ conv1: ComplexConv1d
│   ├─ conv_real: Conv1d
│   │   
│   │   【张量形状 Tensor Shapes】
│   │     Input:  (B, 2, L)
│   │     Output: (B, 8, L')
│   │     说明: B=batch, L=length
│   │   
│   │   【参数统计】
│   │     • weight: (8, 2, 3)  # (out_channels, in_channels, kernel_size) = 48
│   │   
│   └─ conv_imag: Conv1d
│       
│       【张量形状 Tensor Shapes】
│         Input:  (B, 2, L)
│         Output: (B, 8, L')
│         说明: B=batch, L=length
│       
│       【参数统计】
│         • weight: (8, 2, 3)  # (out_channels, in_channels, kernel_size) = 48
```

**数据流分析**：
1. 输入复数张量：`(B, 2, L)` - B个样本，2个复数通道，长度L
2. 分为实部和虚部，各自通过Conv1d
3. 每个Conv1d：
   - 输入通道：2
   - 输出通道：8
   - 卷积核：3
   - 参数量：8 × 2 × 3 = 48
4. 输出：`(B, 8, L')` - 8个复数通道

## 🎓 如何理解形状变化

### 1D卷积示例

```
Input:  (32, 2, 100)    # 32个样本，2通道，100长度
        ↓
Conv1d(in=2, out=8, kernel=3, padding=1)
        ↓
Output: (32, 8, 100)    # 32个样本，8通道，100长度（padding保持）
```

### 转置卷积（上采样）示例

```
Input:  (32, 16, 50)    # 32个样本，16通道，50长度
        ↓
ConvTranspose1d(in=16, out=8, kernel=2, stride=2)
        ↓
Output: (32, 8, 100)    # 32个样本，8通道，100长度（2倍上采样）
```

### BatchNorm示例

```
Input:  (32, 8, 100)    # 32个样本，8特征，100长度
        ↓
BatchNorm1d(num_features=8)
        ↓
Output: (32, 8, 100)    # 形状不变，只是归一化
```

## 💡 实际应用

### 调试维度不匹配

```
错误信息: RuntimeError: size mismatch
          expected (32, 16, 50), got (32, 8, 50)

查看分析报告：
├─ layer1: Conv1d
│   【张量形状】
│     Output: (B, 8, L)    ← 输出8通道
│
├─ layer2: Conv1d
│   【张量形状】
│     Input:  (B, 16, L)   ← 期望16通道输入

→ 发现问题：layer1输出8通道，但layer2期望16通道输入！
```

### 计算参数量

```
Conv1d weight: (64, 32, 3)  # (out, in, kernel)

参数量计算：
  64 × 32 × 3 = 6,144 个参数

内存估算(float32)：
  6,144 × 4 bytes = 24.6 KB
```

### 估算输出尺寸

```
Input:  (B, C_in, L_in)
Conv1d: kernel=k, stride=s, padding=p

Output length: L_out = (L_in + 2×p - k) / s + 1

示例：
Input: (32, 2, 100)
Conv1d(kernel=3, stride=1, padding=1)
L_out = (100 + 2×1 - 3) / 1 + 1 = 100
Output: (32, 8, 100)
```

## 🔧 自定义维度说明

如果你的自定义模块需要特殊的维度说明，可以修改 `infer_tensor_shape_meaning()` 函数添加支持。

## 📚 参考资料

- PyTorch官方文档: https://pytorch.org/docs/stable/nn.html
- 维度约定: https://pytorch.org/docs/stable/nn.html#conv1d

---

**总结**：
- ✅ **张量形状**：Input/Output显示数据流
- ✅ **维度标注**：每个维度的含义清晰标明
- ✅ **参数含义**：参数张量的每个维度说明

这些信息帮助你快速理解模型的数据流动和参数组成！📊
