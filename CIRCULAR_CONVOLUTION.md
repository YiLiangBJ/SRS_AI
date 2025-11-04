# Circular Convolution Implementation

## 概述

为了保持序列长度不变并实现循环卷积，我们在 `ComplexConv1d` 中添加了 `circular` 参数。

## 工作原理

### 传统卷积 vs 循环卷积

**传统卷积（padding='same'）**：
```
序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
填充: [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0]
      ↑  ↑                                      ↑  ↑
     零填充                                    零填充
```

**循环卷积（circular=True）**：
```
序列: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
填充: [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
       ↑                                       ↑  ↑
    循环回绕                              循环回绕到开头
```

### 示例：kernel_size=3

| 位置 | 传统卷积使用 | 循环卷积使用 |
|------|-------------|-------------|
| 第1个输出 | [**0**, 0, 1] | [**11**, 0, 1] ✓ |
| 第2个输出 | [0, 1, 2] | [0, 1, 2] |
| 第11个输出 | [9, 10, 11] | [9, 10, 11] |
| 第12个输出 | [10, 11, **0**] | [10, 11, **0**] ✓ |

## 代码实现

### 1. ComplexConv1d 更新

```python
class ComplexConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, bias=True, circular=False):
        # 新参数: circular - 是否使用循环卷积
```

**关键代码**：
```python
if self.circular:
    # 计算填充量
    pad_total = self.kernel_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    # 循环填充
    x_real_padded = F.pad(x.real, (pad_left, pad_right), mode='circular')
    x_imag_padded = F.pad(x.imag, (pad_left, pad_right), mode='circular')
```

### 2. ComplexResidualBlock 更新

```python
class ComplexResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 use_attention=False, activation='modrelu', circular=True):
        # 默认使用循环卷积
        
        self.conv1 = ComplexConv1d(..., circular=circular)
        self.conv2 = ComplexConv1d(..., circular=circular)
        
        # shortcut 不使用循环（因为是 1x1 卷积）
        self.shortcut = ComplexConv1d(..., circular=False)
```

### 3. ComplexResidualUNet 更新

```python
class ComplexResidualUNet(nn.Module):
    def __init__(self, ..., circular=True):
        # 新参数控制是否使用循环卷积
        
        # 编码器块使用循环卷积
        self.enc_blocks.append(
            ComplexResidualBlock(..., circular=circular)
        )
        
        # 下采样不使用循环（改变尺寸）
        self.down_samples.append(
            ComplexConv1d(..., circular=False)
        )
```

## 测试验证

运行 `test_circular_conv.py` 验证：

```bash
python test_circular_conv.py
```

### 测试项目

1. ✅ **循环填充正确性**
   - 验证 `[0,1,2,...,11]` 填充为 `[11,0,1,2,...,11,0]`

2. ✅ **序列长度保持**
   - 输入: (B, C, 12)
   - 输出: (B, C', 12)
   - 长度不变

3. ✅ **边界值正确性**
   - 第一个卷积: 使用 [11, 0, 1]
   - 最后一个卷积: 使用 [10, 11, 0]

4. ✅ **完整 U-Net 集成**
   - 输入: (8, 4, 2, 12)
   - 输出: (8, 4, 1, 12)
   - 端到端测试通过

## 使用说明

### 默认行为（推荐）

```python
# 创建模型（默认使用循环卷积）
model = ComplexResidualUNet(
    input_channels=2,
    output_channels=1,
    base_channels=32,
    depth=3,
    circular=True  # 默认值
)
```

### 禁用循环卷积

```python
# 如果需要传统卷积
model = ComplexResidualUNet(
    input_channels=2,
    output_channels=1,
    base_channels=32,
    depth=3,
    circular=False  # 使用传统 padding
)
```

## 优势

### 1. 保持序列完整性
- ✅ 序列边界信息不丢失
- ✅ 符合周期性信号的物理特性
- ✅ 适合频域和时域循环数据

### 2. 避免边界效应
- ❌ 传统填充：边界处用0填充，引入不自然的不连续
- ✅ 循环填充：边界自然衔接，符合周期性

### 3. 适用场景
- ✅ **SRS 信道估计**：资源块具有周期性
- ✅ **频域处理**：频谱是周期的
- ✅ **时间序列**：循环数据（如角度、相位）

## 性能影响

| 特性 | 传统卷积 | 循环卷积 |
|------|---------|---------|
| 计算量 | 相同 | 相同 |
| 内存使用 | 相同 | 相同 |
| 参数量 | 相同 | 相同 |
| 序列长度 | 保持 | 保持 |
| 边界处理 | 零填充 | 循环填充 |

**结论**：循环卷积没有额外开销，只是改变了填充方式。

## 实验对比

### 示例：kernel_size=3, 输入 [0,1,2,...,11]

**传统卷积（权重=[1,1,1]/3）**：
```
位置 0: (0+0+1)/3 = 0.333  # 左边用0填充
位置 11: (10+11+0)/3 = 7.0 # 右边用0填充
```

**循环卷积（权重=[1,1,1]/3）**：
```
位置 0: (11+0+1)/3 = 4.0   # 左边循环到11
位置 11: (10+11+0)/3 = 7.0 # 右边循环到0
```

## 注意事项

### 何时使用循环卷积

✅ **推荐使用**：
- 周期性数据（频域、相位）
- 循环缓冲区
- 时间序列预测
- 信道估计（资源块循环）

❌ **不推荐使用**：
- 下采样层（stride > 1）
- 上采样层（转置卷积）
- 1x1 卷积（无需填充）
- 非周期性数据

### 当前实现

```python
# 使用循环卷积的层
- enc_blocks (编码器残差块)
- bottleneck (瓶颈层)
- dec_blocks (解码器残差块)

# 不使用循环卷积的层
- down_samples (下采样)
- up_samples (上采样)
- final_conv (1x1 输出卷积)
- shortcut (1x1 残差连接)
```

## 未来扩展

可以考虑添加：
- 其他填充模式：`reflect`, `replicate`
- 可配置每层的填充方式
- 自适应填充策略

## 参考资料

- PyTorch `F.pad` 文档：https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
- Circular Padding in CNNs: 适用于周期性边界条件
