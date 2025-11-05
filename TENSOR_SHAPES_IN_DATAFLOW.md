# 🎯 数据流中的张量形状显示

## ✅ 实现效果

现在在数据流路径中，每次模块调用都会显示输入输出张量的维度！

### 完整示例：ComplexResidualUNet (depth=2)

```
【数据流路径 Data Flow】
  说明：forward()的实际执行顺序（自动从源码提取）

  ┌─ Loop: for i in range(self.depth)
  │    (展开显示每次迭代的实际参数量)
  │
  │  ─── 迭代 i=0 ───
  │    1. x = self.enc_blocks[0](...)  # ComplexResidualBlock
  │         Shape: (B, 2, L) → (B, 8, L)         ← 🆕 输入输出形状！
  │         Params: 624 params
  │         → skips.append(x)
  │    2. x = self.down_samples[0](...)  # ComplexConv1d
  │         Shape: (B, 8, L) → (B, 8, L/2)       ← 🆕 下采样形状变化！
  │         Params: 272 params
  │
  │  ─── 迭代 i=1 ───
  │    3. x = self.enc_blocks[1](...)  # ComplexResidualBlock
  │         Shape: (B, 8, L) → (B, 16, L)        ← 🆕 通道数变化！
  │         Params: 2,784 params
  │         → skips.append(x)
  │    4. x = self.down_samples[1](...)  # ComplexConv1d
  │         Shape: (B, 16, L) → (B, 16, L/2)     ← 🆕 继续下采样！
  │         Params: 1,056 params
  └─
  5. x = self.bottleneck(...)  # ComplexResidualBlock
       Shape: (B, 16, L) → (B, 32, L)            ← 🆕 瓶颈层形状！
       Params: 10,816 params
  ┌─ Loop: for i in range(self.depth)
  │    (展开显示每次迭代的实际参数量)
  │
  │  ─── 迭代 i=0 ───
  │    6. x = self.up_samples[0](...)  # ComplexConvTranspose1d
  │         Shape: (B, 32, L) → (B, 16, L×2)     ← 🆕 上采样形状变化！
  │         Params: 2,080 params
  │    7. x = self.dec_blocks[0](...)  # ComplexResidualBlock
  │         Shape: (B, 32, L) → (B, 16, L)       ← 🆕 跳跃连接后的形状！
  │         Params: 5,856 params
  │
  │  ─── 迭代 i=1 ───
  │    8. x = self.up_samples[1](...)  # ComplexConvTranspose1d
  │         Shape: (B, 16, L) → (B, 8, L×2)      ← 🆕 继续上采样！
  │         Params: 528 params
  │    9. x = self.dec_blocks[1](...)  # ComplexResidualBlock
  │         Shape: (B, 16, L) → (B, 8, L)        ← 🆕 输出通道减少！
  │         Params: 1,520 params
  └─
  10. residual = self.final_conv(...)  # ComplexConv1d
       Shape: (B, 8, L) → (B, 1, L)              ← 🆕 最终输出形状！
       Params: 18 params
```

## 🔍 关键信息展示

### 每次调用显示的信息

```
1. x = self.enc_blocks[0](...)  # ComplexResidualBlock
     ↓
     Shape: (B, 2, L) → (B, 8, L)        ← 输入输出形状
     Params: 624 params                  ← 精确参数量
     → skips.append(x)                    ← 副作用操作
```

**包含**：
1. ✅ 步骤编号
2. ✅ 模块名称和索引
3. ✅ 模块类型
4. ✅ **输入输出形状** 🆕
5. ✅ 精确参数量
6. ✅ 特殊操作（如append）

## 📊 形状变化追踪

### 编码器路径（通道数递增，长度递减）

```
输入: (B, 2, L)
  ↓ enc_blocks[0]
(B, 8, L)
  ↓ down_samples[0] (stride=2)
(B, 8, L/2)
  ↓ enc_blocks[1]
(B, 16, L/2)
  ↓ down_samples[1] (stride=2)
(B, 16, L/4)
  ↓ bottleneck
(B, 32, L/4)
```

**观察**：
- 通道数：2 → 8 → 16 → 32（倍增）
- 长度：L → L/2 → L/4（减半）

### 解码器路径（通道数递减，长度递增）

```
瓶颈输出: (B, 32, L/4)
  ↓ up_samples[0] (stride=2)
(B, 16, L/2)
  ↓ concat with skip (B, 16, L/2) → (B, 32, L/2)
  ↓ dec_blocks[0]
(B, 16, L/2)
  ↓ up_samples[1] (stride=2)
(B, 8, L)
  ↓ concat with skip (B, 8, L) → (B, 16, L)
  ↓ dec_blocks[1]
(B, 8, L)
  ↓ final_conv
(B, 1, L)
```

**观察**：
- 上采样：L/4 → L/2 → L（长度恢复）
- 跳跃连接：通道数翻倍
- 解码块：通道数减半
- 最终输出：1个通道

## 🎨 自定义模块支持

### ComplexResidualBlock

```python
# 检测逻辑
if 'ComplexResidualBlock' in module_type:
    if hasattr(module, 'conv1') and hasattr(module.conv1, 'conv_real'):
        in_ch = module.conv1.conv_real.in_channels
        out_ch = module.conv1.conv_real.out_channels
        info['input_shape'] = f"(B, {in_ch}, L)"
        info['output_shape'] = f"(B, {out_ch}, L)"
        info['shape_note'] = "B=batch, L=length, complex tensor"
```

**显示**：
```
Shape: (B, 2, L) → (B, 8, L)
```

### ComplexConv1d

```python
# 检测逻辑
if 'ComplexConv1d' in module_type:
    if hasattr(module, 'conv_real'):
        in_ch = module.conv_real.in_channels
        out_ch = module.conv_real.out_channels
        if hasattr(module, 'stride') and module.stride > 1:
            info['output_shape'] = f"(B, {out_ch}, L/{module.stride})"
        else:
            info['output_shape'] = f"(B, {out_ch}, L)"
```

**显示**：
```
Shape: (B, 8, L) → (B, 8, L/2)  # stride=2
Shape: (B, 8, L) → (B, 1, L)    # stride=1
```

### ComplexConvTranspose1d

```python
# 检测逻辑
if 'ComplexConvTranspose1d' in module_type:
    if hasattr(module, 'conv_real'):
        in_ch = module.conv_real.in_channels
        out_ch = module.conv_real.out_channels
        info['input_shape'] = f"(B, {in_ch}, L)"
        info['output_shape'] = f"(B, {out_ch}, L×2)"
```

**显示**：
```
Shape: (B, 32, L) → (B, 16, L×2)  # 上采样
```

## 💡 实际应用场景

### 场景1：调试维度不匹配

```python
# 错误信息: RuntimeError: size mismatch at (B, 16, L/2)
# 
# 查看数据流：
# 6. up_samples[0]: (B, 32, L/4) → (B, 16, L/2)  ✓
# 7. dec_blocks[0]: (B, 32, L/2) → (B, 16, L/2)  ← 期望(B, 32, L/2)
#    
# → 发现：需要跳跃连接！concat后应该是(B, 32, L/2)
```

### 场景2：理解U-Net对称性

```
编码器:                           解码器:
(B, 2, L)                         (B, 1, L)
  ↓ enc[0]                          ↑ final
(B, 8, L) ─────skip────→        (B, 8, L)
  ↓ down[0]                         ↑ dec[1]
(B, 8, L/2)                       (B, 16, L)
  ↓ enc[1]                          ↑ up[1]
(B, 16, L/2) ───skip───→        (B, 8, L/2)
  ↓ down[1]                         ↑ dec[0]
(B, 16, L/4)                      (B, 32, L/2)
  ↓ bottleneck                      ↑ up[0]
(B, 32, L/4) ─────────→         (B, 16, L/4)
```

**清晰可见**：
- 编码器和解码器的对称性
- 跳跃连接的位置和形状
- 通道数和长度的变化规律

### 场景3：验证输出形状

```
需求：输入 (B, 2, 12)，输出 (B, 1, 12)

查看数据流：
  1. enc_blocks[0]: (B, 2, L) → (B, 8, L)     ✓ 通道变化正确
  2. down_samples[0]: (B, 8, L) → (B, 8, L/2) ✓ 长度减半
  ...
  10. final_conv: (B, 8, L) → (B, 1, L)       ✓ 最终输出正确

→ 形状变换符合预期！
```

## 📋 显示格式

### 标准格式

```
Shape: (B, in_ch, L) → (B, out_ch, L')
       ↑    ↑    ↑       ↑    ↑     ↑
     batch 输入  长度   batch 输出  长度'
           通道                通道
```

### 特殊标记

- `L` - 原始长度
- `L'` - 变化后的长度（不确定倍数）
- `L/2` - 长度减半
- `L×2` - 长度翻倍
- `L/4` - 长度缩减到1/4
- `B` - batch size
- `complex tensor` - 复数张量

## 🎯 关键优势

| 方面 | 之前 | 现在 |
|------|------|------|
| 形状可见性 | ❌ 需要猜测 | ✅ 直接显示 |
| 调试效率 | ⚠️ 需要推断 | ✅ 一目了然 |
| 理解流程 | ⚠️ 抽象 | ✅ 具体 |
| 验证输出 | ❌ 困难 | ✅ 容易 |

## ✨ 总结

### 核心改进

1. **形状追踪**：每次调用都显示输入输出形状
2. **完整信息**：形状 + 参数量 + 操作
3. **自动识别**：支持自定义Complex模块
4. **清晰格式**：分行显示，易于阅读

### 显示内容（每个模块调用）

```
步骤编号. 变量 = self.模块名[索引](...)  # 模块类型
   Shape: (输入形状) → (输出形状)       ← 🆕 张量形状
   Params: 精确参数量                   ← 🆕 参数统计
   → 特殊操作                            ← 副作用
```

### 使用建议

1. **查看完整流程**：从顶层数据流路径开始
2. **追踪形状变化**：关注每一步的形状转换
3. **验证通道数**：确保与设计一致
4. **检查对称性**：U-Net等模型的对称性
5. **调试不匹配**：快速定位维度问题

现在可以完整追踪数据在模型中的流动和形状变化！🎊
