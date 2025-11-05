# 🔢 参数量计算说明

## 🎯 你的疑问

```
按照数据流显示:
  编码器循环: (624+272) × 2 = 1,792
  瓶颈层:     10,800
  解码器循环: (2,100+5,900) × 2 = 16,000
  最终卷积:   18
  ──────────────────────────
  总计:       28,610  ← 与实际的25,554不符！
```

## ✅ 正确答案

**问题**：循环中**每次迭代的参数量是不同的**！

### 详细分解（depth=2）

```
【编码器循环】for i in range(2):
  i=0: enc_blocks[0] (2→8 channels)    =   624 params
       down_samples[0] (8→8, L→L/2)   =   272 params
       ────────────────────────────────────────────
       小计:                               896 params

  i=1: enc_blocks[1] (8→16 channels)   = 2,784 params  ← 不是624！
       down_samples[1] (16→16, L/2→L/4)= 1,056 params  ← 不是272！
       ────────────────────────────────────────────
       小计:                             3,840 params

  编码器总计:                            4,736 params

【瓶颈层】
  bottleneck (16→32 channels):         10,816 params

【解码器循环】for i in range(2):
  i=0: up_samples[0] (32→16, L/4→L/2)  = 2,080 params
       dec_blocks[0] (32→16 channels)  = 5,856 params
       ────────────────────────────────────────────
       小计:                             7,936 params

  i=1: up_samples[1] (16→8, L/2→L)     =   528 params  ← 不是2,080！
       dec_blocks[1] (16→8 channels)   = 1,520 params  ← 不是5,856！
       ────────────────────────────────────────────
       小计:                             2,048 params

  解码器总计:                            9,984 params

【最终卷积】
  final_conv (8→1 channels):                18 params

═══════════════════════════════════════════════════════
总参数量: 4,736 + 10,816 + 9,984 + 18 = 25,554 ✓
═══════════════════════════════════════════════════════
```

## 💡 为什么参数量不同？

### 通道数在变化

ComplexResidualUNet的编码器中，每层通道数翻倍：

```python
# __init__中的编码器构建
in_ch = input_channels  # 2
out_ch = base_channels  # 8

for i in range(depth):  # depth=2
    self.enc_blocks.append(
        ComplexResidualBlock(in_ch, out_ch, ...)
    )
    in_ch = out_ch
    out_ch = min(out_ch * 2, 256)  # ← 每次翻倍！
```

**结果**：
- `i=0`: in_ch=2,  out_ch=8   → 小参数量
- `i=1`: in_ch=8,  out_ch=16  → 大参数量（4倍）
- `i=2`: in_ch=16, out_ch=32  → 更大参数量

### 参数量与通道数的关系

对于卷积层：`params ∝ in_channels × out_channels × kernel_size`

```
enc_blocks[0]: ComplexResidualBlock(2, 8)
  - 主要参数来自Conv1d(2, 8, kernel=3)
  - 参数量 ∝ 2 × 8 × 3 = 48 (每个卷积)

enc_blocks[1]: ComplexResidualBlock(8, 16)
  - 主要参数来自Conv1d(8, 16, kernel=3)
  - 参数量 ∝ 8 × 16 × 3 = 384 (每个卷积)  ← 8倍！
```

## 📊 完整参数分布

```
模块                    通道变化          参数量      占比
──────────────────  ────────────────  ─────────  ──────
enc_blocks[0]       2 → 8                  624    2.4%
down_samples[0]     8 → 8 (下采样)         272    1.1%
enc_blocks[1]       8 → 16               2,784   10.9%
down_samples[1]     16 → 16 (下采样)     1,056    4.1%
bottleneck          16 → 32             10,816   42.3%  ← 最大！
up_samples[0]       32 → 16              2,080    8.1%
dec_blocks[0]       32 → 16              5,856   22.9%
up_samples[1]       16 → 8                 528    2.1%
dec_blocks[1]       16 → 8               1,520    5.9%
final_conv          8 → 1                   18    0.1%
──────────────────────────────────────────────────────
总计                                    25,554  100.0%
```

## 🔍 工具显示的改进

现在工具会显示：

```
┌─ Loop: for i in range(self.depth)
│    (注意：循环每次迭代的通道数/参数量可能不同，以下显示首次迭代)
│    (循环总参数: 4.7K params)  ← 🆕 显示循环总参数！
│    1. x = self.enc_blocks[i](...)  # ComplexResidualBlock
│         Params: 624 params  ← 仅第一次迭代(i=0)
│         → skips.append(x)
│    2. x = self.down_samples[i](...)
│         Params: 272 params  ← 仅第一次迭代(i=0)
└─
```

**关键信息**：
1. ✅ **注意提示**：循环每次迭代参数不同
2. ✅ **循环总参数**：整个循环的参数总和
3. ✅ **首次迭代示例**：显示第一次(i=0)的参数量

## 📐 通用规律

### U-Net类模型

```
编码器（下采样）:
  层级越深，通道数越多，参数量越大
  ↓
  2ch → 8ch → 16ch → 32ch
  小    中     大     最大

解码器（上采样）:
  层级越浅，通道数越少，参数量越小
  ↓
  32ch → 16ch → 8ch → 1ch
  大     中     小    最小
```

### ResNet类模型

```
残差块:
  stage1: 64ch  → 中等参数量
  stage2: 128ch → 较大参数量
  stage3: 256ch → 大参数量
  stage4: 512ch → 最大参数量
```

## ✅ 总结

1. **循环≠重复相同参数**
   - 每次迭代可能有不同的通道数
   - 参数量与通道数的平方成正比

2. **工具显示优化**
   - 显示循环总参数量
   - 标注"首次迭代示例"
   - 提醒参数量可能变化

3. **验证方法**
   ```python
   # 逐层验证
   for i in range(depth):
       params = sum(p.numel() for p in model.enc_blocks[i].parameters())
       print(f'enc_blocks[{i}]: {params} params')
   ```

4. **正确计算公式**
   ```
   总参数 = Σ(每次迭代的实际参数) + 非循环部分参数
   
   而不是:
   总参数 ≠ (首次迭代参数 × 循环次数) + 非循环部分参数
   ```

现在你明白了为什么 28,610 ≠ 25,554 了吧！🎯
