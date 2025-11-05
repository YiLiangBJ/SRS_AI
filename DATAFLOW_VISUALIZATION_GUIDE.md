# 🔄 数据流路径可视化指南

## 🎯 新增功能

现在模型分析工具会在顶层显示**【数据流路径 Data Flow】**，清楚展示forward()中的实际执行顺序，特别是循环结构。

## ✨ 解决的问题

### 问题：模块定义顺序 ≠ 执行顺序

**旧的显示方式** ❌：
```
├─ enc_blocks: ModuleList
│   ├─ 0: ComplexResidualBlock
│   └─ 1: ComplexResidualBlock
├─ down_samples: ModuleList
│   ├─ 0: ComplexConv1d
│   └─ 1: ComplexConv1d
├─ bottleneck: ComplexResidualBlock
├─ up_samples: ModuleList
│   ├─ 0: ComplexConvTranspose1d
│   └─ 1: ComplexConvTranspose1d
├─ dec_blocks: ModuleList
│   ├─ 0: ComplexResidualBlock
│   └─ 1: ComplexResidualBlock
```

**看不出来实际执行顺序！** 😕

### 解决方案：显示数据流路径 ✅

**新的显示方式**：
```
【数据流路径 Data Flow】
  说明：展示forward()中的实际执行顺序
  
  ┌─ Encoder Loop: for i in range(self.depth)
  │    1. x = self.enc_blocks[i](...)      ← 先执行enc_blocks[0]
  │    2. skips.append(x)                   ← 保存跳跃连接
  │    3. x = self.down_samples[i](...)    ← 然后down_samples[0]
  │                                          ← 循环：enc_blocks[1] → down_samples[1]
  └─
  4. x = self.bottleneck(...)              ← 瓶颈层
  ┌─ Decoder Loop: for i in range(self.depth)
  │    5. x = self.up_samples[i](...)      ← up_samples[0]
  │    6. x = self.dec_blocks[i](...)      ← dec_blocks[0]
  │                                          ← 循环：up_samples[1] → dec_blocks[1]
  └─
  7. residual = self.final_conv(...)       ← 最终输出
```

**一目了然！** ✨

## 📊 完整示例：ComplexResidualUNet

### 代码结构（forward方法）

```python
def forward(self, x):
    # 编码器
    for i in range(self.depth):           # ← 循环开始
        x = self.enc_blocks[i](x)         # ← 步骤1
        skips.append(x)                    # ← 步骤2
        x = self.down_samples[i](x)       # ← 步骤3
    
    # 瓶颈层
    x = self.bottleneck(x)                # ← 步骤4
    
    # 解码器
    for i in range(self.depth):           # ← 循环开始
        x = self.up_samples[i](x)         # ← 步骤5
        skip = skips[-(i+1)]
        x = torch.cat([x, skip], dim=1)
        x = self.dec_blocks[i](x)         # ← 步骤6
    
    # 最终输出
    residual = self.final_conv(x)         # ← 步骤7
    return residual
```

### 工具显示（depth=2的例子）

```
【数据流路径 Data Flow】
  ┌─ Encoder Loop: for i in range(self.depth)  # depth=2, 循环2次
  │    1. x = self.enc_blocks[i](...)
  │           i=0: enc_blocks[0]  (in:2, out:8)
  │           i=1: enc_blocks[1]  (in:8, out:16)
  │    
  │    2. skips.append(x)
  │           保存跳跃连接，用于解码器
  │    
  │    3. x = self.down_samples[i](...)
  │           i=0: down_samples[0] (8ch → 8ch,  L → L/2)
  │           i=1: down_samples[1] (16ch → 16ch, L/2 → L/4)
  └─
  
  4. x = self.bottleneck(...)
         (in:16, out:32, L/4)
  
  ┌─ Decoder Loop: for i in range(self.depth)  # depth=2, 循环2次
  │    5. x = self.up_samples[i](...)
  │           i=0: up_samples[0] (32ch → 16ch, L/4 → L/2)
  │           i=1: up_samples[1] (16ch → 8ch,  L/2 → L)
  │    
  │    6. x = self.dec_blocks[i](...)
  │           i=0: dec_blocks[0] (32ch → 16ch) # 拼接skip后
  │           i=1: dec_blocks[1] (16ch → 8ch)
  └─
  
  7. residual = self.final_conv(...)
         (in:8, out:1, L)
```

## 🔍 对比：定义顺序 vs 执行顺序

### 模块定义顺序（__init__中）

```python
self.enc_blocks = ModuleList([...])     # 第1个定义
self.down_samples = ModuleList([...])   # 第2个定义
self.bottleneck = ...                   # 第3个定义
self.up_samples = ModuleList([...])     # 第4个定义
self.dec_blocks = ModuleList([...])     # 第5个定义
self.final_conv = ...                   # 第6个定义
```

### 实际执行顺序（forward中）

```python
# 循环depth次
for i in range(self.depth):
    enc_blocks[i]      # 第1组
    down_samples[i]    # 第2组

bottleneck             # 第3个

# 循环depth次
for i in range(self.depth):
    up_samples[i]      # 第4组
    dec_blocks[i]      # 第5组

final_conv             # 第6个
```

**关键**：enc_blocks和down_samples交替执行，不是顺序执行所有enc_blocks后再执行所有down_samples！

## 🎨 数据流可视化

```
Input (B, 2, L)
    │
    ▼
┌──────────────────────────┐
│ Encoder Loop (i=0)       │
│  enc_blocks[0]           │  (2→8 channels)
│      │                   │
│      ├─→ skip[0]         │  保存
│      │                   │
│      ▼                   │
│  down_samples[0]         │  (L → L/2)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Encoder Loop (i=1)       │
│  enc_blocks[1]           │  (8→16 channels)
│      │                   │
│      ├─→ skip[1]         │  保存
│      │                   │
│      ▼                   │
│  down_samples[1]         │  (L/2 → L/4)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Bottleneck               │  (16→32 channels, L/4)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Decoder Loop (i=0)       │
│  up_samples[0]           │  (32→16 channels, L/4→L/2)
│      │                   │
│      ├─← skip[1]         │  拼接
│      │                   │
│      ▼                   │
│  dec_blocks[0]           │  (32→16 channels)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ Decoder Loop (i=1)       │
│  up_samples[1]           │  (16→8 channels, L/2→L)
│      │                   │
│      ├─← skip[0]         │  拼接
│      │                   │
│      ▼                   │
│  dec_blocks[1]           │  (16→8 channels)
└──────────────────────────┘
    │
    ▼
┌──────────────────────────┐
│ final_conv               │  (8→1 channels, L)
└──────────────────────────┘
    │
    ▼
Output (B, 1, L)
```

## 💡 关键理解

### 1. 循环展开

**depth=2 时的实际执行**：
```
Step 1: enc_blocks[0]    # i=0, 第1次循环
Step 2: down_samples[0]  # i=0, 第1次循环
Step 3: enc_blocks[1]    # i=1, 第2次循环
Step 4: down_samples[1]  # i=1, 第2次循环
Step 5: bottleneck
Step 6: up_samples[0]    # i=0, 第1次循环
Step 7: dec_blocks[0]    # i=0, 第1次循环
Step 8: up_samples[1]    # i=1, 第2次循环
Step 9: dec_blocks[1]    # i=1, 第2次循环
Step 10: final_conv
```

### 2. 跳跃连接

```
enc_blocks[0] → skip[0] ─────────┐
                                  │
enc_blocks[1] → skip[1] ───┐     │
                            │     │
bottleneck                  │     │
                            │     │
up_samples[0] ← skip[1] ────┘     │
dec_blocks[0]                     │
                                  │
up_samples[1] ← skip[0] ──────────┘
dec_blocks[1]
```

### 3. 形状变化追踪

```
Input:           (B, 2, L)

enc_blocks[0]:   (B, 2, L)  → (B, 8, L)
down_samples[0]: (B, 8, L)  → (B, 8, L/2)

enc_blocks[1]:   (B, 8, L/2)  → (B, 16, L/2)
down_samples[1]: (B, 16, L/2) → (B, 16, L/4)

bottleneck:      (B, 16, L/4) → (B, 32, L/4)

up_samples[0]:   (B, 32, L/4)  → (B, 16, L/2)
cat with skip:   (B, 16, L/2) + (B, 16, L/2) = (B, 32, L/2)
dec_blocks[0]:   (B, 32, L/2)  → (B, 16, L/2)

up_samples[1]:   (B, 16, L/2) → (B, 8, L)
cat with skip:   (B, 8, L) + (B, 8, L) = (B, 16, L)
dec_blocks[1]:   (B, 16, L)   → (B, 8, L)

final_conv:      (B, 8, L)    → (B, 1, L)
```

## 🔧 技术实现

工具通过解析forward()方法的源码实现：

1. **检测for循环**：识别 `for i in range(self.depth)` 等循环
2. **提取循环体**：获取循环内的模块调用
3. **识别ModuleList索引**：匹配 `self.enc_blocks[i]` 等模式
4. **标记循环类型**：根据模块名自动识别Encoder/Decoder
5. **顺序显示**：按照实际执行顺序展示

## 📚 使用场景

### 1. 理解模型架构

快速看懂复杂的U-Net或其他带循环的模型

### 2. 调试数据流

追踪数据在各个模块间的流动

### 3. 优化模型

识别瓶颈，了解哪些模块被重复调用

### 4. 文档说明

为模型生成清晰的执行流程文档

## 📄 查看完整报告

运行：
```bash
python AnalyzeModelStructure.py
```

生成的文件：
- `model_structure_analysis.txt` - 包含数据流路径的完整报告
- `model_structure_summary.txt` - 简化版概览

---

**总结**：
- ✅ 清楚显示循环结构
- ✅ 展示实际执行顺序
- ✅ 标注Encoder/Decoder循环
- ✅ 解决"定义顺序≠执行顺序"的困惑

现在一眼就能看懂模型的数据流动！🎊
