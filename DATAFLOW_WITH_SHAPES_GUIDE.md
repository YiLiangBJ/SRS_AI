# 📊 数据流与维度显示 - 通用模型分析工具

## 🎯 核心设计原则

### 1. **通用性**
- ✅ 适用于**任何** PyTorch 模型
- ✅ 自动从 `forward()` 源码提取执行流程
- ✅ 不依赖特定模型的硬编码

### 2. **完整性**
- ✅ 显示实际执行顺序
- ✅ 展示循环结构
- ✅ 标注模块类型
- ✅ 显示参数量
- ✅ 显示输入输出形状（针对基础层）

## 📊 新增功能：数据流路径

### 显示内容

```
【数据流路径 Data Flow】
  说明：forward()的实际执行顺序（自动从源码提取）
  
  ┌─ Loop: for i in range(self.depth)
  │    1. x = self.enc_blocks[i](...)  # ComplexResidualBlock
  │         Params: 624 params
  │         → skips.append(x)  # save for later
  │    2. x = self.down_samples[i](...)  # ComplexConv1d
  │         (B, 8, L) → (B, 8, L') | 272 params
  └─
  3. x = self.bottleneck(...)  # ComplexResidualBlock
       Params: 10.8K params
  ┌─ Loop: for i in range(self.depth)
  │    4. x = self.up_samples[i](...)  # ComplexConvTranspose1d
  │         (B, 32, L/4) → (B, 16, L/2) | 2.1K params
  │    5. x = self.dec_blocks[i](...)  # ComplexResidualBlock
  │         Params: 5.9K params
  └─
  6. residual = self.final_conv(...)  # ComplexConv1d
       (B, 8, L) → (B, 1, L) | 18 params
```

### 信息层次

1. **执行步骤编号** - 按实际调用顺序
2. **模块名称** - 如 `self.enc_blocks[i]`
3. **模块类型** - 如 `ComplexConv1d`
4. **形状变换** - `(B, 8, L) → (B, 1, L)` （基础层）
5. **参数量** - `272 params` 或 `10.8K params`
6. **特殊操作** - 如 `skips.append(x)`

## 🔍 针对不同层类型的显示

### 1. 基础层（Conv, Linear等）

```
2. x = self.down_samples[i](...)  # ComplexConv1d
     (B, 8, L) → (B, 8, L') | 272 params
     ↑           ↑             ↑
   输入形状    输出形状       参数量
```

**显示内容**：
- ✅ 输入输出形状（从层属性推断）
- ✅ 参数量统计
- ✅ 维度含义（B=batch, L=length等）

### 2. 复合模块（ResidualBlock, UNet等）

```
1. x = self.enc_blocks[i](...)  # ComplexResidualBlock
     Params: 624 params
     ↑
   参数量统计
```

**显示内容**：
- ✅ 参数量统计（包含所有子模块）
- ⚠️  形状依赖于实际输入，不显示
- ℹ️  可在子模块详细结构中查看

### 3. 无参数操作

```
→ skips.append(x)  # save for later
  ↑
特殊操作标注
```

## 🎨 完整示例：ResNet

```python
class ResNet(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2)
        self.layers = nn.ModuleList([
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 256, stride=2),
        ])
        self.fc = nn.Linear(256, 1000)
    
    def forward(self, x):
        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean([2, 3])
        return self.fc(x)
```

**工具输出**：
```
【数据流路径 Data Flow】
  说明：forward()的实际执行顺序（自动从源码提取）
  
  1. x = self.conv1(...)  # Conv2d
       (B, 3, H, W) → (B, 64, H', W') | 9.5K params
  ┌─ Loop: for layer in self.layers
  │    2. x = layer(...)  # ResidualBlock
  │         Params: 148K params
  └─
  3. return self.fc(...)  # Linear
       (B, 256) → (B, 1000) | 257K params
```

## 📐 形状推断规则

工具从模块属性自动推断形状：

### Conv1d / Conv2d
```python
if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
    in_ch = module.in_channels   # 如 2
    out_ch = module.out_channels  # 如 8
    # Conv1d: (B, 2, L) → (B, 8, L')
    # Conv2d: (B, 2, H, W) → (B, 8, H', W')
```

### BatchNorm
```python
if hasattr(module, 'num_features'):
    nf = module.num_features  # 如 8
    # BatchNorm1d: (B, 8, L) → (B, 8, L)  # 形状不变
```

### Linear
```python
if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
    in_f = module.in_features   # 如 256
    out_f = module.out_features  # 如 1000
    # Linear: (B, 256) → (B, 1000)
```

### 自定义模块
- ⚠️  形状依赖于内部实现
- ✅ 显示参数总量
- ℹ️  可查看子模块的详细结构

## 💡 参数量格式

```python
if params < 1,000:
    "272 params"
elif params < 1,000,000:
    "10.8K params"  # 千
else:
    "2.5M params"   # 百万
```

## 🔄 循环结构识别

### 自动检测
```python
# 源码中的for循环
for i in range(self.depth):
    x = self.enc_blocks[i](x)
    skips.append(x)
    x = self.down_samples[i](x)
```

### 工具显示
```
┌─ Loop: for i in range(self.depth)
│    1. x = self.enc_blocks[i](...)
│         → skips.append(x)  # save for later
│    2. x = self.down_samples[i](...)
└─
```

**关键**：
- ✅ 自动识别循环变量和范围
- ✅ 提取循环体内的模块调用
- ✅ 标注特殊操作（如 append）

## 🛠️ 通用性验证

### 适用场景

| 模型类型 | 支持 | 说明 |
|---------|------|------|
| ResNet | ✅ | 识别残差块循环 |
| UNet | ✅ | 识别编码器/解码器循环 |
| Transformer | ✅ | 识别多层堆叠 |
| LSTM/GRU | ✅ | 显示循环结构 |
| 自定义模型 | ✅ | 只要有forward()方法 |

### 前提条件

1. **模型是 PyTorch `nn.Module`**
2. **有 `forward()` 方法**
3. **使用 Python 源码（非编译）**

### 限制

- ⚠️  无法显示动态计算的形状
- ⚠️  复杂的条件分支可能不完整
- ⚠️  C++扩展模块无法提取源码

## 📊 与传统显示的对比

### 传统方式（模块定义顺序）

```
├─ conv1: Conv2d
├─ layers: ModuleList
│   ├─ 0: ResidualBlock
│   ├─ 1: ResidualBlock
│   └─ 2: ResidualBlock
└─ fc: Linear
```

**问题**：看不出执行顺序

### 新方式（数据流路径）

```
【数据流路径】
  1. conv1(...)
  ┌─ Loop:
  │  2. layers[i](...)
  └─
  3. fc(...)
```

**优势**：清楚展示实际执行流程

## 🎯 实际应用

### 1. 调试模型
```
错误：RuntimeError: size mismatch
查看数据流：
  1. conv1: (B, 3, H, W) → (B, 64, H', W')
  2. layer1: (B, 64, H', W') → (B, 128, H'', W'')
  3. fc: (B, 256) → (B, 1000)  ← 期望输入256维
  
→ 发现：layer输出的特征维度与fc期望不匹配
```

### 2. 理解复杂模型
```
【数据流路径】
  ┌─ Encoder Loop (depth=3)
  │  enc[0] → down[0]
  │  enc[1] → down[1]
  │  enc[2] → down[2]
  └─
  bottleneck
  ┌─ Decoder Loop (depth=3)
  │  up[0] → dec[0]
  │  up[1] → dec[1]
  │  up[2] → dec[2]
  └─
  
→ 清楚看到U-Net的对称结构
```

### 3. 优化模型
```
【数据流路径】
  1. conv1: 9.5K params
  ┌─ Loop:
  │  2. layer[i]: 148K params  ← 每层都很大！
  └─
  3. fc: 257K params
  
→ 发现：ResidualBlock占大部分参数
→ 优化：可以减少通道数
```

## 📚 使用方法

### 分析任何模型

```python
from AnalyzeModelStructure import analyze_model_structure

# 你的模型
model = YourModel(...)

# 分析
analyze_model_structure(
    model, 
    "YourModel",
    output_file="your_model_analysis.txt"
)
```

### 查看报告

```bash
# 生成报告
python AnalyzeModelStructure.py

# 查看文件
cat model_structure_analysis.txt
```

## ✨ 总结

### 核心特点

1. **通用** - 适用于任何PyTorch模型
2. **自动** - 从源码自动提取执行流程
3. **直观** - 清楚展示数据流动
4. **详细** - 包含形状、参数、循环等信息

### 显示的信息

- ✅ 执行顺序（非定义顺序）
- ✅ 循环结构
- ✅ 模块类型
- ✅ 输入输出形状（基础层）
- ✅ 参数量统计
- ✅ 特殊操作（append等）

### 适用模型

- ✅ CNN (ResNet, VGG, etc.)
- ✅ UNet / Encoder-Decoder
- ✅ Transformer
- ✅ RNN / LSTM
- ✅ 自定义架构

现在你可以用这个**通用工具**分析任何PyTorch模型！🎊
