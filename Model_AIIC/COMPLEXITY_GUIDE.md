# 模型复杂度分析指南

## 🎯 功能

自动分析不同超参数组合下的：
1. **参数量** - 模型权重数量
2. **计算复杂度** - FLOPs, MACs, 乘法/加法次数
3. **内存使用** - 参数、输入输出、中间激活

---

## 📊 计算量统计方法

### 1. FLOPs (Floating Point Operations) ⭐⭐⭐⭐⭐

**定义**: 浮点运算总数（乘法 + 加法）

**优点**:
- 最通用的指标
- 论文中最常用
- 反映计算复杂度

**示例**:
- 矩阵乘法 (M×K) @ (K×N): FLOPs = M × N × (2K-1) ≈ 2MNK
- 对于复数: 需要拆分为实数运算

### 2. MACs (Multiply-Accumulate Operations) ⭐⭐⭐⭐

**定义**: 乘加操作数 (1 MAC = 1 乘法 + 1 加法)

**关系**: FLOPs ≈ 2 × MACs (对于矩阵乘法)

**优点**:
- 更接近硬件实现
- 适合 DSP/FPGA 评估

### 3. 实数乘法/加法次数 ⭐⭐⭐

**定义**: 拆分为最基本的实数运算

**适用场景**:
- 复数运算分析
- 硬件设计
- 精确计算量

**复数运算**:
```
复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  - 4 次实数乘法
  - 2 次实数加法

复数加法: (a+bi) + (c+di) = (a+c) + (b+d)i
  - 2 次实数加法
```

---

## 🚀 使用方法

### 基本用法

```bash
python Model_AIIC/analyze_complexity.py
```

这会分析默认配置：
- stages: 2, 3, 4
- share_weights: True, False

### 自定义配置

```bash
# 只分析特定阶段数
python Model_AIIC/analyze_complexity.py \
  --stages "2,3" \
  --share_weights "False"

# 分析所有组合
python Model_AIIC/analyze_complexity.py \
  --stages "2,3,4" \
  --share_weights "True,False"

# 指定批大小（影响 FLOPs 统计）
python Model_AIIC/analyze_complexity.py \
  --batch_size 100 \
  --output ./complexity_batch100
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stages` | "2,3,4" | 阶段数列表 |
| `--share_weights` | "True,False" | 权重共享选项 |
| `--batch_size` | 1 | 推理批大小 |
| `--output` | ./model_complexity_analysis | 结果目录 |

---

## 📁 输出文件

```
output_dir/
├── complexity_analysis.json       # 详细数据（JSON）
└── complexity_comparison.md       # 对比表格（Markdown）
```

### JSON 内容

```json
{
  "config": {
    "num_stages": 2,
    "share_weights": false,
    ...
  },
  "parameters": {
    "total": 104640,
    "trainable": 104640,
    "memory_bytes": 817280,
    "details": {...}
  },
  "complexity": {
    "total_flops": 204992,
    "total_macs": 102400,
    "total_real_muls": 102400,
    "total_real_adds": 102592,
    ...
  },
  "memory": {
    "parameters": 817280,
    "input": 96,
    "output": 384,
    "activations": 4096,
    "total": 821856
  }
}
```

### Markdown 内容

包含三个对比表：
1. **参数量对比**
2. **计算复杂度对比**
3. **内存使用对比**

---

## 📈 结果解读

### 分析结果示例

运行后输出：

```
================================================================================
配置: stages=2, share_weights=False
================================================================================

📊 参数统计:
  总参数量: 104.64K (104,640)
  可训练参数: 104.64K (104,640)
  参数内存: 817.50 KB

⚡ 计算复杂度 (batch_size=1):
  总 FLOPs: 204.99K (204,992)
  总 MACs: 102.40K (102,400)
  实数乘法: 102.40K (102,400)
  实数加法: 102.59K (102,592)

💾 内存估算 (batch_size=1):
  参数内存: 817.50 KB
  输入内存: 96 B
  输出内存: 384 B
  中间激活: 4.00 KB
  总内存: 821.97 KB
```

### 关键指标说明

#### 参数量
- **影响**: 模型存储大小、加载时间
- **典型值**: 50K ~ 200K
- **优化**: 使用权重共享可减少 50%

#### FLOPs
- **影响**: 推理速度、功耗
- **典型值**: 100K ~ 400K (batch_size=1)
- **缩放**: 与 batch_size 线性关系

#### 内存
- **参数内存**: 固定（与模型大小相关）
- **激活内存**: 与 batch_size 和阶段数相关
- **总内存**: 推理时需要的总内存

---

## 🔍 对比分析

### 权重共享的影响

| 配置 | share=True | share=False | 差异 |
|------|------------|-------------|------|
| stages=2 | 52.32K | 104.64K | **2倍** |
| stages=3 | 52.32K | 156.96K | **3倍** |
| stages=4 | 52.32K | 209.28K | **4倍** |

**结论**: 
- ✅ 权重共享可大幅减少参数量
- ✅ 参数量不随阶段数增加（共享时）
- ⚠️ 可能影响性能（需要权衡）

### 阶段数的影响

**参数量** (share=False):
- stages=2: 104.64K
- stages=3: 156.96K (+50%)
- stages=4: 209.28K (+100%)

**FLOPs** (share=False):
- stages=2: 204.99K
- stages=3: 307.49K (+50%)
- stages=4: 409.98K (+100%)

**结论**: 线性增长

---

## 💡 优化建议

### 1. 减少参数量

```bash
# 使用权重共享
--stages "3" --share_weights "True"
# 参数量: 52.32K (vs 156.96K 不共享)
```

### 2. 减少计算量

```bash
# 使用更少阶段
--stages "2" --share_weights "False"
# FLOPs: 204.99K (vs 409.98K for stages=4)
```

### 3. 平衡配置

```bash
# 推荐：2-3 阶段，不共享权重
--stages "2,3" --share_weights "False"
# 在性能和复杂度之间取得平衡
```

---

## 🎓 理论基础

### 线性层复杂度

**实数线性层**: `y = Wx + b`
- 输入: M 维
- 输出: N 维
- **乘法**: M × N
- **加法**: (M-1) × N + N (累加 + 偏置)
- **FLOPs**: ≈ 2MN

**复数线性层**: 拆分为实部和虚部
- 每个部分处理 [real, imag] 拼接的输入
- 输入维度翻倍: 2M → N
- **FLOPs**: ≈ 2 × 2MN = 4MN

### 模型结构

我们的模型包含（每个 port）:
```
1. Linear(seq_len*2, hidden_dim) - Real MLP
2. ReLU
3. Linear(hidden_dim, hidden_dim)
4. ReLU
5. Linear(hidden_dim, seq_len)

6. Linear(seq_len*2, hidden_dim) - Imag MLP
7. ReLU
8. Linear(hidden_dim, hidden_dim)
9. ReLU
10. Linear(hidden_dim, seq_len)
```

**每个 port 每个 stage 的 FLOPs**:
```
Layer 1: 2 × seq_len*2 × hidden_dim = 2 × 24 × 64 = 3,072
Layer 2: 2 × hidden_dim × hidden_dim = 2 × 64 × 64 = 8,192
Layer 3: 2 × hidden_dim × seq_len = 2 × 64 × 12 = 1,536
(×2 for real+imag)

Total per port per stage: 2 × (3072 + 8192 + 1536) = 25,600
Total per stage (4 ports): 4 × 25,600 = 102,400
```

---

## 📚 相关工具

### Python 工具

1. **thop** (推荐)
```bash
pip install thop
```

2. **fvcore**
```bash
pip install fvcore
```

3. **ptflops**
```bash
pip install ptflops
```

### 为什么自己实现？

- ✅ **复数支持**: 现有工具不支持复数运算
- ✅ **详细统计**: 可以分别统计乘法和加法
- ✅ **自定义层**: 我们的 ComplexMLP 是自定义的

---

## 🔗 相关文档

- `EVALUATION_GUIDE.md` - 性能评估指南
- `README.md` - 完整使用手册
- `QUICKSTART.md` - 快速开始

---

## 📊 论文中如何报告

### 参数量表格

| Model | Stages | Share | Parameters | FLOPs |
|-------|--------|-------|------------|-------|
| Ours-S | 2 | ✓ | 52.3K | 102.6K |
| Ours-M | 3 | ✗ | 157.0K | 307.5K |
| Ours-L | 4 | ✗ | 209.3K | 410.0K |

### 复杂度对比

```
Our proposed model achieves competitive performance with only 
52.3K parameters and 102.6K FLOPs per inference, which is 
significantly lower than existing methods.
```

---

**最后更新**: 2025-12-02  
**版本**: v1.0
