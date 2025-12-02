# 模型复杂度 - 快速参考

## 🎯 一行命令

```bash
python Model_AIIC/analyze_complexity.py
```

结果保存在: `./model_complexity_analysis/`

---

## 📊 结果速览

### 参数量对比（K = 千）

| stages | share=True | share=False |
|--------|------------|-------------|
| **2** | **52K** ⚡ | 105K |
| **3** | **52K** ⚡ | 157K |
| **4** | **52K** ⚡ | 209K |

### FLOPs 对比

| stages | share=True | share=False |
|--------|------------|-------------|
| **2** | **103K** ⚡ | 205K |
| **3** | 103K | 308K |
| **4** | 103K | 410K |

### 推荐配置

| 场景 | 配置 | 参数 | FLOPs | 特点 |
|------|------|------|-------|------|
| 🔥 **超轻量** | stages=2, share=True | 52K | 103K | 最小 |
| ⭐ **平衡** | stages=3, share=False | 157K | 308K | 推荐 |
| 💪 **高性能** | stages=4, share=False | 209K | 410K | 最大 |

---

## 📈 关键发现

### 1. 权重共享

- 参数量: **减少 50-75%** 🔥
- FLOPs: **减少 ~50%**
- 性能: 可能略微下降

### 2. 阶段数影响

**不共享时**:
- 每增加 1 stage → +52K 参数
- 每增加 1 stage → +103K FLOPs

**共享时**:
- 参数量不变（52K）
- FLOPs 几乎不变

---

## 🔧 常用命令

### 分析所有配置
```bash
python Model_AIIC/analyze_complexity.py \
  --stages "2,3,4" \
  --share_weights "True,False"
```

### 只分析特定配置
```bash
python Model_AIIC/analyze_complexity.py \
  --stages "2,3" \
  --share_weights "False"
```

### 不同批大小
```bash
python Model_AIIC/analyze_complexity.py \
  --batch_size 100
```

---

## 📊 统计指标说明

| 指标 | 含义 | 用途 |
|------|------|------|
| **FLOPs** | 浮点运算数 | 推理速度 |
| **MACs** | 乘加操作数 | 硬件设计 |
| **参数量** | 权重数量 | 模型大小 |
| **内存** | 总内存占用 | 部署成本 |

### FLOPs vs MACs

- 1 MAC = 1 乘法 + 1 加法
- FLOPs ≈ 2 × MACs
- 论文通常报告 FLOPs

---

## 💡 复数运算

### 复数乘法
```
(a+bi)(c+di) = (ac-bd) + (ad+bc)i
```
- 4 次实数乘法
- 2 次实数加法

### 复数加法
```
(a+bi) + (c+di) = (a+c) + (b+d)i
```
- 2 次实数加法

---

## 🎓 论文报告

### 表格模板

| Model | Params | FLOPs | NMSE |
|-------|--------|-------|------|
| Ours-S | 52K | 103K | -15dB |
| Ours-M | 157K | 308K | -18dB |
| Ours-L | 209K | 410K | -19dB |

### 文字模板

```
Our model achieves X dB NMSE with only Y parameters 
and Z FLOPs, which is W× more efficient than baseline.
```

---

## 📁 输出文件

```
output_dir/
├── complexity_analysis.json    # 详细数据
└── complexity_comparison.md    # 对比表格
```

---

## 🔗 相关文档

- **完整指南**: `COMPLEXITY_GUIDE.md`
- **使用总结**: `COMPLEXITY_SUMMARY.md`

---

## ⚡ 快速对比

### 同样的 stages=3

| 配置 | 参数 | FLOPs | 差异 |
|------|------|-------|------|
| share=True | 52K | 103K | -66% 参数 |
| share=False | 157K | 308K | 基准 |

### 同样的 share=False

| stages | 参数 | FLOPs | 差异 |
|--------|------|-------|------|
| 2 | 105K | 205K | -33% |
| 3 | 157K | 308K | 基准 |
| 4 | 209K | 410K | +33% |

---

**快速上手**: 运行 `python Model_AIIC/analyze_complexity.py` 即可！
