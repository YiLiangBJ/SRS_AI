# 线程监控工具使用指南

## 🎯 目的

诊断为什么反向传播（Backward）占用 80-85% 的训练时间，检查是否是线程数不足导致的。

---

## 🚀 快速测试

### 方法 1：独立测试脚本（推荐）

```bash
cd c:/GitRepo/SRS_AI
python Model_AIIC_onnx/test_thread_usage.py
```

**输出示例**：
```
================================================================================
Thread Usage Report
================================================================================

Phase           Samples    Threads                   Active Cores
--------------------------------------------------------------------------------
data            245        56.3 (48-64)             52.1 (45-56)
forward         189        45.2 (38-52)             38.4 (32-45)
backward        423        28.7 (22-35)             24.3 (18-30)
idle            112        18.2 (15-22)             12.5 (10-15)
================================================================================

Analysis:
⚠️  Backward pass uses 28.7 threads vs 56.3 in data generation
   This may explain why backward takes longer!
   Suggestion: Check if backward operations are not parallelized
```

---

### 方法 2：包装现有训练命令

```bash
# 使用包装脚本监控任何训练命令
python Model_AIIC_onnx/thread_monitor_wrapper.py \
  python Model_AIIC_onnx/test_separator.py \
  --batches 100 --batch_size 4096 --stages "2"
```

**注意**：这会在训练结束后显示报告。

---

## 📊 如何解读结果

### 正常情况 ✅

```
Phase           Threads     Active Cores
data            56.0        52.0
forward         54.0        50.0
backward        52.0        48.0
```

所有阶段的线程数和活跃核心数相近，表示 CPU 利用率均衡。

---

### 问题情况 ⚠️

```
Phase           Threads     Active Cores
data            56.0        52.0        ← 正常
forward         54.0        50.0        ← 正常
backward        28.0        24.0        ← 问题！只用了一半线程！
```

**问题**：反向传播只用了一半的线程！

**可能原因**：
1. PyTorch 自动微分不能完全并行化某些操作
2. 复杂的自定义操作（如 ComplexLinearReal）
3. 内存带宽瓶颈（线程多但在等内存）
4. GIL 限制（如果有 Python 代码在反向传播路径中）

---

## 🔍 诊断步骤

### 步骤 1：运行线程监控测试

```bash
python Model_AIIC_onnx/test_thread_usage.py
```

### 步骤 2：检查结果

查看输出中的：
- **Threads (avg)**: 平均线程数
- **Active Cores (avg)**: 平均活跃核心数
- **Min-Max range**: 波动范围

### 步骤 3：对比不同模型

```bash
# 测试 Model_AIIC（快速版本）
# 修改 test_thread_usage.py 导入 Model_AIIC.channel_separator
python Model_AIIC_onnx/test_thread_usage.py

# 对比结果
```

---

## 💡 可能的解决方案

### 如果 Backward 线程数低：

#### 方案 1：使用更简单的模型结构
- ✅ 使用 `Model_AIIC`（普通 Linear 层）
- ❌ 避免 `ComplexLinearReal`（块矩阵，更复杂）

#### 方案 2：调整线程设置

```python
import torch

# 尝试增加线程数
torch.set_num_threads(112)  # 使用所有物理核心
torch.set_num_interop_threads(4)

# 或尝试减少（有时候太多线程反而慢）
torch.set_num_threads(28)
torch.set_num_interop_threads(2)
```

#### 方案 3：检查内存带宽

```bash
# 使用 Intel VTune 或类似工具
# 查看是否是内存带宽瓶颈
```

#### 方案 4：使用 GPU（如果可用）

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

---

## 📝 工具文件

| 文件 | 说明 |
|------|------|
| `thread_monitor.py` | 核心监控类 |
| `test_thread_usage.py` | 快速测试脚本（推荐）|
| `thread_monitor_wrapper.py` | 包装任意训练命令 |
| `THREAD_MONITOR_INTEGRATION.md` | 集成到训练脚本的示例 |

---

## ⚡ 关键发现

如果监控显示 Backward 线程数明显低于 Data/Forward：

1. **确认问题根源**：不是线程设置问题，而是操作本身不能并行化
2. **最佳解决方案**：使用 `Model_AIIC`（快 2-4 倍）
3. **临时方案**：减小 batch_size，增加 num_batches

---

**现在就运行测试查看你的线程使用情况**：
```bash
python Model_AIIC_onnx/test_thread_usage.py
```
