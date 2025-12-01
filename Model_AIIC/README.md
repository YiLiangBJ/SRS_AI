# Model_AIIC 训练与使用指南# Channel Separator Models for SRS Multi-Port Estimation



**最后更新**: 2025-12-01  ## 🎯 Quick Start

**版本**: v2.0

### Test Models

---

```bash

## 📚 目录cd Model_AIIC



1. [快速开始](#快速开始)# Test all models

2. [test_separator.py 使用指南](#test_separatorpy-使用指南)python channel_separator.py

3. [TensorBoard 可视化](#tensorboard-可视化)

4. [超参数搜索](#超参数搜索)# Test with data generator integration

5. [CPU 性能优化](#cpu-性能优化)python test_separator.py --model mlp --epochs 10

6. [模型部署](#模型部署)python test_separator.py --model residual --epochs 20

7. [故障排查](#故障排查)python test_separator.py --model hinted --epochs 10

```

---

## 📁 Files

## 🚀 快速开始

- `channel_separator.py` - Model implementations (3 variants)

### 基础训练- `test_separator.py` - Integration test with data generator

- `INTEGRATION_GUIDE.md` - Detailed integration guide (Chinese)

```bash

python Model_AIIC/test_separator.py \## 🏗️ Models

  --batches 1000 \

  --batch_size 2048 \### 1. SimpleMLPSeparator

  --stages 3 \- **Parameters**: ~20K

  --save_dir "./my_training"- **Best for**: Fast training, simple baseline

```

### 2. ResidualRefinementSeparator  

### 查看训练过程- **Parameters**: ~15K

- **Best for**: Iterative refinement with residual correction

```bash

# 启动 TensorBoard### 3. PositionHintedSeparator

tensorboard --logdir ./my_training- **Parameters**: ~25K  

- **Best for**: Stable training with position hints

# 在浏览器打开

http://localhost:6006## 📊 Problem

```

$$y = \sum_{p \in P} \text{circshift}(h_p, p) + noise$$

---

**Goal**: Separate mixed signal into individual shifted components

## 📖 test_separator.py 使用指南

## 🔗 Integration

### 命令行参数

See `INTEGRATION_GUIDE.md` for detailed integration steps with existing codebase.

| 参数 | 类型 | 默认值 | 说明 |

|------|------|--------|------|Quick summary:

| `--batches` | int | 10000 | 训练批次数 |1. Add to `data_generator.py` - data generation

| `--batch_size` | int | 2048 | 批大小（建议 2048-4096） |2. Modify `trainMLPmmse.py` - training loop  

| `--stages` | str | "3" | 模型阶段数 |3. Update `evaluate_performance.py` - evaluation

| `--share_weights` | str | "False" | 是否共享权重 |

| `--snr` | str | "20.0" | 信噪比配置 |## ✅ Next Steps

| `--tdl` | str | "A-30" | TDL 信道配置 |

| `--early_stop` | float | None | 早停阈值 |1. Run quick tests (above)

| `--val_interval` | int | 100 | 验证间隔 |2. Read `INTEGRATION_GUIDE.md`

| `--patience` | int | 5 | 早停耐心值 |3. Choose integration approach

| `--save_dir` | str | None | 保存目录 |4. Start with data generation modification

| `--cpu_ratio` | float | 1.0 | CPU 核心使用比例 |

### SNR 配置

#### 固定 SNR（所有端口相同）
```bash
--snr "20.0"
```

#### SNR 范围（随机采样）
```bash
--snr "0,30"  # 每个样本从 [0, 30] dB 均匀采样
```

#### 每端口不同 SNR
```bash
--snr "[15,18,20,22]"  # 端口 0-3 分别使用固定 SNR
```

### TDL 信道配置

#### 单一配置
```bash
--tdl "A-30"   # TDL-A，30ns RMS delay spread
--tdl "B-100"  # TDL-B，100ns RMS delay spread
--tdl "C-300"  # TDL-C，300ns RMS delay spread
```

#### 多配置（随机选择）
```bash
--tdl "A-30,B-100,C-300"  # 每个样本随机选择一种
```

### 早停配置

```bash
--early_stop 0.01        # 当 validation loss < 0.01 时触发
--val_interval 100       # 每 100 个 batch 验证一次
--patience 5             # 连续 5 次满足条件才停止
```

---

## 📊 TensorBoard 可视化

### 启动 TensorBoard

```bash
# 本地查看
tensorboard --logdir ./experiments

# 远程服务器
tensorboard --logdir ./experiments --bind_all --port 6006

# 自定义刷新间隔
tensorboard --logdir ./experiments --reload_interval 10
```

### 远程访问

#### 方法 1: SSH 端口转发（推荐）
```bash
# 在本地终端
ssh -L 6006:localhost:6006 user@remote-server

# 在远程服务器
tensorboard --logdir ./experiments

# 本地浏览器访问
http://localhost:6006
```

#### 方法 2: 直接访问
```bash
# 远程服务器
tensorboard --logdir ./experiments --bind_all

# 本地浏览器
http://<服务器IP>:6006
```

### TensorBoard 面板

#### 1. Scalars（标量图表）

**训练损失**
- `Loss/train` - 训练 loss（线性）
- `Loss/train_db` - 训练 loss（dB）

**验证损失**
- `Loss/validation` - 验证 loss（线性）
- `Loss/validation_db` - 验证 loss（dB）

**测试损失**
- `Loss/test` - 测试 NMSE（线性）
- `Loss/test_db` - 测试 NMSE（dB）

**每端口 NMSE**
- `NMSE_per_port/port_0_db` ~ `port_3_db` - 训练时各端口 NMSE
- `NMSE_per_port_test/port_0_db` ~ `port_3_db` - 测试时各端口 NMSE

**性能指标**
- `Throughput/samples_per_sec` - 吞吐量（样本/秒）
- `Time/data_pct` - 数据生成时间占比
- `Time/forward_pct` - 前向传播时间占比
- `Time/backward_pct` - 反向传播时间占比

#### 2. HParams（超参数对比）

查看所有实验的超参数和最终性能：

| num_stages | share_weights | batch_size | snr_db | test_nmse_db | final_train_loss |
|------------|---------------|------------|--------|--------------|------------------|
| 2          | False         | 2048       | 20.0   | -15.2        | 0.045            |
| 3          | False         | 2048       | 20.0   | -16.5        | 0.038            |
| 3          | True          | 2048       | 20.0   | -15.9        | 0.041            |

**功能**:
- 按任意列排序
- 筛选特定配置
- 散点图对比
- 平行坐标图

#### 3. Graphs（计算图）

可视化模型的网络结构和计算流程。

### TensorBoard 使用技巧

#### 对比多个实验
```bash
# 训练多个配置
python Model_AIIC/test_separator.py --stages 2 --save_dir exp &
python Model_AIIC/test_separator.py --stages 3 --save_dir exp &

# 同时查看
tensorboard --logdir exp
```

在界面中：
- 勾选要对比的实验
- 调整 Smoothing 参数使曲线更平滑
- 使用正则表达式筛选：`.*share=True.*`

#### 实时监控训练
TensorBoard 默认每 30 秒刷新，训练时可以实时看到曲线更新。

---

## 🔍 超参数搜索

### 单个超参数

```bash
# 测试不同阶段数
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --stages "2,3,4" \
  --save_dir "./stage_search"
```

生成 3 个实验：
- `stage_search/stages=2_share=False/`
- `stage_search/stages=3_share=False/`
- `stage_search/stages=4_share=False/`

### 多个超参数组合

```bash
# 网格搜索
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --snr "10,30" \
  --tdl "A-30,B-100,C-300" \
  --early_stop 0.01 \
  --save_dir "./grid_search"
```

总实验数 = 3 × 2 = 6 种组合

**生成的目录结构**:
```
grid_search/
├── stages=2_share=False/
│   ├── tensorboard/
│   ├── model.pth
│   ├── model.pt
│   ├── metrics.json
│   └── training_report.md
├── stages=2_share=True/
├── stages=3_share=False/
├── stages=3_share=True/
├── stages=4_share=False/
├── stages=4_share=True/
└── search_summary.json
```

### 查看搜索结果

#### 1. TensorBoard
```bash
tensorboard --logdir ./grid_search
```

点击 "HPARAMS" 标签页，查看对比表格。

#### 2. JSON 摘要
```bash
cat grid_search/search_summary.json
```

#### 3. Markdown 报告
```bash
cat grid_search/stages=3_share=False/training_report.md
```

### 推荐的搜索策略

#### 粗搜索（快速探索）
```bash
python Model_AIIC/test_separator.py \
  --batches 500 \
  --batch_size 1024 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --early_stop 0.05 \
  --save_dir "./coarse_search"
```

#### 精细搜索（深度训练）
```bash
# 选择最佳配置后
python Model_AIIC/test_separator.py \
  --batches 10000 \
  --batch_size 4096 \
  --stages 3 \
  --share_weights False \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --early_stop 0.001 \
  --save_dir "./final_training"
```

---

## ⚡ CPU 性能优化

### CPU 核心控制

#### 使用全部核心（默认）
```bash
python Model_AIIC/test_separator.py --cpu_ratio 1.0
```

#### 使用 50% 核心
```bash
python Model_AIIC/test_separator.py --cpu_ratio 0.5
```

**输出示例**:
```
🚀 CPU Optimization:
   Available CPUs: 14
   Physical cores: 7
   CPU ratio: 0.50 (50%)
   Using threads: 3
```

### 多任务并行

在同一台机器运行多个实验：

```bash
# 终端 1: 50% 核心
python Model_AIIC/test_separator.py \
  --cpu_ratio 0.5 \
  --stages 2 \
  --save_dir exp1 &

# 终端 2: 另外 50% 核心
taskset -c 8-15 python Model_AIIC/test_separator.py \
  --cpu_ratio 0.5 \
  --stages 3 \
  --save_dir exp2 &
```

### 批大小优化

| CPU 核心数 | 推荐 batch_size | 说明 |
|-----------|----------------|------|
| 4-8       | 512-1024       | 小规模 |
| 8-16      | 1024-2048      | 中等规模 |
| 16-56     | 2048-4096      | 大规模 |
| 56+       | 4096-8192      | 服务器级 |

```bash
# 56 核服务器
python Model_AIIC/test_separator.py \
  --batch_size 4096 \
  --cpu_ratio 1.0
```

### 环境变量优先级

```bash
# 方法 1: 环境变量（最高优先级）
export OMP_NUM_THREADS=8
python Model_AIIC/test_separator.py  # 使用 8 线程

# 方法 2: 命令行参数
python Model_AIIC/test_separator.py --cpu_ratio 0.5

# 方法 3: 默认（使用所有物理核心）
python Model_AIIC/test_separator.py
```

### 性能监控

训练时会显示时间分布：

```
Batch 20/1000, Loss: 0.082974 (-10.81 dB), 
Throughput: 2343 samples/s [Data:5% Fwd:26% Bwd:68%]
```

**优化建议**:
- **Data > 10%**: 增大 `--batch_size` 或减少数据生成复杂度
- **Fwd > 50%**: 模型前向计算占主导，考虑简化模型
- **Bwd > 70%**: 正常情况，反向传播通常最耗时

---

## 🚀 模型部署

### 文件格式

每次训练后生成：

1. **model.pth** - PyTorch 权重（state dict）
2. **model.pt** - TorchScript 格式（推荐部署用）
3. **metrics.json** - 详细指标
4. **training_report.md** - Markdown 报告
5. **train_losses.npy** - 训练曲线
6. **tensorboard/** - TensorBoard 日志

### Python 部署

#### 加载 PyTorch 模型
```python
import torch
from Model_AIIC.channel_separator import ResidualRefinementSeparator

# 加载模型
checkpoint = torch.load('model.pth')
model = ResidualRefinementSeparator(
    seq_len=12,
    num_ports=4,
    hidden_dim=64,
    num_stages=3,
    share_weights_across_stages=False,
    normalize_energy=True
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
y = torch.randn(1, 12, dtype=torch.complex64)
with torch.no_grad():
    h_pred = model(y)  # [1, 4, 12]
```

#### 加载 TorchScript 模型（推荐）
```python
import torch

# 加载
model = torch.jit.load('model.pt')
model.eval()

# 推理
y = torch.randn(1, 12, dtype=torch.complex64)
with torch.no_grad():
    h_pred = model(y)  # [1, 4, 12]
```

### C++ 部署

```cpp
#include <torch/script.h>

// 加载模型
torch::jit::script::Module model = torch::jit::load("model.pt");
model.eval();

// 推理
std::vector<torch::jit::IValue> inputs;
inputs.push_back(torch::randn({1, 12}, torch::kComplexFloat));
auto output = model.forward(inputs).toTensor();
```

### MATLAB 集成

由于 ONNX 不支持复数，建议：

#### 方法 1: Python Engine（推荐）
```matlab
% 启动 Python
pyenv('Version', 'path/to/python');

% 加载模型
model = py.torch.jit.load('model.pt');
model.eval();

% 准备数据并推理
% （需要额外的类型转换代码）
```

#### 方法 2: 重新实现（生产环境）
从 `model.pth` 读取权重，用 MATLAB 重新实现网络结构。

---

## 🔧 故障排查

### 常见问题

#### 1. 内存不足
```
RuntimeError: out of memory
```

**解决方案**:
```bash
# 减小 batch_size
python Model_AIIC/test_separator.py --batch_size 512
```

#### 2. TensorBoard 端口被占用
```
TensorBoard could not bind to port 6006
```

**解决方案**:
```bash
tensorboard --logdir ./exp --port 6007
```

#### 3. 训练速度慢
**检查**:
```bash
# 查看吞吐量
Throughput: XXX samples/s [Data:X% Fwd:X% Bwd:X%]
```

**优化**:
- 增大 `--batch_size`
- 增大 `--cpu_ratio`
- 使用更简单的 TDL 配置

#### 4. Loss 不下降
**检查**:
- SNR 是否过低（试试 `--snr 20.0`）
- 模型阶段数是否足够（试试 `--stages 3`）
- 是否需要更多训练（增大 `--batches`）

#### 5. 验证 loss 波动大
**解决方案**:
```bash
# 增加验证批次数
# 在代码中修改 val_batches = 10  # 默认是 5
```

---

## 📋 完整示例

### 示例 1: 基础训练
```bash
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages 3 \
  --save_dir "./basic_training"
```

### 示例 2: 超参数搜索
```bash
python Model_AIIC/test_separator.py \
  --batches 2000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --snr "10,30" \
  --tdl "A-30,B-100,C-300" \
  --early_stop 0.01 \
  --save_dir "./hyperparam_search"

# 查看结果
tensorboard --logdir ./hyperparam_search
```

### 示例 3: 高性能训练
```bash
python Model_AIIC/test_separator.py \
  --batches 10000 \
  --batch_size 4096 \
  --stages 3 \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --early_stop 0.001 \
  --val_interval 200 \
  --cpu_ratio 1.0 \
  --save_dir "./production_training"
```

### 示例 4: 多任务并行
```bash
# 终端 1
python Model_AIIC/test_separator.py \
  --cpu_ratio 0.33 \
  --stages 2 \
  --save_dir exp &

# 终端 2
python Model_AIIC/test_separator.py \
  --cpu_ratio 0.33 \
  --stages 3 \
  --save_dir exp &

# 终端 3
python Model_AIIC/test_separator.py \
  --cpu_ratio 0.33 \
  --stages 4 \
  --save_dir exp &

# 查看所有实验
tensorboard --logdir exp
```

---

## 📊 性能基准

### 测试环境
- CPU: Intel SPR-EE (56 cores)
- 内存: 256 GB
- Batch size: 4096
- 配置: TDL-A-30, SNR=20dB

### 性能指标

| 阶段数 | 参数量 | 吞吐量 | 最终 NMSE |
|-------|--------|--------|-----------|
| 2     | 104K   | 3500 samples/s | -15 dB |
| 3     | 156K   | 3200 samples/s | -17 dB |
| 4     | 208K   | 2900 samples/s | -18 dB |

---

## 🆘 获取帮助

```bash
# 查看所有参数
python Model_AIIC/test_separator.py --help
```

---

**最后更新**: 2025-12-01  
**文档版本**: v2.0
