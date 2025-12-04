# SRS Multi-Port Channel Separator - 完整使用指南# Model_AIIC 训练与使用指南# Channel Separator Models for SRS Multi-Port Estimation



> **版本**: v2.0  

> **最后更新**: 2025-12-03  

> **维护**: 本文档是 Model_AIIC 的**唯一完整文档**，集成了所有功能说明**最后更新**: 2025-12-01  ## 🎯 Quick Start



---**版本**: v2.0



## 目录### Test Models



- [1. 项目概述](#1-项目概述)---

- [2. 快速开始](#2-快速开始)

- [3. 模型训练](#3-模型训练)```bash

- [4. 性能评估](#4-性能评估)

- [5. 复杂度分析](#5-复杂度分析)## 📚 目录cd Model_AIIC

- [6. 结果可视化](#6-结果可视化)

- [7. 参数配置](#7-参数配置)

- [8. 技术细节](#8-技术细节)

- [9. 常见问题](#9-常见问题)1. [快速开始](#快速开始)# Test all models



---2. [test_separator.py 使用指南](#test_separatorpy-使用指南)python channel_separator.py



## 1. 项目概述3. [TensorBoard 可视化](#tensorboard-可视化)



### 1.1 问题定义4. [超参数搜索](#超参数搜索)# Test with data generator integration



在 5G NR SRS 系统中，多个端口的信道响应在时域混叠：5. [CPU 性能优化](#cpu-性能优化)python test_separator.py --model mlp --epochs 10



```6. [模型部署](#模型部署)python test_separator.py --model residual --epochs 20

y = sum(circshift(h_p, pos_p)) + noise

7. [故障排查](#故障排查)python test_separator.py --model hinted --epochs 10

其中:

- y: 接收信号 (长度 12)```

- h_p: 第 p 个端口的信道响应

- pos_p: 第 p 个端口的位置偏移---

- circshift: 循环移位

```## 📁 Files



### 1.2 解决方案## 🚀 快速开始



使用**残差细化分离器** (Residual Refinement Separator)：- `channel_separator.py` - Model implementations (3 variants)



- **输入**: 混叠信号 y (复数，12 点)### 基础训练- `test_separator.py` - Integration test with data generator

- **输出**: 分离后的信道 h_0, h_1, ..., h_P (已包含移位)

- **架构**: 多阶段残差细化，每端口独立 MLP- `INTEGRATION_GUIDE.md` - Detailed integration guide (Chinese)

- **复数处理**: 实部虚部分离 (ReLU 不支持复数)

```bash

### 1.3 核心特性

python Model_AIIC/test_separator.py \## 🏗️ Models

- ✅ **多端口**: 支持 3-6 端口 (由 `--ports` 配置)

- ✅ **多阶段**: 2-4 个 stage (可配置)  --batches 1000 \

- ✅ **权重共享**: 可选跨 stage 共享

- ✅ **随机时延**: ±256Tc 随机偏移  --batch_size 2048 \### 1. SimpleMLPSeparator

- ✅ **TDL 信道**: A/B/C 型，可配时延扩展

- ✅ **灵活 SNR**: 固定/随机/每端口  --stages 3 \- **Parameters**: ~20K

- ✅ **早停**: 自动停止训练

- ✅ **完整评估**: 多 SNR/TDL 性能分析  --save_dir "./my_training"- **Best for**: Fast training, simple baseline

- ✅ **复杂度统计**: FLOPs/MACs/参数量

```

---

### 2. ResidualRefinementSeparator  

## 2. 快速开始

### 查看训练过程- **Parameters**: ~15K

### 2.1 环境配置

- **Best for**: Iterative refinement with residual correction

```bash

# 克隆并安装```bash

git clone https://github.com/YiLiangBJ/SRS_AI.git

cd SRS_AI# 启动 TensorBoard### 3. PositionHintedSeparator

python -m venv .venv

source .venv/bin/activate  # Linux/Mactensorboard --logdir ./my_training- **Parameters**: ~25K  

# .venv\Scripts\activate   # Windows

pip install -r requirements.txt- **Best for**: Stable training with position hints

```

# 在浏览器打开

### 2.2 五分钟示例

http://localhost:6006## 📊 Problem

```bash

# 1. 训练模型```

python Model_AIIC/test_separator.py \

  --batches 50 \$$y = \sum_{p \in P} \text{circshift}(h_p, p) + noise$$

  --batch_size 128 \

  --stages "2" \---

  --ports "0,3,6,9" \

  --save_dir "./quick_test"**Goal**: Separate mixed signal into individual shifted components



# 2. 评估性能

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./quick_test \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./quick_eval
```

# 3. 绘制曲线

```bash
python Model_AIIC/plot_results.py \
  --input ./quick_eval \
  --layout subplots_tdl
```

### 2.3 六端口模型示例 ⭐

```bash
# 1. 训练 6 端口模型
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --ports "0,2,4,6,8,10" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./out6ports"

# 2. 评估 6 端口模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 3. 绘制对比图
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl

# 4. 查看最佳模型
cat ./out6ports_eval/evaluation_results.json
```

**注意**: 
- `--ports "0,2,4,6,8,10"` 指定 6 个端口位置
- 模型会自动识别端口数并生成对应的测试数据
- 评估时使用较小的 `batch_size` (100) 以避免内存问题

---

## 📖 test_separator.py 使用指南

## 🔗 Integration

### 命令行参数

See `INTEGRATION_GUIDE.md` for detailed integration steps with existing codebase.

python Model_AIIC/plot_results.py --input ./quick_eval

| 参数 | 类型 | 默认值 | 说明 |

# 4. 分析复杂度

python Model_AIIC/analyze_complexity.py \|------|------|--------|------|Quick summary:

  --stages "2" \

  --num_ports 4 \| `--batches` | int | 10000 | 训练批次数 |1. Add to `data_generator.py` - data generation

  --output ./quick_complexity

```| `--batch_size` | int | 2048 | 批大小（建议 2048-4096） |2. Modify `trainMLPmmse.py` - training loop  



---| `--stages` | str | "3" | 模型阶段数 |3. Update `evaluate_performance.py` - evaluation



## 3. 模型训练| `--share_weights` | str | "False" | 是否共享权重 |



### 3.1 基本命令| `--snr` | str | "20.0" | 信噪比配置 |## ✅ Next Steps



```bash| `--tdl` | str | "A-30" | TDL 信道配置 |

python Model_AIIC/test_separator.py \

  --batches <训练批次> \| `--early_stop` | float | None | 早停阈值 |1. Run quick tests (above)

  --batch_size <批大小> \

  --stages <阶段数> \| `--val_interval` | int | 100 | 验证间隔 |2. Read `INTEGRATION_GUIDE.md`

  --share_weights <权重共享> \

  --ports <端口位置> \| `--patience` | int | 5 | 早停耐心值 |3. Choose integration approach

  --snr <SNR配置> \

  --tdl <TDL配置> \| `--save_dir` | str | None | 保存目录 |4. Start with data generation modification

  --save_dir <保存目录>

```| `--cpu_ratio` | float | 1.0 | CPU 核心使用比例 |



### 3.2 核心参数### SNR 配置



| 参数 | 默认值 | 说明 | 示例 |#### 固定 SNR（所有端口相同）

|------|--------|------|------|```bash

| `--batches` | 100 | 训练批次数 | `1000` |--snr "20.0"

| `--batch_size` | 32 | 批大小 | `2048` |```

| `--stages` | "3" | 阶段数 | `"2,3,4"` |

| `--share_weights` | "False" | 权重共享 | `"True,False"` |#### SNR 范围（随机采样）

| `--ports` | "0,3,6,9" | 端口位置 | `"0,2,4,6,8,10"` |```bash

| `--snr` | "20.0" | SNR配置 | `"10,30"` |--snr "0,30"  # 每个样本从 [0, 30] dB 均匀采样

| `--tdl` | "A-30" | TDL配置 | `"A-30,B-100"` |```

| `--save_dir` | None | 保存目录 | `"./exp"` |

#### 每端口不同 SNR

### 3.3 端口配置```bash

--snr "[15,18,20,22]"  # 端口 0-3 分别使用固定 SNR

端口数由 `--ports` 参数的长度决定：```



```bash### TDL 信道配置

# 4 端口 (默认)

--ports "0,3,6,9"#### 单一配置

```bash

# 6 端口--tdl "A-30"   # TDL-A，30ns RMS delay spread

--ports "0,2,4,6,8,10"--tdl "B-100"  # TDL-B，100ns RMS delay spread

--tdl "C-300"  # TDL-C，300ns RMS delay spread

# 3 端口```

--ports "0,4,8"

#### 多配置（随机选择）

# 自定义 5 端口```bash

--ports "0,2,4,7,9"--tdl "A-30,B-100,C-300"  # 每个样本随机选择一种

``````



**重要**: ### 早停配置

- 位置必须在 [0, 11] 范围内

- 端口数 = `len(ports.split(','))````bash

- 模型自动适应端口数--early_stop 0.01        # 当 validation loss < 0.01 时触发

--val_interval 100       # 每 100 个 batch 验证一次

### 3.4 SNR 配置--patience 5             # 连续 5 次满足条件才停止

```

```bash

# 固定 SNR (所有端口)---

--snr "20.0"

## 📊 TensorBoard 可视化

# 随机范围 (每样本随机)

--snr "10,30"  # [10, 30] dB### 启动 TensorBoard



# 每端口固定 (4 端口)```bash

--snr "[15,18,20,22]"# 本地查看

```tensorboard --logdir ./experiments



### 3.5 TDL 配置# 远程服务器

tensorboard --logdir ./experiments --bind_all --port 6006

```bash

# 单一 TDL# 自定义刷新间隔

--tdl "A-30"   # TDL-A, 30nstensorboard --logdir ./experiments --reload_interval 10

--tdl "B-100"  # TDL-B, 100ns```

--tdl "C-300"  # TDL-C, 300ns

### 远程访问

# 随机 TDL (每样本随机选择)

--tdl "A-30,B-100,C-300"#### 方法 1: SSH 端口转发（推荐）

``````bash

# 在本地终端

### 3.6 网格搜索ssh -L 6006:localhost:6006 user@remote-server



训练多个配置组合：# 在远程服务器

tensorboard --logdir ./experiments

```bash

python Model_AIIC/test_separator.py \# 本地浏览器访问

  --batches 1000 \http://localhost:6006

  --batch_size 2048 \```

  --stages "2,3,4" \

  --share_weights "True,False" \#### 方法 2: 直接访问

  --ports "0,3,6,9" \```bash

  --snr "10,30" \# 远程服务器

  --tdl "A-30,B-100,C-300" \tensorboard --logdir ./experiments --bind_all

  --early_stop 0.01 \

  --save_dir "./grid_search"# 本地浏览器

```http://<服务器IP>:6006

```

**输出**: stages(3) × share_weights(2) = 6 个模型

### TensorBoard 面板

### 3.7 训练输出

#### 1. Scalars（标量图表）

```

save_dir/**训练损失**

└── stages=3_share=False/- `Loss/train` - 训练 loss（线性）

    ├── model.pth              # PyTorch 权重- `Loss/train_db` - 训练 loss（dB）

    ├── model.pt               # TorchScript

    ├── metrics.json           # 训练指标**验证损失**

    ├── train_losses.npy       # 损失曲线- `Loss/validation` - 验证 loss（线性）

    ├── training_report.md     # 报告- `Loss/validation_db` - 验证 loss（dB）

    └── tensorboard/           # TensorBoard 日志

```**测试损失**

- `Loss/test` - 测试 NMSE（线性）

### 3.8 TensorBoard 监控- `Loss/test_db` - 测试 NMSE（dB）



```bash**每端口 NMSE**

tensorboard --logdir ./save_dir- `NMSE_per_port/port_0_db` ~ `port_3_db` - 训练时各端口 NMSE

# 访问 http://localhost:6006- `NMSE_per_port_test/port_0_db` ~ `port_3_db` - 测试时各端口 NMSE

```

**性能指标**

---- `Throughput/samples_per_sec` - 吞吐量（样本/秒）

- `Time/data_pct` - 数据生成时间占比

## 4. 性能评估- `Time/forward_pct` - 前向传播时间占比

- `Time/backward_pct` - 反向传播时间占比

### 4.1 基本命令

#### 2. HParams（超参数对比）

```bash

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir <训练目录> \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output <输出目录>
```

查看所有实验的超参数和最终性能：

| num_stages | share_weights | batch_size | snr_db | test_nmse_db | final_train_loss |
|------------|---------------|------------|--------|--------------|------------------|
| 2          | False         | 2048       | 20.0   | -15.2        | 0.045            |
| 3          | False         | 2048       | 20.0   | -16.5        | 0.038            |
| 3          | True          | 2048       | 20.0   | -15.9        | 0.041            |

```

**功能**:

### 4.2 评估参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--exp_dir` | str | **必需** | 训练实验目录（包含多个模型子目录） |
| `--tdl` | str | "A-30,B-100,C-300" | TDL 信道配置（逗号分隔） |
| `--snr_range` | str | "30:-3:0" | SNR 范围 (start:step:end，单位 dB) |
| `--num_batches` | int | 10 | 每个 SNR 点的评估批次数 |
| `--batch_size` | int | 100 | 批大小（推荐 100-200） |
| `--output` | str | ./evaluation_results | 输出目录 |
| `--models` | str | 所有 | 指定模型（逗号分隔，可选） |

**示例**:

```bash
# 4 端口模型评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./full_exp \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./full_eval

# 6 端口模型评估（推荐使用较小的 batch_size）
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 单个 TDL 快速评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./quick_test \
  --tdl "A-30" \
  --snr_range "30:-5:0" \
  --num_batches 5 \
  --batch_size 50 \
  --output ./quick_eval
```

**注意事项**:
- `--exp_dir` 应指向包含多个模型子目录的父目录
- 评估时会自动读取每个模型的 `pos_values` 配置
- 6 端口模型建议使用较小的 `batch_size` (50-100)
- SNR 范围格式: `"起始:步长:结束"` (例如 `"30:-3:0"` 表示 30, 27, 24, ..., 3, 0)

#### 对比多个实验

**总样本数** = `num_batches × batch_size````bash

# 训练多个配置

### 4.3 SNR 范围python Model_AIIC/test_separator.py --stages 2 --save_dir exp &

python Model_AIIC/test_separator.py --stages 3 --save_dir exp &

```bash

# 范围格式 (推荐)# 同时查看

--snr_range "30:-3:0"  # [30, 27, 24, ..., 0]tensorboard --logdir exp

```

# 列表格式

--snr_range "30,20,10,0"在界面中：

```- 勾选要对比的实验

- 调整 Smoothing 参数使曲线更平滑

### 4.4 评估示例- 使用正则表达式筛选：`.*share=True.*`



#### 完整评估#### 实时监控训练

TensorBoard 默认每 30 秒刷新，训练时可以实时看到曲线更新。

```bash

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./grid_search \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./full_eval
```

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

#### 快速测试

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./stage_search \
  --tdl "A-30" \
  --snr_range "30:-3:0" \
  --num_batches 5 \
  --batch_size 100 \
  --output ./stage_eval
```

  --snr_range "30,15,0" \- `stage_search/stages=2_share=False/`

  --num_batches 2 \- `stage_search/stages=3_share=False/`

  --batch_size 50 \- `stage_search/stages=4_share=False/`

  --output ./quick_eval

```### 多个超参数组合



### 4.5 评估输出```bash

# 网格搜索

```python Model_AIIC/test_separator.py \

output_dir/  --batches 1000 \

├── evaluation_results.json    # JSON (人类可读)  --batch_size 2048 \

└── evaluation_results.npy     # NumPy (快速加载)  --stages "2,3,4" \

```  --share_weights "True,False" \

  --snr "10,30" \

### 4.6 读取结果  --tdl "A-30,B-100,C-300" \

  --early_stop 0.01 \

```python  --save_dir "./grid_search"

import numpy as np```



# 加载总实验数 = 3 × 2 = 6 种组合

results = np.load('evaluation_results.npy', allow_pickle=True).item()

**生成的目录结构**:

# 查看```

print("模型:", list(results['models'].keys()))grid_search/

print("SNR:", results['config']['snr_list'])├── stages=2_share=False/

│   ├── tensorboard/

# 提取数据│   ├── model.pth

model = 'stages=3_share=False'│   ├── model.pt

tdl = 'A-30'│   ├── metrics.json

nmse_db = results['models'][model]['tdl_results'][tdl]['nmse_db']│   └── training_report.md

```├── stages=2_share=True/

├── stages=3_share=False/

---├── stages=3_share=True/

├── stages=4_share=False/

## 5. 复杂度分析├── stages=4_share=True/

└── search_summary.json

### 5.1 基本命令```



```bash### 查看搜索结果

python Model_AIIC/analyze_complexity.py \

  --stages <阶段数> \#### 1. TensorBoard

  --share_weights <权重共享> \```bash

  --num_ports <端口数> \tensorboard --logdir ./grid_search

  --batch_size <批大小> \```

  --output <输出目录>

```点击 "HPARAMS" 标签页，查看对比表格。



### 5.2 参数说明#### 2. JSON 摘要

```bash

| 参数 | 默认值 | 说明 |cat grid_search/search_summary.json

|------|--------|------|```

| `--stages` | "2,3,4" | 阶段数列表 |

| `--share_weights` | "True,False" | 权重共享 |#### 3. Markdown 报告

| `--num_ports` | 4 | 端口数 |```bash

| `--batch_size` | 1 | 推理批大小 |cat grid_search/stages=3_share=False/training_report.md

| `--output` | ./model_complexity | 输出 |```



### 5.3 分析示例### 推荐的搜索策略



#### 4 端口模型#### 粗搜索（快速探索）

```bash

```bashpython Model_AIIC/test_separator.py \

python Model_AIIC/analyze_complexity.py \  --batches 500 \

  --stages "2,3,4" \  --batch_size 1024 \

  --share_weights "True,False" \  --stages "2,3,4" \

  --num_ports 4 \  --share_weights "True,False" \

  --output ./complexity_4p  --early_stop 0.05 \

```  --save_dir "./coarse_search"

```

#### 6 端口模型

#### 精细搜索（深度训练）

```bash```bash

python Model_AIIC/analyze_complexity.py \# 选择最佳配置后

  --stages "2,3,4" \python Model_AIIC/test_separator.py \

  --share_weights "True,False" \  --batches 10000 \

  --num_ports 6 \  --batch_size 4096 \

  --output ./complexity_6p  --stages 3 \

```  --share_weights False \

  --snr "0,30" \

### 5.4 统计指标  --tdl "A-30,B-100,C-300" \

  --early_stop 0.001 \

| 指标 | 含义 | 用途 |  --save_dir "./final_training"

|------|------|------|```

| **FLOPs** | 浮点运算数 (乘+加) | 推理速度 |

| **MACs** | 乘加操作数 | 硬件设计 |---

| **参数量** | 权重数量 | 模型大小 |

| **内存** | 总内存占用 | 部署成本 |## ⚡ CPU 性能优化



**关系**: FLOPs ≈ 2 × MACs### CPU 核心控制



### 5.5 输出文件#### 使用全部核心（默认）

```bash

```python Model_AIIC/test_separator.py --cpu_ratio 1.0

output_dir/```

├── complexity_analysis.json       # 详细数据

└── complexity_comparison.md       # 对比表格#### 使用 50% 核心

``````bash

python Model_AIIC/test_separator.py --cpu_ratio 0.5

### 5.6 复杂度结果 (4 端口)```



| stages | share | 参数量 | FLOPs | MACs |**输出示例**:

|--------|-------|--------|-------|------|```

| 2 | True | 52K | 205K | 102K |🚀 CPU Optimization:

| 2 | False | 105K | 205K | 102K |   Available CPUs: 14

| 3 | True | 52K | 308K | 154K |   Physical cores: 7

| 3 | False | 157K | 308K | 154K |   CPU ratio: 0.50 (50%)

   Using threads: 3

### 5.7 权重共享的影响```



| 方面 | share=True | share=False |### 多任务并行

|------|------------|-------------|

| 参数量 | ✅ 少 (÷stages) | 多 |在同一台机器运行多个实验：

| FLOPs | ⚖️ **相同** | ⚖️ **相同** |

| 推理速度 | ⚖️ 相同 | ⚖️ 相同 |```bash

| 性能 | ⚠️ 可能略低 | ✅ 更好 |# 终端 1: 50% 核心

python Model_AIIC/test_separator.py \

**关键**: 权重共享只影响参数量，不影响计算量！  --cpu_ratio 0.5 \

  --stages 2 \

---  --save_dir exp1 &



## 6. 结果可视化

### 6.1 基本命令

```bash
python Model_AIIC/plot_results.py \
  --input <评估结果目录> \
  --layout <布局类型> \
  --filter_models <模型过滤> \
  --filter_tdl <TDL过滤>
```

### 6.2 布局类型

#### single - 单图

所有配置在一张图

```bash
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout single
```

#### subplots_tdl - 按 TDL 分图 ⭐

每个 TDL 一个子图（推荐）

```bash
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

#### subplots_model - 按模型分图

每个模型配置一个子图

```bash
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout subplots_model
```

### 6.3 常用示例

#### 4 端口模型可视化

```bash
# 评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./full_exp \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./full_eval

# 绘图（默认布局）
python Model_AIIC/plot_results.py \
  --input ./full_eval

# 绘图（按 TDL 分图）
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout subplots_tdl
```

#### 6 端口模型可视化 ⭐

```bash
# 评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 绘图（简单命令）
python Model_AIIC/plot_results.py \
  --input out6ports_eval

# 绘图（按 TDL 分图，推荐）
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl

# 绘图（按模型分图）
python Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_model
```

### 6.4 过滤选项

#### 只显示特定模型

```bash
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout subplots_tdl \
  --filter_models "stages=3_share=False,stages=4_share=False"
```

#### 只显示特定 TDL

```bash
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout subplots_model \
  --filter_tdl "A-30,B-100"
```

### 6.5 输出说明

绘图脚本会在输入目录中生成图片：

```
<输出目录>/
├── nmse_comparison_<layout>.png    # NMSE 对比图
├── nmse_comparison_<layout>.pdf    # PDF 版本
└── evaluation_results.json         # 原始数据
```

**图表标题信息**:
- 模型配置（stages, share_weights, loss_type）
- 端口配置（如 "Ports: [0,2,4,6,8,10]" 或 "Ports: [0,3,6,9]"）
- TDL 信道类型
- SNR 范围

---

## 7. CPU 性能优化

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

python Model_AIIC/plot_results.py \

  --input ./full_eval \# 方法 2: 命令行参数

  --layout subplots_tdlpython Model_AIIC/test_separator.py --cpu_ratio 0.5

```

# 方法 3: 默认（使用所有物理核心）

#### subplots_model - 按模型分图python Model_AIIC/test_separator.py

```

每个模型一个子图

### 性能监控

```bash

python Model_AIIC/plot_results.py \训练时会显示时间分布：

  --input ./full_eval \

  --layout subplots_model```

```Batch 20/1000, Loss: 0.082974 (-10.81 dB), 

Throughput: 2343 samples/s [Data:5% Fwd:26% Bwd:68%]

### 6.3 过滤绘图```



```bash**优化建议**:

# 只绘制特定模型- **Data > 10%**: 增大 `--batch_size` 或减少数据生成复杂度

python Model_AIIC/plot_results.py \- **Fwd > 50%**: 模型前向计算占主导，考虑简化模型

  --input ./full_eval \- **Bwd > 70%**: 正常情况，反向传播通常最耗时

  --filter_models "stages=2_share=False,stages=3_share=False"

---

# 只绘制特定 TDL

python Model_AIIC/plot_results.py \## 🚀 模型部署

  --input ./full_eval \

  --filter_tdl "A-30,B-100"### 文件格式

```

每次训练后生成：

### 6.4 输出文件

1. **model.pth** - PyTorch 权重（state dict）

```2. **model.pt** - TorchScript 格式（推荐部署用）

output_dir/3. **metrics.json** - 详细指标

├── nmse_vs_snr_single.png         # 单图4. **training_report.md** - Markdown 报告

├── nmse_vs_snr_single.pdf         # 单图 PDF5. **train_losses.npy** - 训练曲线

├── nmse_vs_snr_subplots.png       # 分图6. **tensorboard/** - TensorBoard 日志

└── nmse_vs_snr_by_model.png       # 按模型

```### Python 部署



---#### 加载 PyTorch 模型

```python

## 7. 参数配置import torch

from Model_AIIC.channel_separator import ResidualRefinementSeparator

### 7.1 模型架构

# 加载模型

| 参数 | 默认值 | 范围 | 说明 |checkpoint = torch.load('model.pth')

|------|--------|------|------|model = ResidualRefinementSeparator(

| `seq_len` | 12 | 固定 | 序列长度 |    seq_len=12,

| `num_ports` | 4 | 3-6 | 端口数 (由 ports 决定) |    num_ports=4,

| `hidden_dim` | 64 | 32-128 | MLP 隐藏维度 |    hidden_dim=64,

| `num_stages` | 3 | 2-4 | 细化阶段 |    num_stages=3,

| `share_weights` | False | bool | 权重共享 |    share_weights_across_stages=False,

    normalize_energy=True

### 7.2 训练参数)

model.load_state_dict(checkpoint['model_state_dict'])

| 参数 | 默认值 | 推荐 | 说明 |model.eval()

|------|--------|------|------|

| `num_batches` | 100 | 1000 | 训练批次 |# 推理

| `batch_size` | 32 | 2048 | 批大小 |y = torch.randn(1, 12, dtype=torch.complex64)

| `learning_rate` | 0.01 | 0.01 | 学习率 |with torch.no_grad():

| `early_stop` | None | 0.01 | 早停阈值 |    h_pred = model(y)  # [1, 4, 12]

| `patience` | 5 | 5 | 耐心值 |```



### 7.3 推荐配置#### 加载 TorchScript 模型（推荐）

```python

#### 快速验证 (~2 分钟)import torch



```bash# 加载

--batches 50 --batch_size 128 --stages "2" --ports "0,3,6,9"model = torch.jit.load('model.pt')

```model.eval()



#### 标准训练 (~15 分钟)# 推理

y = torch.randn(1, 12, dtype=torch.complex64)

```bashwith torch.no_grad():

--batches 1000 --batch_size 2048 --stages "3" --ports "0,3,6,9"    h_pred = model(y)  # [1, 4, 12]

``````



#### 网格搜索 (~1 小时)### C++ 部署



```bash```cpp

--batches 1000 --batch_size 2048 --stages "2,3,4" --share_weights "True,False"#include <torch/script.h>

```

// 加载模型

---torch::jit::script::Module model = torch::jit::load("model.pt");

model.eval();

## 8. 技术细节

// 推理

### 8.1 模型架构std::vector<torch::jit::IValue> inputs;

inputs.push_back(torch::randn({1, 12}, torch::kComplexFloat));

```pythonauto output = model.forward(inputs).toTensor();

class ResidualRefinementSeparator:```

    """

    多阶段残差细化分离器### MATLAB 集成

    

    输入: y (B, 12) complex由于 ONNX 不支持复数，建议：

    输出: h (B, P, 12) complex

    """#### 方法 1: Python Engine（推荐）

    - 每端口独立 MLP```matlab

    - 可选权重共享% 启动 Python

    - 残差连接耦合pyenv('Version', 'path/to/python');

```

% 加载模型

### 8.2 复数处理model = py.torch.jit.load('model.pt');

model.eval();

```python

class ComplexMLP:% 准备数据并推理

    """% （需要额外的类型转换代码）

    实部虚部分离处理```

    

    输入: [real, imag] 拼接 → (B, 24)#### 方法 2: 重新实现（生产环境）

    输出: real, imag 分别 → (B, 12)从 `model.pth` 读取权重，用 MATLAB 重新实现网络结构。

    """

    mlp_real: Linear(24, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 12)---

    mlp_imag: 相同结构

```## 🔧 故障排查



### 8.3 TDL 信道### 常见问题



```python#### 1. 内存不足

# TDL-A: 低时延 (室内)```

TDLChannel(model='A', delay_spread=30e-9)RuntimeError: out of memory

```

# TDL-B: 中等 (城市)

TDLChannel(model='B', delay_spread=100e-9)**解决方案**:

```bash

# TDL-C: 高时延 (郊区)# 减小 batch_size

TDLChannel(model='C', delay_spread=300e-9)python Model_AIIC/test_separator.py --batch_size 512

``````



### 8.4 随机时延#### 2. TensorBoard 端口被占用

```

每个端口 ±256Tc 随机时延：TensorBoard could not bind to port 6006

```

```python

# Tc = 1/(480e3*4096) ≈ 0.509 ns**解决方案**:

offset_Tc = uniform(-256, 256)```bash

h_shifted = IFFT(FFT(h) * exp(j*2π*k*offset/L))tensorboard --logdir ./exp --port 6007

``````



### 8.5 性能指标#### 3. 训练速度慢

**检查**:

```python```bash

# NMSE (Normalized MSE)# 查看吞吐量

mse = |h_pred - h_true|²Throughput: XXX samples/s [Data:X% Fwd:X% Bwd:X%]

signal_power = |h_true|²```

nmse = mse / signal_power

nmse_db = 10 * log10(nmse)**优化**:

```- 增大 `--batch_size`

- 增大 `--cpu_ratio`

---- 使用更简单的 TDL 配置



## 9. 常见问题#### 4. Loss 不下降

**检查**:

### Q1: 如何训练不同端口数？- SNR 是否过低（试试 `--snr 20.0`）

- 模型阶段数是否足够（试试 `--stages 3`）

```bash- 是否需要更多训练（增大 `--batches`）

# 4 端口

--ports "0,3,6,9"#### 5. 验证 loss 波动大

**解决方案**:

# 6 端口```bash

--ports "0,2,4,6,8,10"# 增加验证批次数

# 在代码中修改 val_batches = 10  # 默认是 5

# 3 端口```

--ports "0,4,8"

```---



### Q2: 权重共享有什么影响？## 📋 完整示例



- ✅ **参数量**: 减少 50-75%### 示例 1: 基础训练

- ⚖️ **FLOPs**: 完全相同```bash

- ⚖️ **推理速度**: 相同python Model_AIIC/test_separator.py \

- ⚠️ **性能**: 可能略低  --batches 1000 \

  --batch_size 2048 \

**推荐**: 不共享 (除非参数预算有限)  --stages 3 \

  --save_dir "./basic_training"

### Q3: 如何选择阶段数？```



| stages | 参数 | 性能 | 训练时间 |### 示例 2: 超参数搜索

|--------|------|------|----------|```bash

| 2 | 少 | 基础 | 快 |python Model_AIIC/test_separator.py \

| 3 | 中 | 良好 | 中 |  --batches 2000 \

| 4 | 多 | 最佳 | 慢 |  --batch_size 2048 \

  --stages "2,3,4" \

**推荐**: stages=3  --share_weights "True,False" \

  --snr "10,30" \

### Q4: 评估样本数怎么设？  --tdl "A-30,B-100,C-300" \

  --early_stop 0.01 \

```bash  --save_dir "./hyperparam_search"

# 总样本 = num_batches × batch_size

# 查看结果

# 快速 (100)tensorboard --logdir ./hyperparam_search

--num_batches 2 --batch_size 50```



# 标准 (2000)### 示例 3: 高性能训练

--num_batches 10 --batch_size 200```bash

python Model_AIIC/test_separator.py \

# 高精度 (10000)  --batches 10000 \

--num_batches 50 --batch_size 200  --batch_size 4096 \

```  --stages 3 \

  --snr "0,30" \

### Q5: 如何加速训练？  --tdl "A-30,B-100,C-300" \

  --early_stop 0.001 \

1. 增大批大小: `--batch_size 2048`  --val_interval 200 \

2. 早停: `--early_stop 0.01`  --cpu_ratio 1.0 \

3. 减少阶段: `--stages "2"`  --save_dir "./production_training"

4. 权重共享: `--share_weights "True"````



### Q6: 模型文件在哪？### 示例 4: 多任务并行

```bash

```# 终端 1

save_dir/stages=3_share=False/python Model_AIIC/test_separator.py \

├── model.pth    # ← PyTorch 权重  --cpu_ratio 0.33 \

├── model.pt     # ← TorchScript (推荐)  --stages 2 \

└── metrics.json # 指标  --save_dir exp &

```

# 终端 2

### Q7: 如何加载模型？python Model_AIIC/test_separator.py \

  --cpu_ratio 0.33 \

```python  --stages 3 \

# PyTorch  --save_dir exp &

ckpt = torch.load('model.pth')

model.load_state_dict(ckpt['model_state_dict'])# 终端 3

python Model_AIIC/test_separator.py \

# TorchScript  --cpu_ratio 0.33 \

model = torch.jit.load('model.pt')  --stages 4 \

```  --save_dir exp &



### Q8: 为什么共享权重不减少 FLOPs？# 查看所有实验

tensorboard --logdir exp

```python```

# 共享: 存 1 份参数，计算 N 次

for stage in range(N):---

    x = mlp(x)  # N 次计算

## 📊 性能基准

# 不共享: 存 N 份参数，也计算 N 次

for stage in range(N):### 测试环境

    x = mlps[stage](x)  # N 次计算- CPU: Intel SPR-EE (56 cores)

- 内存: 256 GB

# 结论: 计算量相同！- Batch size: 4096

```- 配置: TDL-A-30, SNR=20dB



### Q9: 如何对比配置？### 性能指标



```bash| 阶段数 | 参数量 | 吞吐量 | 最终 NMSE |

# 1. 训练|-------|--------|--------|-----------|

python Model_AIIC/test_separator.py --stages "2,3,4" --save_dir ./cmp| 2     | 104K   | 3500 samples/s | -15 dB |

| 3     | 156K   | 3200 samples/s | -17 dB |

# 2. 评估| 4     | 208K   | 2900 samples/s | -18 dB |

python Model_AIIC/evaluate_models.py --exp_dir ./cmp --output ./cmp_eval

---

# 3. 绘图

python Model_AIIC/plot_results.py --input ./cmp_eval## 🆘 获取帮助



# 4. 复杂度```bash

python Model_AIIC/analyze_complexity.py --stages "2,3,4" --output ./cmp_complexity# 查看所有参数

```python Model_AIIC/test_separator.py --help

```

### Q10: TensorBoard 在哪？

---

```bash

tensorboard --logdir ./save_dir**最后更新**: 2025-12-01  

# 访问 http://localhost:6006**文档版本**: v2.0

```

---

## 附录

### 代码结构

```
Model_AIIC/
├── channel_separator.py      # 模型定义
├── tdl_channel.py           # TDL 信道
├── test_separator.py        # 训练脚本 ⭐
├── evaluate_models.py       # 评估脚本 ⭐
├── plot_results.py          # 绘图脚本 ⭐
├── analyze_complexity.py    # 复杂度分析 ⭐
└── README.md                # 本文档 ⭐
```

### 完整工作流

```bash
# 1. 训练多个配置（4 端口示例）
python Model_AIIC/test_separator.py \
  --batches 1000 --batch_size 2048 \
  --stages "2,3,4" --share_weights "True,False" \
  --ports "0,3,6,9" --save_dir "./full_exp"

# 2. 训练 6 端口模型
python Model_AIIC/test_separator.py \
  --batches 1000 --batch_size 2048 \
  --stages "2,3,4" --share_weights "True,False" \
  --ports "0,2,4,6,8,10" --save_dir "./out6ports"

# 3. 评估所有模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./full_exp \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 --batch_size 200 \
  --output ./full_eval

# 4. 评估 6 端口模型（推荐参数）
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 --batch_size 100 \
  --output ./out6ports_eval

# 3. 绘制对比图
python Model_AIIC/plot_results.py \
  --input ./full_eval \
  --layout subplots_tdl

# 4. 分析复杂度
python Model_AIIC/analyze_complexity.py \
  --stages "2,3,4" --share_weights "True,False" \
  --num_ports 4 --output ./full_complexity

# 5. 对比不同端口数
python Model_AIIC/analyze_complexity.py \
  --stages "3" --num_ports "4,6" \
  --output ./port_comparison
```

### 更新日志

#### v2.0 (2025-12-03)

- ✅ 支持可配置端口数 (3-6 ports)
- ✅ 修复权重共享 FLOPs bug
- ✅ 统一文档 (单个 README.md)
- ✅ 改进参数命名
- ✅ 完善端口配置

#### v1.1 (2025-12-02)

- ✅ 添加评估/绘图/复杂度脚本
- ✅ 支持权重共享
- ✅ TensorBoard 集成

#### v1.0 (2025-11-30)

- ✅ 初始版本
- ✅ 基础训练
- ✅ TDL 支持

---

**维护说明**: 
- 本文档是 Model_AIIC 的**唯一完整文档**
- 所有功能更新都会同步到本文档
- 不再创建新的独立 Markdown 文件
- 版本号与代码保持一致

**最后更新**: 2025-12-03  
**文档版本**: 2.0  
**代码版本**: 2.0
