# 📁 实验目录结构和时间戳机制

## ✅ 自动时间戳（避免冲突）

### 问题

在同一服务器上同时运行多个训练任务时，可能会互相覆盖：

```bash
# 用户A
python train.py --model_config separator1_default

# 用户B（同时）
python train.py --model_config separator1_default

# ❌ 结果：会覆盖相同的目录！
```

---

### 解决方案：自动时间戳

**每次训练自动创建带时间戳的实验目录！**

```bash
python train.py --model_config separator1_default --training_config default
```

**输出**：
```
================================================================================
🚀 Experiment: 20251212_103045_separator1_default_default
================================================================================
   Save directory: ./experiments_refactored/20251212_103045_separator1_default_default
   Timestamp: 20251212_103045
================================================================================
```

---

## 📂 目录结构

### 完整层次结构

```
./experiments_refactored/
│
├── 20251212_103045_separator1_default_default/  ← 实验1（时间戳）
│   ├── separator1_hd64_stages2_depth3/          ← 模型配置1
│   │   ├── model.pth
│   │   ├── config.yaml
│   │   ├── training_history.png
│   │   └── checkpoints/
│   ├── separator1_hd128_stages3_depth4/         ← 模型配置2
│   │   ├── model.pth
│   │   ├── config.yaml
│   │   └── training_history.png
│   ├── evaluation_results/                      ← 评估结果（所有模型）
│   │   ├── evaluation_results.json
│   │   └── evaluation_results.npy
│   ├── plots/                                   ← 图表（所有模型）
│   │   ├── nmse_vs_snr_TDL_A_30.png
│   │   ├── nmse_vs_snr_TDL_B_100.png
│   │   └── nmse_vs_snr_combined.png
│   └── TRAINING_REPORT.md                       ← 训练报告
│
└── 20251212_105123_separator1_small_quick_test/ ← 实验2（不同时间戳）
    ├── separator1_hd32_stages2_depth3/
    │   ├── model.pth
    │   └── config.yaml
    ├── evaluation_results/
    │   ├── evaluation_results.json
    │   └── evaluation_results.npy
    ├── plots/
    │   └── nmse_vs_snr_combined.png
    └── TRAINING_REPORT.md
```

---

### 层次说明

#### 第1层：实验目录（带时间戳）

```
{timestamp}_{model_config}_{training_config}/
```

**格式**：`YYYYMMDD_HHMMSS_{model_config}_{training_config}`

**示例**：
- `20251212_103045_separator1_default_default`
- `20251212_105123_separator1_grid_search_quick_test`
- `20251212_110000_separator2_default_long_training`

**优点**：
- ✅ 自动避免冲突
- ✅ 按时间排序
- ✅ 可追溯
- ✅ 便于管理

---

#### 第2层：模型配置目录

```
{model_type}_hd{hidden_dim}_stages{num_stages}_depth{mlp_depth}/
```

**示例**：
- `separator1_hd64_stages2_depth3`
- `separator1_hd128_stages3_depth4`
- `separator2_hd64_stages2_depth3`

**内容**：
- `model.pth` - 训练好的模型权重
- `config.yaml` - 模型配置
- `training_history.png` - 训练曲线
- `checkpoints/` - 训练检查点（可选）

---

#### 第3层：共享资源（实验级别）

```
evaluation_results/  ← 评估结果
plots/               ← 图表
TRAINING_REPORT.md   ← 报告
```

**特点**：
- ✅ 所有模型共享
- ✅ 便于对比
- ✅ 避免重复

---

## 🎯 使用示例

### 1. 单模型训练

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda
```

**目录**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
└── separator1_hd64_stages2_depth3/
    ├── model.pth
    ├── config.yaml
    └── training_history.png
```

---

### 2. Grid Search

```bash
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
    --device cuda
```

**目录**：
```
./experiments_refactored/20251212_103045_separator1_grid_search_default/
├── separator1_hd32_stages2_depth3/
│   └── model.pth
├── separator1_hd64_stages2_depth3/
│   └── model.pth
├── separator1_hd128_stages2_depth3/
│   └── model.pth
├── separator1_hd64_stages3_depth3/
│   └── model.pth
└── ... (18 configurations)
```

---

### 3. 训练 + 评估 + 绘图

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**目录**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
├── separator1_hd64_stages2_depth3/
│   ├── model.pth
│   └── config.yaml
├── evaluation_results/              ← ✅ 评估结果
│   ├── evaluation_results.json
│   └── evaluation_results.npy
├── plots/                           ← ✅ 图表
│   ├── nmse_vs_snr_TDL_A_30.png
│   ├── nmse_vs_snr_TDL_B_100.png
│   └── nmse_vs_snr_combined.png
└── TRAINING_REPORT.md               ← ✅ 报告
```

---

## 🔄 并发训练（无冲突）

### 场景：同一服务器，同时运行多个训练

#### 用户A

```bash
# 10:30:45 启动
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda:0
```

**目录**：`20251212_103045_separator1_default_default/`

---

#### 用户B

```bash
# 10:51:23 启动（同时）
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda:1
```

**目录**：`20251212_105123_separator1_default_default/`

---

#### 结果

```
./experiments_refactored/
├── 20251212_103045_separator1_default_default/  ← 用户A
│   └── ...
└── 20251212_105123_separator1_default_default/  ← 用户B
    └── ...
```

**✅ 完全独立，无冲突！**

---

## 📊 实验管理

### 查看所有实验

```bash
ls -lt experiments_refactored/
```

**输出**（按时间倒序）：
```
20251212_110000_separator2_default_long_training/
20251212_105123_separator1_default_default/
20251212_103045_separator1_default_default/
20251211_153020_separator1_grid_search_default/
```

---

### 对比两个实验

```bash
# 实验1
python evaluate_models.py \
    --exp_dir ./experiments_refactored/20251212_103045_separator1_default_default \
    --device cuda

# 实验2
python evaluate_models.py \
    --exp_dir ./experiments_refactored/20251212_105123_separator1_default_default \
    --device cuda
```

---

### 清理旧实验

```bash
# 删除7天前的实验
find experiments_refactored/ -maxdepth 1 -type d -mtime +7 -exec rm -rf {} \;

# 或手动删除
rm -rf experiments_refactored/20251205_*
```

---

## 🎨 自定义保存目录（可选）

### 方法1：指定基础目录

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --save_dir /data/experiments  # 自定义基础路径
```

**结果**：
```
/data/experiments/20251212_103045_separator1_default_default/
```

---

### 方法2：完全自定义（高级）

如果你想**完全控制**目录名（不使用时间戳），可以修改代码：

```python
# train.py (手动修改)
# 注释掉自动时间戳逻辑
# args.save_dir = str(base_save_dir / experiment_name)
```

**不推荐**：容易冲突

---

## ✅ 优势总结

| 特性 | 说明 |
|------|------|
| **自动时间戳** | 每次训练自动创建唯一目录 |
| **无需手动命名** | 自动生成描述性名称 |
| **并发安全** | 多个训练互不干扰 |
| **可追溯** | 时间戳记录训练时间 |
| **层次清晰** | 实验 → 模型 → 资源 |
| **便于管理** | 按时间排序，易于清理 |

---

## 🚀 立即使用

```bash
# 无需额外配置，直接运行即可
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

**自动创建**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
├── separator1_hd64_stages2_depth3/
│   └── model.pth
├── evaluation_results/
├── plots/
└── TRAINING_REPORT.md
```

**再运行一次（不同参数）**：
```bash
python train.py \
    --model_config separator1_small \
    --training_config quick_test \
    --device cuda
```

**新目录**：
```
./experiments_refactored/20251212_103520_separator1_small_quick_test/
```

**✅ 完全独立，无冲突！**
