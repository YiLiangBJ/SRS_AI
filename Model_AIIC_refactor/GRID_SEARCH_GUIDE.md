# 🔍 Grid Search - 超参数搜索空间使用指南

## 📋 概述

新增的**Grid Search**功能允许你通过配置文件定义超参数搜索空间，自动训练和对比多个模型配置，无需手动创建大量重复的配置。

---

## 🎯 核心概念

### 1. Fixed Params（固定参数）
不参与搜索的参数，所有配置都使用相同的值。

**例如**：
```yaml
fixed_params:
  mlp_depth: 3                      # 所有配置都用3层
  share_weights_across_stages: false  # 所有配置都不共享权重
```

### 2. Search Space（搜索空间）
需要搜索的参数，会生成所有可能的组合。

**例如**：
```yaml
search_space:
  hidden_dim: [32, 64, 128]    # 3个选项
  num_stages: [2, 3, 4]        # 3个选项
  # 总共生成: 3 x 3 = 9 个配置
```

---

## 📝 配置格式

### 方式1：单一配置（向后兼容）

```yaml
# configs/model_configs.yaml
separator1_default:
  model_type: separator1
  hidden_dim: 64
  num_stages: 3
  mlp_depth: 3
```

**使用**：
```bash
python train.py --model_config separator1_default
# 训练 1 个配置
```

---

### 方式2：基础搜索空间

```yaml
separator1_grid_search:
  model_type: separator1
  fixed_params:
    mlp_depth: 3
    share_weights_across_stages: false
  search_space:
    hidden_dim: [32, 64, 128]
    num_stages: [2, 3]
```

**使用**：
```bash
python train.py --model_config separator1_grid_search
# 自动训练 3 x 2 = 6 个配置
```

**生成的配置**：
- `separator1_grid_search_hd32_stages2_...`
- `separator1_grid_search_hd32_stages3_...`
- `separator1_grid_search_hd64_stages2_...`
- `separator1_grid_search_hd64_stages3_...`
- `separator1_grid_search_hd128_stages2_...`
- `separator1_grid_search_hd128_stages3_...`

---

### 方式3：高级搜索空间（范围）

```yaml
separator1_advanced_search:
  model_type: separator1
  fixed_params:
    mlp_depth: 3
  search_space:
    hidden_dim:
      type: choice
      values: [32, 64, 128, 256]
    num_stages:
      type: range
      min: 2
      max: 5
      step: 1
```

**生成**：4 x 4 = 16 个配置

---

## 🎨 搜索空间类型

### 1. 离散值列表（最常用）
```yaml
search_space:
  hidden_dim: [32, 64, 128]
  activation_type: [relu, split_relu, mod_relu]
```

### 2. Choice（明确指定）
```yaml
search_space:
  hidden_dim:
    type: choice
    values: [32, 64, 128]
```

### 3. Range（整数范围）
```yaml
search_space:
  num_stages:
    type: range
    min: 2
    max: 5
    step: 1
  # 生成: [2, 3, 4, 5]
```

### 4. Uniform（均匀采样）
```yaml
search_space:
  learning_rate:
    type: uniform
    min: 0.001
    max: 0.1
    num_samples: 5
  # 生成: [0.001, 0.02575, 0.0505, 0.07525, 0.1]
```

### 5. LogUniform（对数均匀采样）
```yaml
search_space:
  learning_rate:
    type: loguniform
    min: 0.0001
    max: 0.1
    num_samples: 4
  # 生成: [0.0001, 0.001, 0.01, 0.1]
```

---

## 🚀 使用示例

### 示例1：快速对比隐藏维度

**配置**：
```yaml
# configs/model_configs.yaml
separator1_hd_comparison:
  model_type: separator1
  fixed_params:
    num_stages: 3
    mlp_depth: 3
    share_weights_across_stages: false
  search_space:
    hidden_dim: [32, 64, 128]
```

**训练**：
```bash
python train.py \
  --model_config separator1_hd_comparison \
  --training_config grid_search_quick
```

**输出**：
```
🔍 Search space: 3 configurations
   Fixed parameters:
     num_stages: 3
     mlp_depth: 3
     share_weights_across_stages: False
   
   Search parameters:
     hidden_dim: [32, 64, 128] (3 values)

Training Summary:
1. separator1_hd_comparison_hd128_...: NMSE -25.3 dB ⭐ Best
2. separator1_hd_comparison_hd64_...:  NMSE -23.1 dB
3. separator1_hd_comparison_hd32_...:  NMSE -20.8 dB
```

---

### 示例2：架构搜索

**配置**：
```yaml
separator1_architecture_search:
  model_type: separator1
  fixed_params:
    share_weights_across_stages: false
  search_space:
    hidden_dim: [32, 64, 128]
    num_stages: [2, 3]
    mlp_depth: [2, 3, 4]
```

**训练**：
```bash
python train.py \
  --model_config separator1_architecture_search \
  --training_config grid_search_full
```

**结果**：3 x 2 x 3 = **18 个配置**

---

### 示例3：激活函数对比（Separator2）

**配置**：
```yaml
separator2_activation_comparison:
  model_type: separator2
  fixed_params:
    hidden_dim: 64
    num_stages: 3
    mlp_depth: 3
    share_weights_across_stages: false
    onnx_mode: false
  search_space:
    activation_type: [relu, split_relu, mod_relu]
```

**训练**：
```bash
python train.py \
  --model_config separator2_activation_comparison \
  --training_config default
```

**结果**：3 个配置，对比不同激活函数的效果

---

## 📊 结果分析

### 训练输出

```bash
================================================================================
Training: separator1_grid_search
================================================================================

🔍 Search space: 6 configurations
   Name: separator1_grid_search

   Fixed parameters:
     mlp_depth: 3
     share_weights_across_stages: False

   Search parameters:
     hidden_dim: [32, 64, 128] (3 values)
     num_stages: [2, 3] (2 values)

   Total combinations: 6

────────────────────────────────────────────────────────────────────────────────
Configuration 1/6
────────────────────────────────────────────────────────────────────────────────

Configuration: separator1_grid_search_hd32_stages2_...
  Model type: separator1
  Parameters: {'hidden_dim': 32, 'num_stages': 2, 'mlp_depth': 3, ...}
  Total parameters: 18,048

🚀 Starting training on cuda
...

────────────────────────────────────────────────────────────────────────────────
Training Summary
────────────────────────────────────────────────────────────────────────────────

Total configurations trained: 6

1. separator1_grid_search_hd128_stages3_...:
   Final loss: 0.000234
   Min loss: 0.000198
   Eval NMSE: -26.3 dB
   Parameters: 217,344
   Duration: 342.1s

2. separator1_grid_search_hd64_stages3_...:
   Final loss: 0.000312
   Min loss: 0.000267
   Eval NMSE: -25.1 dB
   Parameters: 54,528
   Duration: 198.5s

...

🏆 Best configuration: separator1_grid_search_hd128_stages3_...
   NMSE: -26.3 dB

✓ All training completed!
```

---

## 💡 最佳实践

### 1. 从小规模开始

```yaml
# 先用小搜索空间快速测试
separator1_small_search:
  model_type: separator1
  fixed_params:
    mlp_depth: 3
    num_stages: 3
  search_space:
    hidden_dim: [32, 64]  # 只搜索2个值
```

```bash
python train.py \
  --model_config separator1_small_search \
  --training_config grid_search_quick  # 短训练
```

### 2. 逐步扩大搜索空间

```yaml
# 确认方向后，扩大搜索范围
separator1_medium_search:
  model_type: separator1
  fixed_params:
    mlp_depth: 3
  search_space:
    hidden_dim: [32, 64, 128]
    num_stages: [2, 3, 4]
```

### 3. 固定不重要的参数

```yaml
fixed_params:
  mlp_depth: 3              # 经验值，不搜索
  share_weights_across_stages: false  # 已知不共享更好
search_space:
  hidden_dim: [64, 128, 256]  # 只搜索关键参数
  num_stages: [3, 4]
```

---

## 🔧 命令行选项

### 基础用法

```bash
# 单一配置（向后兼容）
python train.py --model_config separator1_default

# 网格搜索
python train.py --model_config separator1_grid_search

# 多个配置对比
python train.py --model_config separator1_default,separator2_default
```

### 覆盖参数

```bash
# 覆盖训练配置
python train.py \
  --model_config separator1_grid_search \
  --training_config grid_search_quick \
  --batch_size 4096 \
  --num_batches 20000
```

---

## 📈 性能考虑

### 组合数量估算

| 参数数量 | 每参数选项数 | 总配置数 |
|---------|------------|---------|
| 2 | 3 | 3² = 9 |
| 3 | 3 | 3³ = 27 |
| 4 | 3 | 3⁴ = 81 |
| 2 | [3, 2] | 3 x 2 = 6 |
| 3 | [3, 2, 4] | 3 x 2 x 4 = 24 |

**建议**：
- ✅ 小规模：< 10 个配置
- ⚠️ 中规模：10-50 个配置
- ❌ 大规模：> 50 个配置（考虑分批或随机搜索）

### 训练时间估算

```python
# 单个配置训练时间：10分钟
# 搜索空间：18个配置
# 总时间：10 x 18 = 180分钟 = 3小时
```

**优化**：
1. 使用 `grid_search_quick` 配置（少batches）
2. 减小 batch_size
3. 使用早停（early stopping）

---

## 🎓 高级功能

### 1. 编程式使用

```python
from utils import parse_model_config, print_search_space_summary

config = {
    'model_type': 'separator1',
    'fixed_params': {'mlp_depth': 3},
    'search_space': {
        'hidden_dim': [32, 64, 128],
        'num_stages': [2, 3]
    }
}

# 解析配置
configs = parse_model_config(config)

# 打印摘要
print_search_space_summary(configs, 'my_search')

# 遍历所有配置
for cfg in configs:
    model = create_model(cfg['model_type'], cfg)
    # ... 训练 ...
```

### 2. 自定义配置名称

```python
from utils import generate_config_name

config = {'model_type': 'separator1', 'hidden_dim': 64, 'num_stages': 3}
name = generate_config_name(config, base_name='exp1')
# 输出: 'exp1_hd64_stages3'
```

---

## 📚 完整示例

查看 `grid_search_example.py` 了解更多示例：

```bash
# 基础示例
python grid_search_example.py

# 完整示例
python grid_search_example.py --full
```

---

## ✅ 总结

### 优势
- ✅ **自动化**：无需手动创建多个配置
- ✅ **清晰**：明确区分固定参数和搜索参数
- ✅ **灵活**：支持多种搜索类型
- ✅ **可扩展**：易于添加新的搜索类型
- ✅ **向后兼容**：不影响现有单一配置

### 适用场景
- 🔍 超参数调优
- 📊 模型架构对比
- 🎯 寻找最佳配置
- 📈 消融实验（ablation study）

---

**开始使用Grid Search优化你的模型吧！** 🚀
