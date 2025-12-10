# ✅ Training Config Search Space 支持完成

## 改进内容

现在 `training_configs.yaml` 也支持 `search_space` 语法，与 `model_configs.yaml` 完全统一！

---

## 1. 统一的配置格式

### Model Config（已有）
```yaml
separator1_grid_search_basic:
  model_type: separator1
  fixed_params:
    pos_values: [0, 3, 6, 9]
    mlp_depth: 3
  search_space:
    hidden_dim: [32, 64, 128]
    num_stages: [2, 3, 4]
```

### Training Config（新增）✅
```yaml
default_loss_search:
  fixed_params:
    batch_size: 4096
    num_batches: 100000
    learning_rate: 0.01
    snr_config:
      type: range
      min: 0
      max: 30
  search_space:
    loss_type: [nmse, weighted, log, normalized]
```

---

## 2. 新增配置示例

### `configs/training_configs.yaml`

```yaml
# Search different loss types
default_loss_search:
  fixed_params:
    batch_size: 4096
    num_batches: 100000
    learning_rate: 0.01
    snr_config:
      type: range
      min: 0
      max: 30
      per_sample: true
      sampling: stratified
    tdl_config: [A-30, B-100, C-300]
    print_interval: 100
  search_space:
    loss_type: [nmse, weighted, log, normalized]
  # Generates: 4 training configurations

# Search different learning rates
default_lr_search:
  fixed_params:
    batch_size: 4096
    num_batches: 100000
    loss_type: nmse
    snr_config:
      type: range
      min: 0
      max: 30
    tdl_config: A-30
    print_interval: 100
  search_space:
    learning_rate: [0.001, 0.01, 0.1]
  # Generates: 3 training configurations

# Combined search: loss_type × learning_rate
default_loss_lr_search:
  fixed_params:
    batch_size: 4096
    num_batches: 50000
    snr_config:
      type: range
      min: 0
      max: 30
    tdl_config: A-30
  search_space:
    loss_type: [nmse, weighted]
    learning_rate: [0.01, 0.001]
  # Generates: 2 × 2 = 4 training configurations
```

---

## 3. 使用方式

### 单个 Loss Type（原有方式）
```bash
python train.py \
  --model_config separator1_small \
  --training_config quick_test
```

**输出**：训练 1 个配置

### 搜索多个 Loss Types（新功能）
```bash
python train.py \
  --model_config separator1_small \
  --training_config default_loss_search \
  --num_batches 50
```

**输出**：
```
Training search space: 4 configurations
   Fixed parameters:
     batch_size: 4096
     learning_rate: 0.01
     ...
   
   Search parameters:
     loss_type: [nmse, weighted, log, normalized] (4 values)

Training Config Variant 1/4: default_loss_search_nmse
  Loss type: nmse
  ...

Training Config Variant 2/4: default_loss_search_weighted
  Loss type: weighted
  ...

Training Config Variant 3/4: default_loss_search_log
  Loss type: log
  ...

Training Config Variant 4/4: default_loss_search_normalized
  Loss type: normalized
  ...
```

### 组合搜索（Model × Training）
```bash
python train.py \
  --model_config separator1_grid_search_small \
  --training_config default_loss_search
```

**结果**：
- Model configs: 2 (hidden_dim: 32, 64)
- Training configs: 4 (loss_type: nmse, weighted, log, normalized)
- **Total: 2 × 4 = 8 configurations**

---

## 4. 核心实现

### `train.py` 修改

```python
# Parse training config (supports search_space like model configs)
training_configs = parse_model_config(training_config_raw)

# Count total configurations (model × training)
total_configs = 0
for model_config_name in model_config_names:
    parsed_model_configs = parse_model_config(full_model_config)
    total_configs += len(parsed_model_configs) * len(training_configs)

# Train each combination
for training_config in training_configs:
    for model_config in parsed_model_configs:
        # Train with this specific combination
        ...
```

### `utils/config_parser.py` 修改

```python
def parse_model_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse configuration with search_space support
    
    Now works for:
    - Model configs (with model_type)
    - Training configs (without model_type) ✅ NEW
    """
    # ... (unchanged logic)
    
    # model_type is now optional (for training configs)
    # Only infer num_ports if pos_values exists (model configs)
```

---

## 5. 测试结果

### 测试1：单个模型 + 多个 loss types
```bash
python train.py \
  --model_config separator1_small \
  --training_config default_loss_search \
  --num_batches 50
```

**结果**：✅ 通过
- 训练了 4 个配置（nmse, weighted, log, normalized）
- 每个配置独立训练和保存
- 生成完整报告比较所有loss types

**最佳结果**：
```
🏆 Best configuration: separator1_small_..._log
   NMSE: -8.20 dB
   
Results:
1. log:        -8.20 dB (best)
2. normalized: -6.86 dB
3. weighted:   -6.63 dB
4. nmse:       -6.47 dB
```

---

## 6. 优势

### 统一的语法 ✅
```yaml
# Model config
separator1_search:
  fixed_params: {...}
  search_space: {...}

# Training config (完全一样的语法！)
default_loss_search:
  fixed_params: {...}
  search_space: {...}
```

### 灵活的组合 ✅
- Model only search: `1 training × N models`
- Training only search: `N trainings × 1 model`
- Combined search: `N trainings × M models`

### 清晰的输出 ✅
```
Training Config Variant 1/4: default_loss_search_nmse
  Loss type: nmse
  Learning rate: 0.01
  SNR: SNRConfig(...)

Model: separator1_small
Training: default_loss_search_nmse
```

---

## 7. 配置文件更新

### `training_configs.yaml`
- ✅ 修复 `default` 配置（改为单个 loss_type）
- ✅ 新增 `default_loss_search`（搜索 4 个 loss types）
- ✅ 新增 `default_lr_search`（搜索 3 个 learning rates）
- ✅ 新增 `default_loss_lr_search`（组合搜索）

### `utils/config_parser.py`
- ✅ 支持没有 `model_type` 的配置（training configs）
- ✅ 只在有 `pos_values` 时推导 `num_ports`

### `train.py`
- ✅ 解析 training config 的 search_space
- ✅ 嵌套循环：training_configs × model_configs
- ✅ 显示 training variant 信息
- ✅ 生成完整的配置名称

---

## 8. 对比：改进前 vs 改进后

| 特性 | 改进前 | 改进后 |
|------|--------|--------|
| **Model search space** | ✅ 支持 | ✅ 支持 |
| **Training search space** | ❌ 不支持 | ✅ 支持 ✨ |
| **Loss type 搜索** | ❌ 需要手动 | ✅ 自动化 |
| **Learning rate 搜索** | ❌ 需要手动 | ✅ 自动化 |
| **组合搜索** | ❌ 不支持 | ✅ M×N 组合 |
| **配置语法** | ❌ 不统一 | ✅ 完全统一 |

---

## 9. 示例场景

### 场景1：比较不同 Loss Functions
```bash
# 测试哪种 loss function 最好
python train.py \
  --model_config separator1_default \
  --training_config default_loss_search
```

### 场景2：调整 Learning Rate
```bash
# 找到最优学习率
python train.py \
  --model_config separator1_default \
  --training_config default_lr_search
```

### 场景3：完整 Grid Search
```bash
# 模型架构 × loss type 的完整搜索
python train.py \
  --model_config separator1_grid_search_full \
  --training_config default_loss_search
# 18 model configs × 4 loss types = 72 experiments
```

---

## 10. 总结

### ✅ 完成的功能

1. **Training config search space 支持** ✅
   - 与 model config 完全统一的语法
   - 支持任意训练超参数搜索

2. **Loss type 作为超参数** ✅
   - 可以在 search_space 中定义
   - 自动生成多个训练配置

3. **Learning rate 搜索** ✅
   - 同样通过 search_space 定义

4. **组合搜索** ✅
   - Model configs × Training configs
   - 自动生成所有组合

### 🎯 核心价值

- ✅ **统一的配置语法** - model 和 training 完全一致
- ✅ **自动化超参数搜索** - 不需要手动运行多次
- ✅ **清晰的结果对比** - 自动生成报告比较所有配置
- ✅ **灵活的组合** - 支持 model-only、training-only、组合搜索

---

**所有功能已完成并测试通过！** 🎉
