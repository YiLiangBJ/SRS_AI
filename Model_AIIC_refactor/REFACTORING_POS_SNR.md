# 🎉 配置重构完成 - pos_values 和 SNR 配置

## ✅ 完成状态

**所有重构已完成并测试通过！** 🚀

---

## 📊 主要改动

### 1. ✅ `pos_values` 移动到 model_configs

#### 改动前：
```yaml
# training_configs.yaml
default:
  pos_values: [0, 3, 6, 9]  # ❌ 在训练配置中
```

#### 改动后：
```yaml
# model_configs.yaml
common:
  seq_len: 12
  num_ports: 4
  pos_values: [0, 3, 6, 9]  # ✅ 在模型配置中

separator1_default:
  model_type: separator1
  # 继承 common 配置

separator1_6ports:
  model_type: separator1
  num_ports: 6
  pos_values: [0, 2, 4, 6, 8, 10]  # ✅ 6端口配置
```

**原因**：
- ✅ `pos_values` 决定数据结构（端口数量和位置）
- ✅ 与 `num_ports` 紧密绑定
- ✅ 是模型参数，不是训练策略

---

### 2. ✅ SNR 配置改进

#### 改动前（不明确）：
```yaml
# training_configs.yaml
default:
  snr_range: [0, 30]        # 是范围还是离散值？
  snr_per_sample: false
  snr_sampling: stratified
```

#### 改动后（明确区分）：
```yaml
# training_configs.yaml
default:
  snr_config:
    type: range             # ✅ 明确：连续范围
    min: 0
    max: 30
    per_sample: false
    sampling: stratified    # uniform, stratified, round_robin
    num_bins: 10

discrete_snr:
  snr_config:
    type: discrete          # ✅ 明确：离散值
    values: [0, 5, 10, 15, 20, 25, 30]
    per_sample: false
```

**优势**：
- ✅ **清晰**：`type: range` vs `type: discrete`
- ✅ **灵活**：支持两种采样方式
- ✅ **扩展性**：易于添加新类型（如 per-port SNR）

---

### 3. ✅ 分层命名方式

#### 改动前：
```
experiments_refactored/
  separator1_default/
    model.pth
```

#### 改动后（分层）：
```
experiments_refactored/
  {experiment_name}/               # 模型配置_训练配置
    {model_instance}/              # 具体模型参数组合
      model.pth
      config.yaml

# 例如：
experiments_refactored/
  separator1_small_quick_test/     # experiment: model+training
    separator1_small_hd32_stages2/ # model instance
      model.pth
      config.yaml
```

**优势**：
- ✅ 清晰的层次结构
- ✅ 便于对比同一模型的不同训练配置
- ✅ 便于对比不同模型在相同训练配置下的效果

---

## 🎯 核心设计原则

### 模型参数 vs 训练参数

| 参数 | 位置 | 判断依据 | 示例 |
|------|------|---------|------|
| **模型参数** | model_configs | 改变需要重建模型 | hidden_dim, num_stages, pos_values |
| **训练参数** | training_configs | 不需要重建模型 | batch_size, learning_rate, snr_config |

#### 模型参数（model_configs.yaml）：
```yaml
separator1_default:
  model_type: separator1
  # 架构参数
  hidden_dim: 64
  num_stages: 3
  mlp_depth: 3
  # 数据结构参数
  seq_len: 12
  num_ports: 4
  pos_values: [0, 3, 6, 9]  # ⭐ 决定数据维度
```

#### 训练参数（training_configs.yaml）：
```yaml
default:
  # 优化参数
  batch_size: 2048
  learning_rate: 0.01
  loss_type: weighted
  # 数据增强参数
  snr_config:              # ⭐ 不是超参数，是数据增强
    type: range
    min: 0
    max: 30
```

---

## 🚀 使用示例

### 示例1：4端口模型训练

```bash
# 4端口默认配置
python train.py \
  --model_config separator1_default \
  --training_config default

# pos_values 自动从 model_config 读取: [0, 3, 6, 9]
# SNR 自动从 training_config 读取: range [0, 30]
```

### 示例2：6端口模型训练

```bash
# 6端口配置
python train.py \
  --model_config separator1_6ports \
  --training_config default

# pos_values 自动读取: [0, 2, 4, 6, 8, 10]
```

### 示例3：离散SNR训练

```bash
# 使用离散SNR值
python train.py \
  --model_config separator1_default \
  --training_config discrete_snr

# SNR 只从这些值中选择: [0, 5, 10, 15, 20, 25, 30]
```

### 示例4：网格搜索（4端口）

```bash
# 网格搜索
python train.py \
  --model_config separator1_grid_search_small \
  --training_config grid_search_quick

# 所有配置都使用 4 端口
# pos_values 在 fixed_params 中: [0, 3, 6, 9]
```

---

## 📁 文件结构变化

### model_configs.yaml

```yaml
# 添加了 pos_values
common:
  seq_len: 12
  num_ports: 4
  pos_values: [0, 3, 6, 9]    # ⭐ 新增

# 新增 6端口配置
separator1_6ports:
  model_type: separator1
  num_ports: 6
  pos_values: [0, 2, 4, 6, 8, 10]  # ⭐ 新增
  
# 新增 6端口网格搜索
separator1_6ports_search:
  model_type: separator1
  num_ports: 6
  pos_values: [0, 2, 4, 6, 8, 10]  # ⭐ 新增
  search_space:
    hidden_dim: [32, 64]
    num_stages: [2, 3]
```

### training_configs.yaml

```yaml
# 移除了 pos_values，改进了 SNR 配置
default:
  snr_config:                 # ⭐ 新格式
    type: range
    min: 0
    max: 30
    per_sample: false
    sampling: stratified
    num_bins: 10
  # pos_values: [0, 3, 6, 9]  # ❌ 已移除

# 新增离散 SNR 配置
discrete_snr:                 # ⭐ 新增
  snr_config:
    type: discrete
    values: [0, 5, 10, 15, 20, 25, 30]
```

---

## 🔧 代码变化

### 新增文件

1. **`utils/snr_config.py`** - SNR 配置解析器
   ```python
   from utils import SNRConfig, parse_snr_config
   
   # Range-based
   config = {'type': 'range', 'min': 0, 'max': 30}
   snr = SNRConfig(config)
   snr_value = snr.sample()  # 采样
   
   # Discrete
   config = {'type': 'discrete', 'values': [0, 10, 20, 30]}
   snr = SNRConfig(config)
   snr_value = snr.sample()  # 从离散值中随机选择
   ```

### 修改文件

1. **`train.py`**
   - ✅ 从 model_config 读取 `pos_values`
   - ✅ 使用新的 SNRConfig
   - ✅ 实现分层命名

2. **`training/trainer.py`**
   - ✅ 接受 SNRConfig 对象
   - ✅ 向后兼容旧格式

3. **`utils/config_parser.py`**
   - ✅ 优化配置名称生成（只包含关键参数）

---

## 📊 测试结果

### 配置测试

```bash
$ python test_refactoring.py

Test 1: Model configs - pos_values
  Common: pos_values = [0, 3, 6, 9] ✅
  6-ports: pos_values = [0, 2, 4, 6, 8, 10] ✅

Test 2: Training configs - SNR
  Default: type=range, min=0, max=30 ✅
  Discrete: values=[0, 5, 10, 15, 20, 25, 30] ✅

Test 3: SNRConfig class
  Range sampling: working ✅
  Discrete sampling: working ✅

✓ All tests passed!
```

### 训练测试

```bash
$ python train.py --model_config separator1_small --training_config quick_test --num_batches 2

SNR Configuration: SNRConfig(range: [10, 20], sampling=uniform, per_sample=False) ✅
Configuration: separator1_small_hd32_stages2 ✅
Training completed ✅
Saved to: experiments_refactored/separator1_small_quick_test/separator1_small_hd32_stages2/ ✅
```

---

## 🎓 设计总结

### 问题解决

**你提出的问题**：
> "pos_values 应该放到 model_configs 里吧？需要明确知道哪些参数是只是设置，哪些参数是需要搜索多种可能"

**解决方案**：
```yaml
# model_configs.yaml - 明确：模型参数
fixed_params:
  pos_values: [0, 3, 6, 9]    # 固定的模型结构参数
  mlp_depth: 3
  
search_space:
  hidden_dim: [32, 64, 128]   # 搜索的超参数
  num_stages: [2, 3]
```

**你提出的问题**：
> "SNR 比较特殊，如果是'0,30'是范围，如果是'[0,30]'是离散值？"

**解决方案**：
```yaml
# training_configs.yaml - 明确区分
range_snr:
  snr_config:
    type: range               # ✅ 明确：连续范围
    min: 0
    max: 30

discrete_snr:
  snr_config:
    type: discrete            # ✅ 明确：离散值
    values: [0, 10, 20, 30]
```

### 核心价值

1. ✅ **清晰性**：模型参数 vs 训练参数完全分离
2. ✅ **明确性**：SNR range vs discrete 明确区分
3. ✅ **可维护性**：分层命名，结构清晰
4. ✅ **扩展性**：易于添加新类型（如 per-port SNR）

---

## 📚 文档

- **`REFACTORING_POS_SNR.md`** - 本文档
- **`GRID_SEARCH_GUIDE.md`** - 网格搜索指南
- **`test_refactoring.py`** - 测试脚本

---

## 🎉 重构完成！

**立即开始使用**：
```bash
cd Model_AIIC_refactor

# 4端口训练
python train.py --model_config separator1_default --training_config default

# 6端口训练
python train.py --model_config separator1_6ports --training_config default

# 离散SNR训练
python train.py --model_config separator1_default --training_config discrete_snr

# 网格搜索
python train.py --model_config separator1_grid_search_small --training_config grid_search_quick
```

**配置说明**：
- `pos_values` 在 **model_configs** 中 ✅
- SNR 配置在 **training_configs** 中 ✅
- 分层命名自动生成 ✅

**Happy Training!** 🚀
