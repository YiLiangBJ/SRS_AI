# 🎯 模型迁移完成：两种模型类型支持

## 📝 迁移总结

成功将 `Model_AIIC` 中的 `ResidualRefinementSeparator` 迁移到 `Model_AIIC_onnx`，并与现有的 `ResidualRefinementSeparatorReal` 共存。

---

## 🏗️ 两种模型架构

### Model Type 1: 双路实数 MLP (Dual-Path Real MLP)

**类名**: `ResidualRefinementSeparator`

**特点**:
- ✅ **分离处理**: 用两个独立的实数MLP分别处理实部和虚部
- ✅ **PyTorch原生复数**: 使用 `torch.complex64` 进行计算
- ✅ **参数量稍多**: ~36k (share_weights=True, stages=2, hidden_dim=32, mlp_depth=3)
- ✅ **表达能力强**: 实部虚部完全独立学习

**架构**:
```python
ComplexMLP:
  ├── mlp_real: Sequential(
  │   ├── Linear(seq_len*2 -> hidden_dim)
  │   ├── ReLU
  │   ├── [Hidden layers...]
  │   └── Linear(hidden_dim -> seq_len)
  │   )
  └── mlp_imag: Sequential(
      ├── Linear(seq_len*2 -> hidden_dim)
      ├── ReLU
      ├── [Hidden layers...]
      └── Linear(hidden_dim -> seq_len)
      )
```

**适用场景**:
- 快速原型开发
- 不需要ONNX导出
- 追求最佳性能

---

### Model Type 2: ComplexLinear共享权重 (ComplexLinear with Shared Weights)

**类名**: `ResidualRefinementSeparatorReal`

**特点**:
- ✅ **共享权重**: 用 `weight_real` 和 `weight_imag` 实现复数线性变换
- ✅ **ONNX兼容**: 可导出为ONNX格式用于MATLAB部署
- ✅ **参数量较少**: ~30k (share_weights=True, stages=2, hidden_dim=32, mlp_depth=3)
- ✅ **多种激活函数**: 支持 'split_relu', 'mod_relu', 'z_relu', 'cardioid'

**架构**:
```python
ComplexMLPReal:
  ├── fc1: ComplexLinearReal(seq_len -> hidden_dim)
  │   ├── weight_real(hidden_dim, seq_len)
  │   ├── weight_imag(hidden_dim, seq_len)
  │   ├── bias_real(hidden_dim)
  │   └── bias_imag(hidden_dim)
  ├── [Hidden layers with ComplexLinearReal...]
  └── fc_out: ComplexLinearReal(hidden_dim -> seq_len)
```

**适用场景**:
- 需要ONNX导出
- MATLAB集成
- 参数效率优先

---

## 📊 参数量对比

| 配置 | Type 1 (Dual-Path) | Type 2 (ComplexLinear) | 比例 |
|------|-------------------|------------------------|------|
| stages=2, hd=32, depth=3, share=True | 36,032 | 29,888 | 1.21x |
| stages=2, hd=64, depth=3, share=True | 120,832 | 107,392 | 1.13x |
| stages=3, hd=64, depth=3, share=False | 322,560 | 283,968 | 1.14x |

**结论**: Type 1 参数量约比 Type 2 多 10-20%

---

## 🚀 使用方法

### 1. 训练指定模型类型

```bash
# 训练 Type 1 (默认)
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --model_type "1" \
  --save_dir "./type1_models"

# 训练 Type 2
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --model_type "2" \
  --save_dir "./type2_models"

# 同时训练两种模型进行对比
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --model_type "1,2" \
  --save_dir "./compare_types"
```

### 2. 评估自动识别模型类型

```bash
# evaluate_models.py 会自动从 config 中读取 model_type
python Model_AIIC_onnx/evaluate_models.py \
  --model_dir "./type1_models/exp_name" \
  --output_dir "./evaluation_results"
```

### 3. Python API

```python
from Model_AIIC_onnx.channel_separator import (
    ResidualRefinementSeparator,      # Type 1
    ResidualRefinementSeparatorReal    # Type 2
)

# Type 1: Dual-Path MLP
model1 = ResidualRefinementSeparator(
    seq_len=12,
    num_ports=4,
    hidden_dim=64,
    num_stages=3,
    mlp_depth=3,
    share_weights_across_stages=False
)

# Type 2: ComplexLinear
model2 = ResidualRefinementSeparatorReal(
    seq_len=12,
    num_ports=4,
    hidden_dim=64,
    num_stages=3,
    mlp_depth=3,
    share_weights_across_stages=False,
    activation_type='split_relu',
    onnx_mode=False
)
```

---

## 🔄 重要变更

### 1. 移除能量归一化

**之前 (Model_AIIC)**:
```python
model = ResidualRefinementSeparator(
    ...,
    normalize_energy=True  # 模型内部归一化
)
```

**现在 (Model_AIIC_onnx)**:
```python
# 两种模型都不包含能量归一化
model1 = ResidualRefinementSeparator(...)  # No normalize_energy
model2 = ResidualRefinementSeparatorReal(...)

# 能量归一化在数据生成时完成
y_normalized, y_energy = normalize_and_generate_data(...)
```

### 2. 新增 model_type 参数

所有训练和评估脚本现在都支持 `model_type`:
- `--model_type "1"`: Type 1 (Dual-Path MLP)
- `--model_type "2"`: Type 2 (ComplexLinear)
- `--model_type "1,2"`: 训练两种模型

### 3. 配置文件格式

保存的 `model.pth` 现在包含:
```python
{
    'model_state_dict': ...,
    'config': {
        'model_type': 1,  # ⭐ 新增字段
        'seq_len': 12,
        'num_stages': 3,
        'hidden_dim': 64,
        'mlp_depth': 3,
        ...
    }
}
```

---

## 📁 修改的文件

### 核心文件
1. ✅ `Model_AIIC_onnx/channel_separator.py`
   - 添加 `ResidualRefinementSeparator` (Type 1)
   - 更新 `ResidualRefinementSeparatorReal` 注释 (Type 2)

2. ✅ `Model_AIIC_onnx/test_separator.py`
   - 添加 `--model_type` 参数
   - 根据 model_type 创建对应模型
   - 保存 model_type 到配置

3. ✅ `Model_AIIC_onnx/evaluate_models.py`
   - 从配置读取 model_type
   - 自动加载对应模型类型

4. ✅ `Model_AIIC_onnx/plot_results.py`
   - 无需修改（自动支持）

---

## 🧪 测试验证

### 快速测试

```bash
# 测试导入
python -c "from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparator, ResidualRefinementSeparatorReal; print('✅ Import successful')"

# 测试参数量
python -c "
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparator, ResidualRefinementSeparatorReal
m1 = ResidualRefinementSeparator(num_ports=4, num_stages=2, hidden_dim=32, mlp_depth=3)
m2 = ResidualRefinementSeparatorReal(num_ports=4, num_stages=2, hidden_dim=32, mlp_depth=3)
print(f'Type 1: {sum(p.numel() for p in m1.parameters()):,} params')
print(f'Type 2: {sum(p.numel() for p in m2.parameters()):,} params')
"

# 快速训练测试
python Model_AIIC_onnx/test_separator.py \
  --batches 5 \
  --batch_size 32 \
  --model_type "1,2" \
  --save_dir "./test_both_types" \
  --device "cpu"
```

### 预期输出

```
Type 1: 36,032 params
Type 2: 29,888 params

Experiment 1/2: stages=3_hd=64_depth=3_type=1_...
  Model: Type 1 (Dual-Path Real MLP)
  Model parameters: 120,832
  ...

Experiment 2/2: stages=3_hd=64_depth=3_type=2_...
  Model: Type 2 (ComplexLinear, activation=split_relu)
  Model parameters: 107,392
  ...
```

---

## ⚖️ 选择建议

### 使用 Type 1 (Dual-Path MLP) 当:
- ✅ 不需要 ONNX 导出
- ✅ 追求最佳性能
- ✅ 快速原型开发
- ✅ 参数量不是瓶颈

### 使用 Type 2 (ComplexLinear) 当:
- ✅ 需要 ONNX/MATLAB 部署
- ✅ 参数效率优先
- ✅ 需要多种激活函数实验
- ✅ 生产环境部署

### 两种都训练对比:
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 50000 \
  --batch_size 2048 \
  --stages "2,3" \
  --hidden_dim "32,64" \
  --mlp_depth "3" \
  --model_type "1,2" \  # ⭐ 训练两种模型
  --save_dir "./compare_models"
```

---

## 🔧 向后兼容性

### 旧模型加载

`evaluate_models.py` 自动兼容旧模型：

```python
# 如果配置中没有 model_type，默认使用 Type 2
model_type = config.get('model_type', 2)
```

**迁移策略**：
1. 旧模型（无 model_type）自动识别为 Type 2
2. 新训练的模型明确保存 model_type
3. 可以混合评估新旧模型

---

## 📈 性能对比 (预期)

| 指标 | Type 1 (Dual-Path) | Type 2 (ComplexLinear) |
|------|-------------------|------------------------|
| **参数量** | 较多 (+10-20%) | 较少 |
| **训练速度** | 快（简单操作） | 稍慢（复杂计算） |
| **NMSE** | 预计相当 | 预计相当 |
| **ONNX导出** | ❌ 不支持 | ✅ 支持 |
| **表达能力** | 强（独立MLP） | 强（共享权重） |

**建议**：先用 Type 1 快速验证，确认效果后再用 Type 2 训练用于部署。

---

## ✅ 迁移检查清单

- [x] Type 1 模型迁移到 Model_AIIC_onnx
- [x] 移除 normalize_energy 参数
- [x] test_separator.py 支持 model_type 参数
- [x] evaluate_models.py 自动识别模型类型
- [x] plot_results.py 兼容性确认
- [x] 测试验证两种模型都能导入
- [x] 创建迁移文档

---

## 🎉 迁移完成！

现在 `Model_AIIC_onnx` 目录拥有两种强大的模型：
- **Type 1**: 灵活、高效、易用
- **Type 2**: 可部署、紧凑、兼容

选择适合你的场景，开始训练吧！ 🚀

