# Model_AIIC_refactor - Refactored Channel Separator# Model_AIIC_onnx - ONNX 兼容的通道分离器



🎯 **Clean, modular, extensible architecture for channel separator models**完全基于实数张量的 SRS 通道分离器，支持 ONNX 导出和 MATLAB 部署。



## 📁 Directory Structure## 🎯 核心特性



```- ✅ **双模式支持** - `onnx_mode` 参数切换训练模式和 ONNX 兼容模式

Model_AIIC_refactor/- ✅ **ONNX Opset 9** - 支持导出为 Opset 9 格式

├── models/              # Model definitions- ✅ **MATLAB 部署** - 使用 `importONNXFunction` 导入并推理

│   ├── base_model.py        # Base class- ✅ **数学等价** - 使用实数块矩阵，完全等价于复数运算

│   ├── separator1.py        # Dual-Path MLP- ✅ **多种激活函数** - 支持 split_relu, mod_relu, z_relu, cardioid

│   ├── separator2.py        # ComplexLinear- ✅ **参数高效** - 与原复数版本参数量相同

│   └── __init__.py          # Model registry

│## 📋 目录

├── data/                # Data generation

│   ├── data_generator.py    # TDL channel generation- [快速开始](#-快速开始)

│   └── __init__.py- [训练模型](#-训练模型)

│- [验证等价性](#-验证等价性)

├── training/            # Training infrastructure- [导出 ONNX](#-导出-onnx)

│   ├── trainer.py           # Unified trainer- [MATLAB 部署](#-matlab-部署)

│   ├── loss_functions.py    # Loss functions- [性能评估](#-性能评估)

│   ├── metrics.py           # Evaluation metrics- [onnx_mode 说明](#-onnx_mode-说明)

│   └── __init__.py- [常见问题](#-常见问题)

│- [文件结构](#-文件结构)

├── utils/               # Utilities

│   ├── device_utils.py      # GPU/CPU management---

│   ├── snr_sampler.py       # SNR sampling strategies

│   ├── logging_utils.py     # Logging tools## 🚀 快速开始

│   └── __init__.py

│### 1. 训练一个简单模型

├── configs/             # YAML configurations

│   ├── model_configs.yaml```bash

│   └── training_configs.yamlcd c:/GitRepo/SRS_AI

│

├── tests/               # Unit tests# 快速测试（4 端口，50 批次，训练模式）

│python Model_AIIC_onnx/test_separator.py \

├── train.py             # Main training script  --batches 50 \

└── test_refactored.py   # Quick verification test  --batch_size 128 \

```  --stages "2" \

  --ports "0,3,6,9" \

---  --save_dir "./Model_AIIC_onnx/quick_test"

```

## 🚀 Quick Start

### 2. 导出为 ONNX

### 1. Verify Installation

```bash

```bash# 导出模型（自动切换到 onnx_mode）

cd Model_AIIC_refactorpython Model_AIIC_onnx/export_onnx.py \

python test_refactored.py  --checkpoint ./Model_AIIC_onnx/quick_test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \

```  --output my_model.onnx \

  --opset 9

Should output: `✓ All tests passed!````



### 2. Train a Model### 3. 在 MATLAB 中测试



```bash```matlab

# Train Separator1 with default configcd('c:/GitRepo/SRS_AI')

python train.py --model_config separator1_default --training_config defaulttest_onnx_function('my_model.onnx')

```

# Quick test (2 batches)

python train.py --model_config separator1_small --training_config quick_test---



# Train both models for comparison## 🎓 训练模型

python train.py --model_config separator1_default,separator2_default --training_config default

```### 基本训练命令



### 3. Custom Training```bash

python Model_AIIC_onnx/test_separator.py \

```bash  --batches 1000 \

# Override parameters  --batch_size 2048 \

python train.py \  --stages "2,3" \

  --model_config separator2_default \  --save_dir "./trained_models"

  --training_config default \```

  --batch_size 4096 \

  --num_batches 20000 \### 完整参数说明

  --device cuda

```| 参数 | 说明 | 默认值 | 示例 |

|------|------|--------|------|

---| `--batches` | 训练批次数 | 100 | `1000` |

| `--batch_size` | 批次大小 | 32 | `2048` |

## 💡 Key Features| `--stages` | 优化阶段数 | "3" | `"2,3,4"` |

| `--share_weights` | 是否共享权重 | "False" | `"True,False"` |

### ✅ Unified Interface| `--ports` | 端口位置 | "0,3,6,9" | `"0,2,4,6,8,10"` |

All models inherit from `BaseSeparatorModel`:| `--snr` | 信噪比 (dB) | "20" | `"0,30"` 或 `"[10,20,30]"` |

```python| `--tdl` | TDL 信道配置 | "A-30" | `"A-30,B-100,C-300"` |

from models import create_model| `--loss_type` | 损失函数类型 | "nmse" | `"normalized"` |

| `--activation_type` | 激活函数 | "split_relu" | `"split_relu,cardioid"` |

config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 64}| `--onnx_mode` | ONNX 兼容模式 | False | 添加此标志启用 |

model = create_model('separator1', config)| `--save_dir` | 保存目录 | None | `"./models"` |

```

### 训练模式选择

### ✅ Factory Pattern

Easy model creation:#### 模式 1: 训练模式（推荐，最快）

```python

# List available models```bash

from models import list_models# 默认模式，训练速度最快（100% 性能）

print(list_models())  # ['separator1', 'separator2', 'dual_path', 'complex_linear', ...]python Model_AIIC_onnx/test_separator.py \

  --batches 1000 \

# Create any model  --batch_size 2048 \

model = create_model('separator1', config)  --stages "2" \

model = create_model('separator2', config)  --save_dir "./models_training"

``````



### ✅ Configuration-Driven#### 模式 2: ONNX 兼容模式（验证用）

Edit `configs/model_configs.yaml`:

```yaml```bash

models:# 使用 ONNX Opset 9 兼容操作（~80-85% 性能）

  my_custom_model:# 用于验证 ONNX 模式的训练等价性

    model_type: separator1python Model_AIIC_onnx/test_separator.py \

    hidden_dim: 128  --batches 1000 \

    num_stages: 4  --batch_size 2048 \

```  --stages "2" \

  --onnx_mode \

Then use:  --save_dir "./models_onnx"

```bash```

python train.py --model_config my_custom_model

```### 多配置搜索



### ✅ Modular Training```bash

```python# 搜索最佳超参数组合

from models import create_modelpython Model_AIIC_onnx/test_separator.py \

from training import Trainer  --batches 1000 \

  --batch_size 2048 \

model = create_model('separator1', config)  --stages "2,3,4" \

trainer = Trainer(model, learning_rate=0.01, loss_type='weighted')  --share_weights "True,False" \

losses = trainer.train(num_batches=10000, batch_size=2048, snr_db=(0, 30))  --activation_type "split_relu,cardioid" \

```  --loss_type "nmse,normalized" \

  --save_dir "./hyperparameter_search"

---```



## 📊 Model Types这会训练 2×2×2×2 = 16 个不同配置。



### Separator1 - Dual-Path MLP### 训练输出

- Two independent real MLPs (one for real, one for imaginary)

- Simple architecture训练完成后会生成：

- Slightly more parameters (~20% more than Separator2)

- Fast training```

models_training/

**Parameters**: ~120k (stages=3, hidden_dim=64, share_weights=False)├── stages=2_share=False_loss=nmse_act=split_relu/

│   ├── model.pth              # PyTorch 模型（包含权重和配置）

### Separator2 - ComplexLinear│   ├── training_curves.png    # 训练曲线

- Shared weight matrices (weight_real, weight_imag)│   ├── final_predictions.png  # 最终预测结果

- ONNX exportable│   ├── energy_distribution.png # 能量分布

- Multiple activation functions│   └── metrics.json           # 训练指标

- ~20% fewer parameters└── search_summary.json        # 多配置搜索总结

```

**Parameters**: ~107k (stages=3, hidden_dim=64, share_weights=False)

---

---

## 🔬 验证等价性

## 🔧 Adding a New Model

验证 `onnx_mode=True` 和 `onnx_mode=False` 的数学等价性：

1. Create `models/my_model.py`:

```python```bash

from .base_model import BaseSeparatorModel# 运行等价性测试

python Model_AIIC_onnx/verify_onnx_mode_equivalence.py

class MyModel(BaseSeparatorModel):```

    def __init__(self, seq_len, num_ports, **kwargs):

        super().__init__(seq_len, num_ports)### 期望输出

        # Your network here

    ```

    def forward(self, y):================================================================================

        # Your forward passONNX Mode vs Training Mode - Equivalence Test

        return h================================================================================

    

    @classmethodTest with batch_size=1:

    def from_config(cls, config):  Max absolute diff:  0.00e+00

        return cls(**config)  Mean absolute diff: 0.00e+00

```  ✓ PASSED (diff < 1e-6)



2. Register in `models/__init__.py`:Test with batch_size=2:

```python  Max absolute diff:  0.00e+00

from .my_model import MyModel  ✓ PASSED (diff < 1e-6)

register_model('my_model', MyModel)

```Gradient Computation Test:

  Max gradient difference: 5.96e-08

3. Add config in `configs/model_configs.yaml`:  ✓ Gradients are equivalent (diff < 1e-6)

```yaml

my_model_default:✓ All tests passed!

  model_type: my_model```

  your_param: value

```**结论**：两种模式完全等价，可以：

1. 用训练模式训练（更快）

4. Train:2. 导出时切换到 ONNX 模式

```bash3. 或直接用 ONNX 模式训练来验证

python train.py --model_config my_model_default

```---



---## 📤 导出 ONNX



## 📈 Training Configurations### 基本导出



Available in `configs/training_configs.yaml`:```bash

python Model_AIIC_onnx/export_onnx.py \

- `default`: Standard training (10k batches, SNR 0-30 dB)  --checkpoint ./models_training/stages=2_share=False_loss=nmse_act=split_relu/model.pth \

- `quick_test`: Fast test (100 batches)  --output my_model.onnx \

- `high_snr`: High SNR regime (20-40 dB)  --opset 9

- `low_snr`: Low SNR regime (-10 to 10 dB)```

- `multi_tdl`: Multiple TDL channel models

- `six_ports`: 6-port configuration### 导出过程



---1. **自动切换模式** - 导出时自动设置 `onnx_mode=True`

2. **生成 ONNX 文件** - Opset 9 格式

## 🧪 Testing3. **显示使用说明** - MATLAB 代码示例



```bash### 导出输出

# Quick verification

python test_refactored.py```

================================================================================

# Run unit testsExporting Model to ONNX Format

pytest tests/================================================================================

```Checkpoint: ./models_training/.../model.pth

Output:     my_model.onnx

---Opset:      9



## 📦 Comparison with OriginalModel Configuration:

  Sequence length: 12

| Aspect | Original (Model_AIIC_onnx) | Refactored |  Num ports:       4

|--------|---------------------------|------------|  Num stages:      2

| **test_separator.py** | 1700+ lines | ~150 lines (train.py) |  ⭐ ONNX mode set to: True (for MATLAB compatible export)

| **Model definition** | Mixed in one file | Separate files |

| **Data generation** | Embedded in test | Separate module |✓ ONNX export successful!

| **Training logic** | Embedded | Unified Trainer |  File size: 0.41 MB

| **Adding new model** | Modify multiple files | One file + register |

| **Configuration** | Command-line args | YAML configs |MATLAB Usage:

| **Maintainability** | Medium | High |────────────────────────────────────────────────────────────────────────────────

% 1. 导入模型

---params = importONNXFunction('my_model.onnx', 'model_func');



## 🎯 Design Principles% 2. 准备数据（必须归一化）

y = randn(1, 12) + 1i*randn(1, 12);

1. **Separation of Concerns**: Models, data, training are separatey_stacked = [real(y), imag(y)];

2. **DRY (Don't Repeat Yourself)**: Common code in shared modulesy_energy = sqrt(mean(abs(y).^2));

3. **Extensibility**: Easy to add new modelsy_normalized = y_stacked / y_energy;

4. **Configuration-Driven**: Less hardcoded values

5. **Testability**: Each module can be tested independently% 3. 推理

[h_normalized, ~] = model_func(y_normalized, params, ...

---                               'InputDataPermutation', 'none', ...

                               'OutputDataPermutation', 'none');

## ✅ Advantages

% 4. 恢复能量

- ✅ **Cleaner code**: Each file has single responsibilityh_stacked = h_normalized * y_energy;

- ✅ **Easy to extend**: Add new models without touching existing code────────────────────────────────────────────────────────────────────────────────

- ✅ **Better testing**: Modular structure enables unit testing```

- ✅ **Configuration management**: YAML instead of long command lines

- ✅ **Reusable**: Training logic works for all models### 检查导出的 ONNX 算子

- ✅ **Maintainable**: Changes localized to relevant modules

```bash

---# 查看模型中的算子

python -c "

## 🔄 Migration Guideimport onnx

model = onnx.load('my_model.onnx')

### From Model_AIIC_onnx to Model_AIIC_refactorops = {}

for n in model.graph.node:

**Old way**:    ops[n.op_type] = ops.get(n.op_type, 0) + 1

```bashprint('ONNX Operators:')

python test_separator.py \for op, count in sorted(ops.items()):

  --batches 10000 \    print(f'  {op:20s}: {count:3d}')

  --batch_size 2048 \"

  --model_type 1 \```

  --stages 3 \

  --hidden_dim 64 \### 期望的算子（onnx_mode=True）

  --share_weights False \

  --loss_type weighted```

```ONNX Operators:

  Add                 :  92

**New way**:  Concat              :  47

```bash  Constant            :  32

# Create config once  Gather              :  32   ⚠️ MATLAB importONNXNetwork 不支持

# configs/model_configs.yaml:  Identity            :  48

#   my_config:  MatMul              :  96

#     model_type: separator1  Relu                :  32

#     hidden_dim: 64  Slice               :  84   ⚠️ MATLAB importONNXNetwork 不支持

#     num_stages: 3  Sub                 :  26

  Unsqueeze           :  17   ⚠️ MATLAB importONNXNetwork 不支持

python train.py --model_config my_config --training_config default```

```

**注意**：这些算子 `importONNXNetwork` 不支持，但 **`importONNXFunction` 支持**。

---

---

## 📚 Next Steps

## 🎯 MATLAB 部署

1. ✅ Add unit tests in `tests/`

2. ✅ Add more model variants### 方法 1: 使用测试脚本（推荐）

3. ✅ Add model comparison utilities

4. ✅ Add ONNX export functionality```matlab

5. ✅ Add documentation generation (Sphinx)cd('c:/GitRepo/SRS_AI')



---% 使用默认模型

test_onnx_function

## 🤝 Contributing

% 或指定模型

To add a new feature:test_onnx_function('my_model.onnx')

1. Create module in appropriate directory```

2. Add to `__init__.py`

3. Add tests in `tests/`### 方法 2: 手动部署

4. Update this README

```matlab

---%% 步骤 1: 导入模型

cd('c:/GitRepo/SRS_AI')

**Happy Training!** 🚀params = importONNXFunction('my_model.onnx', 'model_func');

% ✓ 生成 model_func.m
% ✓ params 是 ONNXParameters 对象

%% 步骤 2: 准备测试数据
L = 12;
y_complex = randn(1, L) + 1i*randn(1, L);
y_stacked = [real(y_complex), imag(y_complex)];  % (1, 24)

%% 步骤 3: 能量归一化（必须！）
y_energy = sqrt(mean(abs(y_complex).^2));
y_normalized = y_stacked / y_energy;

fprintf('Input energy: %.6f\n', y_energy);
fprintf('Normalized energy: %.6f\n', sqrt(mean(y_normalized.^2)));

%% 步骤 4: 推理
tic;
[h_normalized, ~] = model_func(y_normalized, params, ...
                               'InputDataPermutation', 'none', ...
                               'OutputDataPermutation', 'none');
inference_time = toc;

fprintf('Inference time: %.2f ms\n', inference_time * 1000);
fprintf('Output shape: %s\n', mat2str(size(h_normalized)));

%% 步骤 5: 恢复能量
h_stacked = h_normalized * y_energy;

%% 步骤 6: 转换为复数
P = size(h_stacked, 2);  % 端口数
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);
h_complex = complex(h_real, h_imag);

fprintf('Number of ports: %d\n', P);
fprintf('Complex shape: %s\n', mat2str(size(h_complex)));

%% 步骤 7: 验证重建
y_recon = squeeze(sum(h_complex, 2));
recon_error = norm(y_complex - y_recon) / norm(y_complex);

fprintf('Reconstruction error: %.2f%%\n', recon_error * 100);

if recon_error < 0.05
    fprintf('✓ GOOD reconstruction quality!\n');
else
    fprintf('⚠️  Check if model is trained properly\n');
end

%% 步骤 8: 能量分布
fprintf('\nEnergy per port:\n');
for p = 1:P
    port_data = squeeze(h_complex(:, p, :));
    port_energy = sqrt(mean(abs(port_data).^2));
    port_ratio = (port_energy^2) / (y_energy^2) * 100;
    fprintf('  Port %d: %.6f (%.1f%% of input)\n', p, port_energy, port_ratio);
end
```

### MATLAB 要求

- **MATLAB 版本**: R2020b 或更新
- **工具箱**: Deep Learning Toolbox
- **导入方法**: `importONNXFunction`（不能用 `importONNXNetwork`）

### 为什么不能用 importONNXNetwork？

当前模型包含以下算子，`importONNXNetwork` 不支持：

```
✗ 不支持的算子：
  - Slice (84个)        - 来自索引操作
  - Gather (32个)       - 来自索引操作
  - Unsqueeze (17个)    - 来自 .unsqueeze()
  - MatMul (48个)       - 需要一个输入是常量
  - Sub (26个)          - 需要一个输入是常量
```

**解决方案**: 使用 `importONNXFunction`，它支持更多算子。

---

## 📊 性能评估

### 评估训练好的模型

```bash
# 使用 test_separator.py 评估（包含可视化）
python Model_AIIC_onnx/test_separator.py \
  --batches 100 \
  --batch_size 128 \
  --stages "2" \
  --save_dir "./evaluation"
```

### 生成的图表

训练/评估会自动生成以下图表：

1. **training_curves.png** - 训练和验证损失曲线
2. **final_predictions.png** - 最终预测结果对比
3. **energy_distribution.png** - 各端口能量分布
4. **reconstruction_quality.png** - 重建质量分析
5. **snr_performance.png** - 不同 SNR 下的性能（如果启用）

### 手动评估脚本

```python
import torch
import numpy as np
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

# 加载模型
checkpoint = torch.load('model.pth')
config = checkpoint['config']

model = ResidualRefinementSeparatorReal(
    seq_len=config['seq_len'],
    num_ports=len(checkpoint['hyperparameters']['pos_values']),
    hidden_dim=config['hidden_dim'],
    num_stages=config['num_stages'],
    share_weights_across_stages=config['share_weights'],
    activation_type=config.get('activation_type', 'split_relu'),
    onnx_mode=False  # 评估时使用训练模式
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 生成测试数据
# ... (见 test_separator.py)

# 评估
with torch.no_grad():
    output = model(input_normalized)
    
# 计算指标
nmse = ((output - target)**2).mean() / (target**2).mean()
print(f'NMSE: {nmse.item():.6f} ({10*np.log10(nmse.item()):.2f} dB)')
```

---

## 🔧 onnx_mode 说明

### 什么是 onnx_mode？

`onnx_mode` 是一个布尔参数，控制模型使用哪种实现：

| 模式 | onnx_mode | 特点 | 用途 |
|------|-----------|------|------|
| **训练模式** | False | 使用高效操作（unsqueeze, repeat, sum） | 日常训练 |
| **ONNX 模式** | True | 使用 Opset 9 兼容操作（显式循环，concat） | ONNX 导出 |

### 两种模式的区别

#### 训练模式 (onnx_mode=False)

```python
# 特征初始化 - 使用 unsqueeze + repeat
features = y.unsqueeze(1).repeat(1, P, 1)

# 残差计算 - 使用 sum
y_recon = features.sum(dim=1)

# 残差添加 - 使用 broadcasting
features = features + residual.unsqueeze(1)
```

**优点**: 训练速度快（100% 性能）  
**缺点**: 生成的 ONNX 包含不兼容算子（Unsqueeze, Tile, ReduceSum）

#### ONNX 模式 (onnx_mode=True)

```python
# 特征初始化 - 使用显式循环 + concat
features_list = [y.unsqueeze(1) for _ in range(P)]
features = torch.cat(features_list, dim=1)

# 残差计算 - 使用显式循环
y_recon = features[:, 0, :].clone()
for p in range(1, P):
    y_recon = y_recon + features[:, p, :]

# 残差添加 - 使用显式循环
features_list = []
for p in range(P):
    features_list.append((features[:, p, :] + residual).unsqueeze(1))
features = torch.cat(features_list, dim=1)
```

**优点**: ONNX 兼容（避免大部分不支持算子）  
**缺点**: 训练稍慢（~80-85% 性能）

### 数学等价性

两种模式**完全等价**：
- ✅ 前向传播差异: **0.00e+00**
- ✅ 梯度计算差异: **< 1e-7**
- ✅ 训练结果相同

### 使用建议

#### 推荐工作流程 ⭐

```bash
# 1. 训练时使用训练模式（更快）
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --stages "2" \
  --save_dir "./models"

# 2. 导出时自动切换到 ONNX 模式
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint ./models/.../model.pth \
  --output model.onnx \
  --opset 9
# ✓ export_onnx.py 会自动设置 onnx_mode=True
```

#### 验证工作流程

```bash
# 如果想验证 ONNX 模式训练是否工作
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --stages "2" \
  --onnx_mode \
  --save_dir "./models_onnx"

# 对比两种模式的结果（应该相同）
python Model_AIIC_onnx/verify_onnx_mode_equivalence.py
```

### 切换 onnx_mode

#### 在训练时

```python
# 创建模型时指定
model = ResidualRefinementSeparatorReal(
    seq_len=12,
    num_ports=4,
    onnx_mode=True  # ⭐ ONNX 模式
)
```

#### 在导出时

```python
# 加载训练好的模型
model = ResidualRefinementSeparatorReal(..., onnx_mode=False)
model.load_state_dict(checkpoint['model_state_dict'])

# 切换到 ONNX 模式
model.onnx_mode = True  # ⭐ 动态切换

# 导出
torch.onnx.export(model, ...)
```

---

## ❓ 常见问题

### Q1: MATLAB 导入失败：算子不支持

**错误信息**:
```
Operator 'Slice' is not supported.
Operator 'Gather' is not supported.
Operator 'Unsqueeze' is not supported.
```

**原因**: 使用了 `importONNXNetwork`

**解决方案**: 使用 `importONNXFunction`

```matlab
% ❌ 不要用这个
net = importONNXNetwork('model.onnx');

% ✅ 用这个
params = importONNXFunction('model.onnx', 'model_func');
[output, ~] = model_func(input, params, 'InputDataPermutation', 'none', ...
                                         'OutputDataPermutation', 'none');
```

---

### Q2: 重建误差很大（> 50%）

**可能原因**:

1. **模型未训练** - 检查 checkpoint 的损失值
2. **忘记能量归一化** - 必须在推理前归一化
3. **测试数据不匹配** - 需要与训练数据相似的分布

**检查步骤**:

```python
# 检查模型损失
checkpoint = torch.load('model.pth')
print(f"Final loss: {checkpoint.get('final_train_loss', 'N/A')}")
print(f"Test NMSE: {checkpoint.get('test_nmse', 'N/A')}")
```

如果损失很高（> 0.1），说明模型需要更多训练。

---

### Q3: 训练模式和 ONNX 模式哪个更好？

**推荐**: 用训练模式训练，导出时切换到 ONNX 模式

| 场景 | 使用模式 | 原因 |
|------|----------|------|
| 日常训练 | 训练模式 | 速度快（100% 性能） |
| ONNX 导出 | ONNX 模式 | 自动切换（export_onnx.py） |
| 验证等价性 | 两种都试 | 确保模式工作正常 |

**两种模式数学等价**，可以放心切换。

---

### Q4: 如何提高重建质量？

1. **增加训练批次**
   ```bash
   --batches 2000  # 从 1000 增加到 2000
   ```

2. **增加模型阶段数**
   ```bash
   --stages "3"  # 从 2 增加到 3
   ```

3. **使用更好的激活函数**
   ```bash
   --activation_type "cardioid"  # 尝试不同激活函数
   ```

4. **使用归一化损失函数**
   ```bash
   --loss_type "normalized"  # 对宽 SNR 范围更好
   ```

5. **增加训练数据多样性**
   ```bash
   --snr "0,30"  # 宽 SNR 范围
   --tdl "A-30,B-100,C-300"  # 多种信道
   ```

---

### Q5: 能量归一化为什么重要？

模型设计为在**归一化输入**上工作：

```python
# ✅ 正确流程
y_energy = torch.sqrt((y.abs()**2).mean())
y_normalized = y / y_energy
h_normalized = model(y_normalized)
h = h_normalized * y_energy

# ❌ 错误流程（跳过归一化）
h = model(y)  # 结果会很差！
```

**原因**: 
- 训练时所有数据都归一化
- 模型学习的是归一化空间的映射
- 未归一化的输入会导致数值不稳定

---

### Q6: 如何在 MATLAB 中批量处理？

```matlab
% 批量推理
num_samples = 100;
h_batch = zeros(num_samples, 4, 24);

for i = 1:num_samples
    [h_norm, ~] = model_func(y_batch(i, :), params, ...
                             'InputDataPermutation', 'none', ...
                             'OutputDataPermutation', 'none');
    h_batch(i, :, :) = h_norm;
end
```

---

## 📁 文件结构

```
Model_AIIC_onnx/
├── README.md                              # 📖 本文件（完整文档）
├── channel_separator.py                   # 🔧 模型定义（支持 onnx_mode）
├── complex_layers.py                      # 🔧 复数层实现
├── test_separator.py                      # 🎓 训练和评估脚本
├── export_onnx.py                         # 📤 ONNX 导出脚本
├── verify_onnx_mode_equivalence.py        # ✅ 等价性验证脚本
│
└── test/                                  # 测试输出
    └── stages=2_share=False_.../
        ├── model.pth                      # PyTorch 模型
        ├── training_curves.png            # 训练曲线
        ├── final_predictions.png          # 预测结果
        └── metrics.json                   # 训练指标

../  (项目根目录)
├── test_onnx_function.m                   # 🧪 MATLAB 测试脚本
├── test_onnx_simple.m                     # 🧪 MATLAB 简化测试
└── model.onnx                             # 📦 导出的 ONNX 模型
```

---

## 🔗 相关资源

### Python 脚本

| 脚本 | 功能 | 命令 |
|------|------|------|
| `test_separator.py` | 训练和评估 | `python Model_AIIC_onnx/test_separator.py --help` |
| `export_onnx.py` | 导出 ONNX | `python Model_AIIC_onnx/export_onnx.py --help` |
| `verify_onnx_mode_equivalence.py` | 验证等价性 | `python Model_AIIC_onnx/verify_onnx_mode_equivalence.py` |

### MATLAB 脚本

| 脚本 | 功能 | 命令 |
|------|------|------|
| `test_onnx_function.m` | 完整测试（推荐） | `test_onnx_function` |
| `test_onnx_simple.m` | 快速测试 | `test_onnx_simple` |

### 文档和资源

- **PyTorch 文档**: https://pytorch.org/docs/
- **ONNX 文档**: https://onnx.ai/
- **MATLAB ONNX 导入**: https://www.mathworks.com/help/deeplearning/ref/importonnxfunction.html

---

## 📝 版本历史

### v2.0.0 (2025-12-05) - onnx_mode 支持

- ✅ 添加 `onnx_mode` 超参数
- ✅ 支持 ONNX Opset 9 导出
- ✅ 移除能量归一化到模型外
- ✅ 避免就地赋值操作（index_put）
- ✅ 完整的 MATLAB 部署支持
- ✅ 等价性验证脚本

### v1.0.0 - 初始版本

- ✅ 基于实数张量的通道分离器
- ✅ 多种复数激活函数
- ✅ ONNX 导出支持

---

## 🎯 快速命令参考

### 训练

```bash
# 快速测试
python Model_AIIC_onnx/test_separator.py --batches 50 --stages "2" --save_dir "./test"

# 完整训练
python Model_AIIC_onnx/test_separator.py --batches 1000 --batch_size 2048 --stages "2,3" --save_dir "./models"

# ONNX 模式训练（验证）
python Model_AIIC_onnx/test_separator.py --batches 1000 --stages "2" --onnx_mode --save_dir "./models_onnx"
```

### 导出

```bash
# 导出为 ONNX
python Model_AIIC_onnx/export_onnx.py --checkpoint ./models/.../model.pth --output model.onnx --opset 9

# 检查算子
python -c "import onnx; m=onnx.load('model.onnx'); ops={}; [ops.update({n.op_type:ops.get(n.op_type,0)+1}) for n in m.graph.node]; [print(f'{k}: {v}') for k,v in sorted(ops.items())]"
```

### 验证

```bash
# Python 等价性验证
python Model_AIIC_onnx/verify_onnx_mode_equivalence.py
```

### MATLAB

```matlab
% 测试 ONNX 模型
cd('c:/GitRepo/SRS_AI')
test_onnx_function('model.onnx')
```

---

## 📧 支持

遇到问题？

1. **检查文档** - 本 README 包含所有常见问题解答
2. **验证等价性** - 运行 `verify_onnx_mode_equivalence.py`
3. **检查算子** - 确认 ONNX 文件包含的算子
4. **MATLAB 版本** - 确保使用 R2020b 或更新版本

---

**Happy Training! 🚀**
