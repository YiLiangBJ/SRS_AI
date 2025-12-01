# 功能更新总结

## ✅ 完成的功能

### 1. Markdown 训练报告

每次训练后自动生成 `training_report.md`，包含：
- 模型配置
- 训练配置
- 训练结果
- 端口性能分析
- 模型使用说明

**位置**：`{save_dir}/{experiment_name}/training_report.md`

### 2. 移除 Epoch 概念

所有地方改用"batch"或"iteration"：
- ❌ `num_epochs` → ✅ `num_batches_trained`
- ❌ "Epochs" 列 → ✅ "Batches" 列
- 所有打印和文档使用 "batches" 而不是 "epochs"

### 3. NMSE 计算简化

移除了 shifted/unshifted 两个版本的 NMSE：
- ✅ 只保留一个 NMSE（因为 h_targets 本身就包含 shift）
- ✅ 同时打印线性值和 dB 值
- ✅ 端口级别也显示两种格式

**打印示例**：
```
Batch 20/100, Loss: 0.082974 (-10.81 dB), Throughput: 2343 samples/s
Test NMSE: 0.018155 (-17.41 dB)
Port-wise NMSE (linear): ['0.017479', '0.017863', '0.018753', '0.018525']
Port-wise NMSE (dB):     ['-17.57', '-17.48', '-17.27', '-17.32'] dB
```

### 4. ONNX 导出问题解决

**问题**：ONNX 不支持复数类型（ComplexFloat）

**解决方案**：
- ❌ ONNX 导出（不支持复数）
- ✅ TorchScript 导出（`.pt` 格式，支持复数）

**优势**：
- ✅ 完整保留模型（包括复数运算）
- ✅ Python/C++ 可直接加载
- ✅ 推理性能更好

**使用示例**：
```python
import torch

# 加载 TorchScript 模型
model = torch.jit.load('model.pt')
model.eval()

# 推理
y = torch.randn(1, 12, dtype=torch.complex64)
h = model(y)  # [1, 4, 12]
```

---

## 📁 输出文件结构

```
{save_dir}/
├── {experiment_name}/
│   ├── model.pth              # PyTorch 权重
│   ├── model.pt               # TorchScript (推荐用于部署)
│   ├── metrics.json           # 详细指标
│   ├── train_losses.npy       # 训练曲线
│   ├── val_losses.npy         # 验证曲线（如果有）
│   └── training_report.md     # 📄 Markdown 报告
└── search_summary.json        # 所有实验汇总
```

---

## 🎯 使用建议

### 单次训练

```bash
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages 3 \
  --save_dir "./my_training"
```

**输出**：
- 终端显示训练进度
- `my_training/stages=3_share=False/training_report.md` - 可读性强的报告

### 超参数搜索

```bash
python Model_AIIC/test_separator.py \
  --batches 5000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --save_dir "./hyperparam_search"
```

**输出**：
- 6 个实验的独立 Markdown 报告
- `hyperparam_search/search_summary.json` - JSON 格式汇总
- 终端显示排名表格

### 查看训练报告

```bash
# Linux/Mac
cat my_training/stages=3_share=False/training_report.md

# Windows
type my_training\stages=3_share=False\training_report.md

# 或用 VS Code 打开
code my_training/stages=3_share=False/training_report.md
```

---

## 🔄 MATLAB 集成建议

由于 ONNX 不支持复数，有以下选项：

### 选项 1：使用 Python Engine（推荐）

```matlab
% 配置 Python
pyenv('Version', 'path/to/python');

% 加载模型
torch = py.importlib.import_module('torch');
model = torch.jit.load('model.pt');
model.eval();

% 准备数据（MATLAB → Python）
y_matlab = randn(12, 1) + 1i*randn(12, 1);
y_real = py.numpy.array(real(y_matlab));
y_imag = py.numpy.array(imag(y_matlab));
% ... 需要更多转换逻辑
```

### 选项 2：重新实现模型（推荐用于生产）

从 `model.pth` 读取权重，用 MATLAB 重新实现网络：

```matlab
% 加载权重
weights = py.torch.load('model.pth');

% 用 MATLAB 实现相同的网络结构
% （参考 Model_AIIC/channel_separator.py）
```

### 选项 3：导出中间格式

修改模型为实数版本（real/imag 分离），然后导出 ONNX。这需要重构模型代码。

---

## 📊 Markdown 报告示例

见 `final_test/stages=2_share=False/training_report.md`：
- ✅ 清晰的表格格式
- ✅ 配置参数一目了然
- ✅ 性能指标（线性 + dB）
- ✅ 模型使用示例
- ✅ 易于版本控制（文本格式）
- ✅ GitHub/GitLab 自动渲染

---

## 🎉 总结

| 功能 | 状态 |
|------|------|
| Markdown 报告 | ✅ 完成 |
| 移除 Epoch | ✅ 完成 |
| NMSE 简化 | ✅ 完成 |
| 线性+dB 显示 | ✅ 完成 |
| ONNX 问题 | ✅ 改用 TorchScript |
| 模型部署 | ✅ TorchScript (.pt) |
| 依赖清理 | ✅ 移除 TensorFlow/Sionna |

所有功能已测试通过！🚀

---

## 🧹 依赖清理 (2025-12-01)

### 移除的依赖
- ❌ `tensorflow-cpu==2.15.0` - 已移除
- ❌ `sionna==0.17.0` - 已移除
- ❌ `tensorboard==2.15.0` - 已移除（不再需要）

### 替代方案
- ✅ 使用 `Model_AIIC.tdl_channel.TDLChannel` - 纯 NumPy 实现
- ✅ 无 GIL 限制，CPU 并行性能更好
- ✅ 完全符合 3GPP TR 38.901 标准

### 修改的文件
- `pyproject.toml` - 移除 TensorFlow/Sionna 依赖
- `Model_AIIC/test_separator.py` - 移除未使用的导入
- `Model_AIIC/channel_models.py` - 清理旧的 TDL 代码，只保留 SimpleRayleighChannel

### 迁移指南
如果其他脚本还依赖 Sionna/TensorFlow，请：
1. 使用 `Model_AIIC.tdl_channel.TDLChannel` 替代
2. 或保留旧的 `channel_models_old.py` 作为参考
