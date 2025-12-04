# Model_AIIC_onnx 实现进度

## ✅ 已完成

### 1. 复数层模块 (`complex_layers.py`)

**功能**：
- ✅ `ComplexLinearReal` - 使用实数块矩阵实现复数线性层
- ✅ 4 种复数激活函数：
  - `split_relu` - 实部虚部分别 ReLU（推荐，最常用）
  - `mod_relu` - 模 ReLU（保留相位）
  - `z_relu` - zReLU（门控激活）
  - `cardioid` - 心形激活（平滑相位依赖）
- ✅ `ComplexMLPReal` - 3 层 MLP（可选激活函数）
- ✅ 测试通过

**数学基础**：
```
Block matrix representation:
[y_R]   [W_R  -W_I] [x_R]   [b_R]
[y_I] = [W_I   W_R] [x_I] + [b_I]
```

### 2. 通道分离器 (`channel_separator.py`)

**功能**：
- ✅ `ResidualRefinementSeparatorReal` - 全实数版本
- ✅ 支持权重共享 (`share_weights_across_stages`)
- ✅ 支持能量归一化 (`normalize_energy`)
- ✅ 支持多种激活函数 (`activation_type`)
- ✅ 测试通过

**参数量**：
- `share=False, 4 ports, 3 stages`: 138,528 参数
- `share=True, 4 ports, 3 stages`: 46,176 参数
- **比 Model_AIIC 的复数版本减少约 50%**

## 🚧 待完成

### 3. 训练脚本 (`test_separator.py`)

**需要实现**：
1. 复制 `Model_AIIC/test_separator.py` 的结构
2. 数据生成：复数形式生成，输入网络前转换为 `[real; imag]`
3. 添加 `--activation_type` 超参数
4. 训练循环：
   - 输入：将复数转换为 stacked real
   - 网络：使用 `ResidualRefinementSeparatorReal`
   - 损失：在实数域计算 NMSE
   - 输出：转换回复数进行评估（可选）
5. 保存模型和配置

**关键修改点**：
```python
# 数据转换
y_complex = generate_data(...)  # 复数
y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)

# 模型
model = ResidualRefinementSeparatorReal(
    activation_type=args.activation_type  # 新增
)

# 前向传播
h_stacked = model(y_stacked)

# 损失计算（实数域）
target_stacked = torch.cat([target_complex.real, target_complex.imag], dim=-1)
loss = nmse_loss(h_stacked, target_stacked)
```

### 4. 评估脚本 (`evaluate_models.py`)

**需要实现**：
1. 复制 `Model_AIIC/evaluate_models.py`
2. 加载实数模型
3. 数据转换：复数 → stacked real → 模型 → stacked real → 复数
4. 评估并保存结果

### 5. 绘图脚本 (`plot_results.py`)

**需要实现**：
1. 直接复制 `Model_AIIC/plot_results.py`
2. 无需修改（读取评估结果 JSON）

### 6. ONNX 导出脚本

**需要实现**：
```python
# export_onnx.py
def export_to_onnx(checkpoint_path, output_path):
    # 加载模型
    model = ResidualRefinementSeparatorReal(...)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 导出
    torch.onnx.export(
        model,
        torch.randn(1, seq_len * 2),
        output_path,
        input_names=['y_real_imag'],
        output_names=['h_real_imag'],
        opset_version=14
    )
```

### 7. 文档和 README

**需要创建**：
- `README.md` - 完整使用指南
- MATLAB 集成示例
- 与 `Model_AIIC` 的对比

## 📊 测试结果

### 初始性能（未训练）

| 配置 | 参数量 | NMSE (dB) |
|------|--------|-----------|
| share=False, split_relu | 138,528 | 10.07 |
| share=True, split_relu | 46,176 | 9.10 |
| share=False, mod_relu | 138,528 | 30.68 |
| share=False, cardioid | 138,528 | 7.48 |

**观察**：
- `split_relu` 和 `cardioid` 表现最好
- `mod_relu` 需要调整（可能需要不同的 bias）
- 权重共享减少约 67% 参数量

## 🎯 下一步

### 优先级 1：创建训练脚本

这是最重要的，因为需要训练模型才能评估性能。

**建议**：
1. 从 `Model_AIIC/test_separator.py` 开始
2. 复制整个文件
3. 修改数据转换部分
4. 修改模型创建部分
5. 测试训练一个小模型

### 优先级 2：ONNX 导出

一旦训练好模型，立即导出 ONNX 并在 MATLAB 中测试。

### 优先级 3：完整评估流程

实现评估和绘图，生成完整的性能报告。

## 💡 使用建议

### 激活函数选择

根据初步测试：
1. **`split_relu`** - 推荐用于快速实验（最常用）
2. **`cardioid`** - 可能更好（相位平滑）
3. **`mod_relu`** - 需要调整（保留相位可能有用）
4. **`z_relu`** - 待测试

### 超参数建议

从 `Model_AIIC` 继承的推荐配置：
- `seq_len=12`
- `hidden_dim=64`
- `num_stages=3`
- `share_weights=False`（更多参数，可能更好）
- `batch_size=2048`（CPU）

新增参数：
- `activation_type='split_relu'`（推荐从这个开始）

## 📝 代码模板

### 快速训练命令

```bash
# 训练 4 端口模型
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3" \
  --share_weights "True,False" \
  --activation_type "split_relu,cardioid" \
  --ports "0,3,6,9" \
  --save_dir "./Model_AIIC_onnx/experiments"

# 导出 ONNX
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint "./Model_AIIC_onnx/experiments/stages=3_share=False/model.pth" \
  --output "model.onnx"
```

### MATLAB 使用

```matlab
% 加载模型
net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');

% 准备数据
y = randn(1, 12) + 1i*randn(1, 12);  % 复数
y_stacked = [real(y), imag(y)];      % [real; imag]

% 预测
h_stacked = predict(net, y_stacked);  % (1, 4, 24)

% 转换回复数
L = 12; P = 4;
h = complex(h_stacked(:, :, 1:L), h_stacked(:, :, L+1:end));
```

---

**状态**: 2/7 完成  
**下一步**: 创建训练脚本 `test_separator.py`
