# MATLAB ONNX 极度简化版本说明

## ⚠️ 问题根源

MATLAB 的 ONNX 导入器**极度受限**，即使 Opset 9 也不支持：

### 不支持的基础操作

| 操作 | PyTorch | MATLAB 支持 |
|------|---------|-------------|
| `torch.stack()` | ✅ | ❌ |
| `torch.cat()` (非最后维度) | ✅ | ❌ |
| `tensor.unsqueeze()` | ✅ | ❌ |
| `tensor.expand()` | ✅ | ❌ |
| `tensor[:, :, start:end]` | ✅ | ❌ |
| `tensor.sum(dim=1)` | ✅ | ❌ |

### 我们原始模型中使用的操作

```python
# 这些操作在 MATLAB 中都不支持！
features = y.unsqueeze(1).expand(-1, num_ports, -1)  # ❌
y_recon = features.sum(dim=1)  # ❌
residual = y - y_recon  # ❌ (如果不是简单向量)
features[:, :, :L]  # ❌ 动态切片
```

---

## ✅ 解决方案：极度简化

### 修改策略

1. **移除所有多维操作**
   - 不用 `(B, P, L)` 三维张量
   - 只用 `(B, features)` 二维张量

2. **移除残差耦合**
   - 原本：所有端口通过残差相互影响
   - 现在：每个端口独立处理

3. **扁平化输出**
   - 原本：`(1, P, L*2)` 三维
   - 现在：`(1, P*L*2)` 二维（扁平）
   - 在 MATLAB 中 reshape

### 新的前向传播

```python
def forward(self, y_real):
    outputs = []
    
    # 每个端口独立处理（没有残差耦合）
    for port_idx in range(num_ports):
        x = y_real  # (1, L*2)
        
        # 通过该端口的所有阶段
        for stage_idx in range(num_stages):
            x = mlp(x)  # (1, L*2) -> (1, L*2)
        
        outputs.append(x)
    
    # 拼接为扁平向量（MATLAB 可以 reshape）
    return torch.cat(outputs, dim=-1)  # (1, P*L*2)
```

---

## 📊 对比

### 原始模型（完整功能）

```python
# 初始化：所有端口从 y 开始
features = y.unsqueeze(1).expand(-1, P, -1)  # (1, P, L*2) ❌

# 迭代优化
for stage in range(num_stages):
    # 每个端口独立处理
    for p in range(P):
        features[:, p] = mlp(features[:, p])
    
    # 残差耦合：所有端口相互影响
    y_recon = features.sum(dim=1)  # ❌
    residual = y - y_recon  # ❌
    features = features + residual.unsqueeze(1)  # ❌

return features  # (1, P, L*2) ❌
```

**问题**：
- `unsqueeze` ❌
- `expand` ❌
- `sum(dim=1)` ❌
- 三维张量 ❌

### 简化模型（MATLAB 兼容）

```python
# 每个端口独立处理（无耦合）
outputs = []
for p in range(P):
    x = y  # (1, L*2) ✅
    
    for stage in range(num_stages):
        x = mlp(x)  # (1, L*2) -> (1, L*2) ✅
    
    outputs.append(x)  # List of (1, L*2) ✅

# 扁平化输出
return torch.cat(outputs, dim=-1)  # (1, P*L*2) ✅
```

**优点**：
- 只有 2D 张量 ✅
- 只有 `torch.cat` 在最后一维 ✅
- 无 `unsqueeze/expand/sum` ✅

---

## ⚠️ 性能影响

### 移除的功能

1. **残差耦合**
   - **原本**：各端口通过残差相互调整，提高分离质量
   - **现在**：各端口独立，无相互影响
   - **影响**：性能可能下降 5-10%

2. **初始化策略**
   - **原本**：所有端口从相同的 y 开始
   - **现在**：每个端口独立从 y 开始（实际上相同）
   - **影响**：无

3. **能量归一化**
   - **原本**：在模型内部
   - **现在**：在 MATLAB 中
   - **影响**：无（数学等价）

### 预期性能

| 指标 | 完整模型 | 简化模型 |
|------|----------|----------|
| NMSE | -25 dB | -22 dB (估计) |
| 训练收敛 | 快 | 稍慢 |
| 推理速度 | 基准 | 相似 |
| MATLAB 兼容 | ❌ | ✅ |

---

## 🔧 使用方法

### 1. 导出模型

```bash
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_matlab.onnx \
  --opset 9
```

### 2. MATLAB 中使用

```matlab
% 加载
net = importONNXNetwork('model_matlab.onnx', 'OutputLayerType', 'regression');

% 准备输入
y = randn(1, 12) + 1i*randn(1, 12);
y_stacked = [real(y), imag(y)];

% 归一化
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% 推理（输出是扁平的）
h_flat = predict(net, y_normalized);  % (1, 96) for 4 ports

% ⚠️ 重要：Reshape
L = 12; P = 4;
h_stacked = reshape(h_flat, [1, P, L*2]);  % (1, 4, 24)

% 恢复能量
h_stacked = h_stacked * y_energy;

% 转换为复数
h = complex(h_stacked(:,:,1:L), h_stacked(:,:,L+1:end));
```

---

## 🎯 关键点

### 为什么这样能工作？

1. **只用简单的线性层**
   ```
   Input (1, 24) → MLP → Output (1, 24)
   ```

2. **多个端口 = 多次运行**
   ```
   Port 1: y → MLP → h1
   Port 2: y → MLP → h2
   Port 3: y → MLP → h3
   Port 4: y → MLP → h4
   Concatenate: [h1, h2, h3, h4] → (1, 96)
   ```

3. **Reshape 在 MATLAB**
   ```matlab
   (1, 96) → reshape → (1, 4, 24)
   ```

### 为什么移除残差耦合？

残差计算需要：
```python
y_recon = features.sum(dim=1)  # ❌ MATLAB 不支持
residual = y - y_recon          # ❌ 张量形状不匹配
features = features + residual  # ❌ Broadcasting 问题
```

**权衡**：牺牲少量性能换取 MATLAB 兼容性。

---

## 📈 测试结果

### 应该能工作

```matlab
>> net = importONNXNetwork('model_matlab.onnx', 'OutputLayerType', 'regression');
% 应该无错误 ✅

>> h_flat = predict(net, y_normalized);
>> size(h_flat)
ans =
     1    96  % (1, 4*12*2) ✅
```

### 如果还有错误

可能需要进一步简化：
1. 移除所有 BatchNorm/LayerNorm
2. 只用最基础的 Linear + ReLU
3. 考虑使用 `importONNXFunction` 而不是 `importONNXNetwork`

---

## 💡 备选方案

如果简化版本仍然失败：

### 方案 1：使用 importONNXFunction

```matlab
% 使用函数导入（支持更多算子）
net = importONNXFunction('model_matlab.onnx', 'model');
h_flat = model(y_normalized);
```

### 方案 2：直接导出权重

```python
# 导出为 .mat 文件
import scipy.io as sio
weights = {k: v.numpy() for k, v in model.state_dict().items()}
sio.savemat('model_weights.mat', weights)
```

```matlab
% 在 MATLAB 中手动实现前向传播
load('model_weights.mat');
h = manual_forward(y, weights);
```

### 方案 3：使用 ONNX Runtime MEX

编译 ONNX Runtime 的 MATLAB 接口（最复杂但最强大）。

---

## 📝 总结

**当前策略**：
- 极度简化模型结构
- 移除残差耦合
- 扁平化输出
- Reshape 在 MATLAB

**如果成功**：
- ✅ MATLAB 直接使用
- ⚠️ 性能可能略降

**如果失败**：
- 尝试 `importONNXFunction`
- 或直接导出权重到 `.mat`

---

**测试命令**：
```bash
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_matlab.onnx
```

```matlab
run('read_onnx_matlab.m')
```
