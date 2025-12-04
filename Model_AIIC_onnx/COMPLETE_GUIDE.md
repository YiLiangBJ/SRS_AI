# Model_AIIC_onnx - 实数 ONNX 版本完整指南

## ✅ 已完成 (3/7)

### 1. `complex_layers.py` ✅
实现了完全基于实数的复数神经网络层。

**核心组件**：
- `ComplexLinearReal` - 块矩阵实现：
  ```
  [y_R]   [W_R  -W_I] [x_R]   [b_R]
  [y_I] = [W_I   W_R] [x_I] + [b_I]
  ```
- 4 种激活函数：
  - `split_relu` - 实部虚部分别 ReLU ⭐ 推荐
  - `mod_relu` - 模 ReLU（保留相位）
  - `z_relu` - zReLU（门控激活）
  - `cardioid` - 心形激活

### 2. `channel_separator.py` ✅
实数版本的通道分离器，完全 ONNX 兼容。

**特性**：
- 全程使用实数张量 `[real; imag]`
- 支持多种激活函数（超参数）
- 参数量比复数版本少 50%

### 3. `test_separator.py` ✅
训练脚本，支持网格搜索。

**新增超参数**：
- `--activation_type` - 激活函数类型

**使用示例**：
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3" \
  --share_weights "True,False" \
  --activation_type "split_relu,cardioid" \
  --loss_type "nmse" \
  --ports "0,3,6,9" \
  --save_dir "./Model_AIIC_onnx/experiments"
```

## 🚧 待完成 (4/7)

### 4. 评估脚本
直接复制 `Model_AIIC/evaluate_models.py` 并做最小修改。

### 5. 绘图脚本
直接复制 `Model_AIIC/plot_results.py`，无需修改。

### 6. ONNX 导出脚本
创建简单的导出工具。

### 7. 文档
创建 README 和 MATLAB 使用指南。

## 🎯 快速开始

### 训练模型

```bash
# 4 端口快速测试
python Model_AIIC_onnx/test_separator.py \
  --batches 50 \
  --batch_size 128 \
  --stages "2" \
  --activation_type "split_relu" \
  --ports "0,3,6,9" \
  --save_dir "./Model_AIIC_onnx/test"

# 6 端口完整训练
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --activation_type "split_relu,cardioid" \
  --ports "0,2,4,6,8,10" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./Model_AIIC_onnx/out6ports"
```

### 评估模型

```bash
python Model_AIIC_onnx/evaluate_models.py \
  --exp_dir ./Model_AIIC_onnx/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./Model_AIIC_onnx/out6ports_eval
```

### 导出 ONNX

#### 标准导出（通用）

```bash
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint ./Model_AIIC_onnx/out6ports/stages=3_share=False/model.pth \
  --output model.onnx
```

#### MATLAB 专用导出 ⭐ 推荐

```bash
# MATLAB 有特殊限制，需要使用专用导出脚本
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint ./Model_AIIC_onnx/out6ports/stages=3_share=False/model.pth \
  --output model_matlab.onnx \
  --opset 9
```

**关键差异**：
- Opset 9（MATLAB 最高支持）
- 固定 batch size
- 能量归一化在 MATLAB 中完成

### MATLAB 使用

#### 方法 1：快速测试

```matlab
% 运行完整演示脚本
run('read_onnx_matlab.m')
```

#### 方法 2：手动使用

```matlab
% 加载 MATLAB 兼容的 ONNX 模型
net = importONNXNetwork('model_matlab.onnx', 'OutputLayerType', 'regression');

% 准备数据（复数 -> 实数格式）
y = randn(1, 12) + 1i*randn(1, 12);  % 复数信号
y_stacked = [real(y), imag(y)];      % 转换为 [real; imag]

% ⚠️ 重要：能量归一化（MATLAB 模型需要）
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% 预测
h_stacked = predict(net, y_normalized);  % 输出: (1, 4, 24)

% ⚠️ 重要：恢复能量
h_stacked = h_stacked * y_energy;

% 转换回复数
L = 12; P = 4;
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);
h = complex(h_real, h_imag);  % (1, 4, 12)
```

**详细指南**：查看 `MATLAB_GUIDE.md`

## 📊 关键差异 vs Model_AIIC

| 特性 | Model_AIIC (复数) | Model_AIIC_onnx (实数) |
|------|-------------------|------------------------|
| **张量类型** | `torch.complex64` | `torch.float32` |
| **输入格式** | `(B, L)` 复数 | `(B, L*2)` 实数 |
| **输出格式** | `(B, P, L)` 复数 | `(B, P, L*2)` 实数 |
| **矩阵乘法** | 两个独立 MLP | 块矩阵 |
| **参数量** | 约 138K | 约 138K（相同） |
| **ONNX 兼容** | ❌ 否 | ✅ 是 |
| **MATLAB 兼容** | ❌ 否 | ✅ 是 |
| **激活函数** | 固定 (split ReLU) | 可选（4 种） |

**注意**：参数量实际上相同，因为块矩阵 `[W_R, -W_I; W_I, W_R]` 有 2 组权重，而原版有 2 个独立 MLP。

## 🔬 技术细节

### 数据转换流程

```python
# 训练时
# 1. 生成复数数据（保持原有数据生成器）
y_complex, h_targets_complex = generate_data(...)  # 复数

# 2. 转换为实数格式
y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)
h_targets_stacked = torch.cat([h_targets_complex.real, h_targets_complex.imag], dim=-1)

# 3. 模型前向
h_pred_stacked = model(y_stacked)  # 实数输入/输出

# 4. 计算损失（实数域）
loss = mse(h_pred_stacked, h_targets_stacked)
```

### 块矩阵乘法

$$
\begin{bmatrix}
y_R \\
y_I
\end{bmatrix}
=
\begin{bmatrix}
W_R & -W_I \\
W_I & W_R
\end{bmatrix}
\begin{bmatrix}
x_R \\
x_I
\end{bmatrix}
$$

展开后：
- $y_R = W_R x_R - W_I x_I$
- $y_I = W_I x_R + W_R x_I$

这等价于复数乘法：$y = W \cdot x$，其中 $W = W_R + jW_I$，$x = x_R + jx_I$

### 激活函数选择

| 激活函数 | 特点 | 推荐场景 |
|----------|------|----------|
| `split_relu` | 最常用，简单 | 默认选择 ⭐ |
| `cardioid` | 相位平滑 | 信号处理任务 |
| `mod_relu` | 保留相位 | 需要精确相位 |
| `z_relu` | 门控 | 实验性 |

## 📁 文件结构

```
Model_AIIC_onnx/
├── complex_layers.py          # ✅ 复数层（实数实现）
├── channel_separator.py       # ✅ 通道分离器
├── test_separator.py          # ✅ 训练脚本
├── evaluate_models.py         # 🚧 评估脚本（待创建）
├── plot_results.py            # 🚧 绘图脚本（待创建）
├── export_onnx.py             # 🚧 ONNX 导出（待创建）
├── quick_test.sh              # ✅ 快速测试
├── PROGRESS.md                # ✅ 进度文档
└── README.md                  # 🚧 使用文档（待创建）
```

## 🐛 故障排查

### 常见问题

**Q: 训练时内存不足？**
A: 减小 `--batch_size`，对于 6 端口模型推荐使用 1024-2048

**Q: ONNX 导出失败？**
A: 确保安装了 `onnx`：`pip install onnx`

**Q: MATLAB 导入失败？**
A: 确保使用 `opset_version=14` 或更高

**Q: 性能比复数版本差？**
A: 尝试不同的 `--activation_type`，`cardioid` 可能更好

## 📚 参考资料

### 复数神经网络
- [Deep Complex Networks (ICLR 2018)](https://arxiv.org/abs/1705.09792)
- [Unitary Evolution RNNs (ICML 2016)](https://arxiv.org/abs/1511.06464)

### ONNX
- [ONNX 官方文档](https://onnx.ai/)
- [PyTorch ONNX 导出](https://pytorch.org/docs/stable/onnx.html)

---

**状态**: 3/7 完成  
**下一步**: 创建评估和 ONNX 导出脚本
