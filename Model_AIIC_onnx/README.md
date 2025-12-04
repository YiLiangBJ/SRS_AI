# Model_AIIC_onnx - ONNX Compatible Channel Separator

完全基于实数张量的通道分离器，可直接导出为 ONNX 并在 MATLAB 中使用。

## 🎯 主要特性

- ✅ **ONNX 兼容** - 全程使用实数张量，无复数类型
- ✅ **MATLAB 部署** - 可直接导入 MATLAB 进行推理
- ✅ **数学等价** - 使用块矩阵实现，完全等价于复数运算
- ✅ **多种激活** - 支持 4 种复数激活函数（超参数）
- ✅ **参数高效** - 与原复数版本参数量相同

## 📦 安装

```bash
# 已有环境，无需额外安装
cd Model_AIIC_onnx
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 快速测试（4 端口，50 批次）
python Model_AIIC_onnx/test_separator.py \
  --batches 50 \
  --batch_size 128 \
  --stages "2" \
  --activation_type "split_relu" \
  --ports "0,3,6,9" \
  --save_dir "./Model_AIIC_onnx/test"

# 完整训练（6 端口，多配置）
python Model_AIIC_onnx/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --activation_type "split_relu,cardioid" \
  --loss_type "nmse" \
  --ports "0,2,4,6,8,10" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./Model_AIIC_onnx/out6ports"
```

### 2. 评估性能

```bash
python Model_AIIC_onnx/evaluate_models.py \
  --exp_dir ./Model_AIIC_onnx/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./Model_AIIC_onnx/out6ports_eval
```

### 3. 可视化结果

```bash
# 按 TDL 分图
python Model_AIIC_onnx/plot_results.py \
  --input ./Model_AIIC_onnx/out6ports_eval \
  --layout subplots_tdl

# 单图显示
python Model_AIIC_onnx/plot_results.py \
  --input ./Model_AIIC_onnx/out6ports_eval \
  --layout single
```

### 4. 导出 ONNX

```bash
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint ./Model_AIIC_onnx/out6ports/stages=3_share=False_act=split_relu/model.pth \
  --output model.onnx
```

### 5. MATLAB 使用

```matlab
%% 加载 ONNX 模型
net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');

%% 准备输入数据
% 生成复数信号
y_complex = randn(1, 12) + 1i*randn(1, 12);

% 转换为实数格式 [real; imag]
y_real_imag = [real(y_complex), imag(y_complex)];  % (1, 24)

%% 推理
h_real_imag = predict(net, y_real_imag);  % (1, 6, 24)

%% 转换回复数
L = 12;  % 序列长度
P = 6;   % 端口数

h_real = h_real_imag(:, :, 1:L);
h_imag = h_real_imag(:, :, L+1:end);
h_complex = complex(h_real, h_imag);  % (1, 6, 12)

%% 显示结果
disp('Separated channels:');
disp(size(h_complex));  % [1, 6, 12]
```

## 📖 参数说明

### 训练参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--batches` | 训练批次数 | 10000 | `1000` |
| `--batch_size` | 批大小 | 2048 | `128`, `2048` |
| `--stages` | 网络阶段数 | 3 | `"2,3,4"` |
| `--share_weights` | 权重共享 | False | `"True,False"` |
| `--activation_type` | 激活函数 ⭐ | split_relu | `"split_relu,cardioid"` |
| `--loss_type` | 损失函数 | nmse | `"nmse"`, `"normalized"` |
| `--ports` | 端口位置 | 0,3,6,9 | `"0,2,4,6,8,10"` |
| `--snr` | 信噪比 (dB) | 20.0 | `"0,30"`, `"20"` |
| `--tdl` | TDL 信道 | A-30 | `"A-30,B-100,C-300"` |
| `--save_dir` | 保存目录 | None | `"./experiments"` |

### 激活函数选项 ⭐

| 激活函数 | 特点 | 推荐度 |
|----------|------|--------|
| `split_relu` | 实部虚部分别 ReLU，最常用 | ⭐⭐⭐ 推荐 |
| `cardioid` | 心形激活，相位平滑 | ⭐⭐ 信号处理 |
| `mod_relu` | 模 ReLU，保留相位 | ⭐ 实验性 |
| `z_relu` | zReLU，门控激活 | ⭐ 实验性 |

## 🔬 技术细节

### 实数格式

所有张量使用 `[real; imag]` 格式：

```python
# 输入: y_complex = (B, L) 复数
y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)  # (B, L*2)

# 输出: h_stacked = (B, P, L*2)
h_complex = torch.complex(
    h_stacked[:, :, :L],  # 实部
    h_stacked[:, :, L:]   # 虚部
)  # (B, P, L)
```

### 块矩阵运算

复数线性变换的实数等价：

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
+
\begin{bmatrix}
b_R \\
b_I
\end{bmatrix}
$$

等价于复数乘法：$y = Wx + b$，其中 $W = W_R + jW_I$

### 参数量

与 `Model_AIIC` 相同（约 138K 参数，3 阶段，不共享权重）：
- 每个 MLP: 11,544 参数
- 4 端口 × 3 阶段 = 138,528 参数

## 📊 性能对比

### 与 Model_AIIC 对比

| 指标 | Model_AIIC | Model_AIIC_onnx |
|------|------------|-----------------|
| 张量类型 | `complex64` | `float32` |
| ONNX 兼容 | ❌ | ✅ |
| MATLAB 部署 | ❌ | ✅ |
| 训练速度 | 基准 | 相似 |
| 推理速度 | 基准 | 相似 |
| 参数量 | 138K | 138K |
| 精度 | 基准 | 相似 |

### 激活函数性能（初步）

| 激活函数 | NMSE (dB) | 收敛速度 |
|----------|-----------|----------|
| split_relu | 10.07 | 快 |
| cardioid | 7.48 | 中 |
| mod_relu | 30.68 | 慢 |

*注：未训练的初始值，仅供参考*

## 📁 项目结构

```
Model_AIIC_onnx/
├── complex_layers.py          # 复数层（实数实现）
├── channel_separator.py       # 通道分离器
├── test_separator.py          # 训练脚本
├── evaluate_models.py         # 评估脚本
├── plot_results.py            # 绘图脚本
├── export_onnx.py             # ONNX 导出
├── quick_test.sh              # 快速测试
├── COMPLETE_GUIDE.md          # 完整指南
├── PROGRESS.md                # 开发进度
└── README.md                  # 本文件
```

## 🐛 故障排查

### Q: 训练时内存不足？

**A**: 减小 `--batch_size`：
```bash
# 大内存系统
--batch_size 2048

# 中等内存
--batch_size 1024

# 小内存
--batch_size 256
```

### Q: ONNX 导出失败？

**A**: 安装 ONNX：
```bash
pip install onnx
```

### Q: MATLAB 导入失败？

**A**: 检查 MATLAB 和 ONNX 版本：
- MATLAB R2021a+ 支持 opset 14+
- 使用 `--opset 14` 参数

### Q: 性能不如预期？

**A**: 尝试不同的激活函数：
```bash
--activation_type "split_relu,cardioid"
```

### Q: 如何查看训练日志？

**A**: 使用 TensorBoard：
```bash
tensorboard --logdir ./Model_AIIC_onnx/out6ports
```

## 📚 相关文档

- [COMPLETE_GUIDE.md](COMPLETE_GUIDE.md) - 完整技术指南
- [PROGRESS.md](PROGRESS.md) - 开发进度
- [../Model_AIIC/README.md](../Model_AIIC/README.md) - 原始复数版本

## 🤝 贡献

欢迎提出问题和改进建议！

## 📄 许可证

与主项目相同

---

**版本**: 1.0.0  
**状态**: ✅ 可用  
**更新**: 2025-12-04
