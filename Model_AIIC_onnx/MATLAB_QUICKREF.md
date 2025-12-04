# MATLAB ONNX 快速参考

## 🚀 导出模型

```bash
python Model_AIIC_onnx/export_onnx_matlab.py \
  --checkpoint <path_to_model.pth> \
  --output model_matlab.onnx \
  --opset 9
```

## 📋 MATLAB 使用模板

```matlab
% 1. 加载模型
net = importONNXNetwork('model_matlab.onnx', 'OutputLayerType', 'regression');

% 2. 准备数据
y = randn(1, 12) + 1i*randn(1, 12);  % 复数信号
y_stacked = [real(y), imag(y)];      % 转换为 [real; imag]

% 3. ⚠️ 归一化能量（重要！）
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% 4. 推理
h_stacked = predict(net, y_normalized);

% 5. ⚠️ 恢复能量（重要！）
h_stacked = h_stacked * y_energy;

% 6. 转换回复数
L = 12;
h = complex(h_stacked(:,:,1:L), h_stacked(:,:,L+1:end));
```

## ⚠️ 两个关键步骤

### 推理前：归一化

```matlab
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;  % 必须！
```

### 推理后：恢复

```matlab
h_stacked = h_stacked * y_energy;  % 必须！
```

**忘记这两步 → 输出错误！**

## 🐛 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| "Operator X not supported" | 用了标准导出 | 用 `export_onnx_matlab.py` |
| "Opset 14 not supported" | 错误的 opset | 加 `--opset 9` |
| 输出值不对 | 忘记能量归一化 | 按模板加归一化 |

## 📝 完整测试脚本

```bash
# 在 MATLAB 中运行
run('read_onnx_matlab.m')
```

## 📚 详细文档

- **MATLAB_GUIDE.md** - 完整指南
- **read_onnx_matlab.m** - 演示脚本
- **export_onnx_matlab.py** - 导出工具

---

**记住**：
1. 使用 `export_onnx_matlab.py` (不是 `export_onnx.py`)
2. 推理前后做能量归一化
3. 用 Opset 9

**就这么简单！** ✅
