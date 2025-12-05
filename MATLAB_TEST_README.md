# MATLAB 测试说明

## 🎯 快速开始

### 方法 1：简化测试（推荐）⭐

在 MATLAB 命令窗口运行：

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_simple
```

这个脚本会：
- ✅ 自动查找 `model_onnx_mode.onnx`
- ✅ 导入模型并测试推理
- ✅ 验证重建误差
- ✅ 显示简洁报告（无可视化）
- ⏱️ 约 10 秒完成

### 方法 2：完整测试（带可视化）

在 MATLAB 命令窗口运行：

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_model
```

这个脚本会：
- ✅ 导入模型并测试推理
- ✅ 生成 6 个可视化图表
- ✅ 详细的诊断信息
- ⏱️ 约 30 秒完成

---

## 📋 前提条件

### 1. 导出 ONNX 模型

如果 `model_onnx_mode.onnx` 不存在，先导出：

```bash
cd c:/GitRepo/SRS_AI

python Model_AIIC_onnx/export_onnx.py \
  --checkpoint Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_onnx_mode.onnx \
  --opset 9
```

### 2. MATLAB 环境

- **MATLAB 版本**: R2020b 或更新
- **工具箱**: Deep Learning Toolbox
- **操作系统**: Windows/Linux/macOS

检查工具箱：
```matlab
ver('deeplearning')
```

---

## 🔍 测试脚本对比

| 特性 | test_onnx_simple | test_onnx_model |
|------|------------------|-----------------|
| 导入测试 | ✅ | ✅ |
| 推理测试 | ✅ | ✅ |
| 重建验证 | ✅ | ✅ |
| 可视化图表 | ❌ | ✅ (6 个) |
| 能量分布 | ✅ 简洁 | ✅ 详细 |
| 执行时间 | ~10 秒 | ~30 秒 |
| 适用场景 | 快速验证 | 深入分析 |

---

## ⚠️ 常见问题

### 问题 1: "Model file not found"

**原因**：未导出 ONNX 模型

**解决**：
```bash
cd c:/GitRepo/SRS_AI
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model_onnx_mode.onnx \
  --opset 9
```

---

### 问题 2: "Operator 'Slice' is not supported"

**原因**：当前导出的模型仍包含 MATLAB 不支持的算子：
- `Slice` (84个)
- `Unsqueeze` (17个)
- `Gather` (32个)

**解决方案 A**：使用 `importONNXFunction`（如果 MATLAB R2022a+）

在 MATLAB 中尝试：
```matlab
net = importONNXFunction('model_onnx_mode.onnx', 'model_func');
h_normalized = net(y_normalized);
```

**解决方案 B**：进一步修改模型

需要完全避免：
1. 索引操作 `features[:, port_idx, :]` → 生成 `Slice`/`Gather`
2. `.unsqueeze()` 操作 → 生成 `Unsqueeze`

这需要更深入的架构改造。

---

### 问题 3: "Deep Learning Toolbox not found"

**解决**：
1. 打开 MATLAB
2. 点击 "Add-Ons" → "Get Add-Ons"
3. 搜索 "Deep Learning Toolbox"
4. 点击安装

---

### 问题 4: 重建误差很大（> 50%）

**可能原因**：
1. ❌ 忘记能量归一化
2. ❌ 忘记恢复能量
3. ❌ 输入格式错误

**检查清单**：
```matlab
% ✅ 正确的流程
y_complex = randn(1, 12) + 1i*randn(1, 12);
y_stacked = [real(y_complex), imag(y_complex)];

% ⚠️ 必须归一化
y_energy = sqrt(mean(abs(y_complex).^2));
y_normalized = y_stacked / y_energy;

% 推理
h_normalized = predict(net, y_normalized);

% ⚠️ 必须恢复能量
h_stacked = h_normalized * y_energy;
```

---

## 📊 期望输出

### 成功的测试输出

```
========================================
ONNX Model Quick Test (onnx_mode=True)
========================================

Step 1: Importing ONNX model...
  Path: model_onnx_mode.onnx
  ✓ SUCCESS: Model imported!

Step 2: Testing inference...
  Input shape: (1, 24)
  Input energy: 1.234567
  Normalized energy: 1.000000
  ✓ Inference successful! (8.32 ms)
  Output shape: (1, 4, 24)

Step 3: Restoring energy...
Step 4: Converting to complex...
  Number of ports: 4
  Output shape: (1, 4, 12)

Step 5: Verifying reconstruction...
  Reconstruction error: 1.23e-02 (1.23%)
  Reconstruction error: -38.21 dB
  ✓ EXCELLENT reconstruction quality!

Step 6: Energy distribution per port:
  Port 1: 0.423456 (14.5% of input)
  Port 2: 0.356789 (10.3% of input)
  Port 3: 0.289012 (6.8% of input)
  Port 4: 0.154321 (1.9% of input)

========================================
Summary
========================================
Model:              model_onnx_mode.onnx
Import:             ✓ SUCCESS
Inference:          ✓ SUCCESS (8.32 ms)
Reconstruction:     1.23% (-38.21 dB)
Number of ports:    4
========================================
✓ All tests passed!
```

---

## 🎯 下一步

### 如果测试成功 ✅

1. **用真实数据测试**：
   ```matlab
   % 加载实际 SRS 数据
   load('real_srs_data.mat');
   
   % 归一化
   y_energy = sqrt(mean(abs(y_srs).^2));
   y_normalized = [real(y_srs), imag(y_srs)] / y_energy;
   
   % 推理
   h_normalized = predict(net, y_normalized);
   h_stacked = h_normalized * y_energy;
   ```

2. **批量处理**：
   ```matlab
   % 处理多个样本
   for i = 1:num_samples
       h_batch(:, :, :, i) = predict(net, y_batch(i, :));
   end
   ```

3. **性能基准测试**：
   ```matlab
   % 测试推理速度
   tic;
   for i = 1:1000
       h = predict(net, y_normalized);
   end
   avg_time = toc / 1000;
   fprintf('Average inference time: %.2f ms\n', avg_time * 1000);
   ```

### 如果测试失败 ❌

1. **检查算子兼容性**：
   ```python
   # 检查 ONNX 模型中的算子
   python -c "import onnx; model = onnx.load('model_onnx_mode.onnx'); ops = {}; [ops.update({n.op_type: ops.get(n.op_type, 0) + 1}) for n in model.graph.node]; [print(f'{op}: {count}') for op, count in sorted(ops.items())]"
   ```

2. **尝试更新的 MATLAB 版本**：
   - R2022a+ 有更好的 ONNX 支持
   - 支持更多算子

3. **联系开发者**：
   - 提供错误信息
   - 提供 MATLAB 版本
   - 提供 `model_onnx_mode.onnx` 文件

---

## 📚 相关文档

- `ONNX_MODE_SUMMARY.md` - ONNX 模式实现总结
- `DEPLOYMENT_GUIDE.md` - 完整部署指南
- `verify_onnx_mode_equivalence.py` - Python 等价性验证

---

**准备好了吗？运行测试吧！** 🚀

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_simple
```
