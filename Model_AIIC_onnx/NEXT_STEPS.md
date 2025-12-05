# 下一步：ONNX 导出和 MATLAB 测试

## 🎯 你现在要做的

按照以下步骤将训练好的模型导出为 ONNX 并在 MATLAB 中测试。

---

## 📋 步骤清单

### ✅ 已完成
- [x] 代码修改为 Opset 9 兼容
- [x] 验证测试全部通过 (6/6)
- [x] 创建部署指南和脚本

### 🔄 现在要做

- [ ] **步骤 1**: 导出 ONNX 模型
- [ ] **步骤 2**: 在 MATLAB 中导入
- [ ] **步骤 3**: 运行测试脚本
- [ ] **步骤 4**: 验证结果

---

## 🚀 详细操作步骤

### 步骤 1: 导出 ONNX 模型

```bash
cd c:/GitRepo/SRS_AI

# 使用你已有的训练好的模型
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model.onnx \
  --opset 9
```

**期望看到**：
```
================================================================================
Exporting Model to ONNX Format
================================================================================
Checkpoint: ./Model_AIIC_onnx/test/...
Output:     model.onnx
Opset:      9

✓ ONNX model exported!
  File size: 0.40 MB
  Opset:     9

✓ ONNX model validated successfully!
✓ Inference test passed!
  Max difference:  5.42e-07
  Mean difference: 1.73e-07
```

**如果出错**：
```bash
# 运行诊断工具查看详细信息
python Model_AIIC_onnx/diagnose_onnx.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --opset 9
```

---

### 步骤 2: 复制模型到合适的位置

```bash
# Windows PowerShell
cp model.onnx c:/Users/YiLiang/Documents/MATLAB/

# 或者放在当前目录
# MATLAB 可以直接访问
```

---

### 步骤 3: 在 MATLAB 中测试

打开 MATLAB，然后：

```matlab
% 方法 1: 使用提供的测试脚本（推荐）
cd('c:/GitRepo/SRS_AI')
test_onnx_model('model.onnx')

% 方法 2: 手动测试
cd('c:/Users/YiLiang/Documents/MATLAB')  % 或你的 ONNX 文件路径

% 导入模型
net = importONNXNetwork('model.onnx', 'OutputLayerType', 'regression');

% 生成测试数据
L = 12;
y_complex = randn(1, L) + 1i*randn(1, L);
y_stacked = [real(y_complex), imag(y_complex)];

% 归一化
y_energy = sqrt(mean(abs(y_complex).^2));
y_normalized = y_stacked / y_energy;

% 推理
h_stacked = predict(net, y_normalized);

% 恢复能量
h_stacked = h_stacked * y_energy;

% 转换为复数
h_complex = complex(h_stacked(:,:,1:L), h_stacked(:,:,L+1:end));

% 验证
y_recon = squeeze(sum(h_complex, 2));
error = norm(y_complex - y_recon) / norm(y_complex);
fprintf('重建误差: %.2f%%\n', error * 100);
```

---

### 步骤 4: 验证结果

#### 成功的标志

使用 `test_onnx_model.m` 脚本，你应该看到：

```
========================================
ONNX Model Test (Opset 9)
========================================

Step 1: Importing ONNX model...
  ✓ Model imported successfully!

Step 2: Generating test data...
  Input shape: (1, 24)
  Input energy: 1.123456

Step 3: Energy normalization...
  ⚠️  Original energy: 1.123456
  ⚠️  Normalized energy: 1.000000
  ✓ Normalization complete

Step 4: Running inference...
  ✓ Inference complete! (8.32 ms)
  Output shape: (1, 4, 24)

Step 5: Restoring energy...
  ✓ Energy restored

Step 6: Converting to complex format...
  Output shape: (1, 4, 12)
  Number of ports: 4

  Energy per port:
    Port 1: 0.423456
    Port 2: 0.356789
    Port 3: 0.289012
    Port 4: 0.154321

Step 7: Verifying reconstruction...
  Reconstruction error: 1.23e-02 (1.23%)
  Reconstruction error (dB): -38.21 dB
  ✓ Excellent reconstruction quality!

Step 8: Generating visualization...
  ✓ Visualization generated

========================================
Summary
========================================
Model:              model.onnx
Sequence length:    12
Number of ports:    4
Inference time:     8.32 ms
Reconstruction err: 1.23% (-38.21 dB)
========================================
✓ Test complete!
```

**并且会弹出一个图形窗口，显示 6 个子图**：
1. 输入信号（实部和虚部）
2. 分离的通道（幅度）
3. 重建对比（实部）
4. 重建对比（虚部）
5. 能量分布
6. 重建误差

---

## ⚠️ 常见问题和解决方案

### 问题 1: MATLAB 导入失败

**错误信息**：
```
Error: Operator 'Slice' is not supported.
```

**解决**：
```bash
# 确认使用了 --opset 9
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint <path> \
  --output model.onnx \
  --opset 9  # ← 重要！
```

---

### 问题 2: 重建误差很大

**症状**：重建误差 > 50%

**可能原因**：
1. 忘记能量归一化
2. 忘记恢复能量
3. 模型未训练好

**检查清单**：
```matlab
% 1. 检查是否归一化
y_energy = sqrt(mean(abs(y_complex).^2));
y_normalized = y_stacked / y_energy;  % ← 必须做

% 2. 检查是否恢复能量
h_stacked = h_stacked * y_energy;  % ← 必须做

% 3. 检查输入格式
size(y_stacked)  % 应该是 (1, 24) 对于 L=12

% 4. 检查输出格式
size(h_stacked)  % 应该是 (1, P, 24)
```

---

### 问题 3: MATLAB 版本或工具箱

**要求**：
- MATLAB R2020b 或更新
- Deep Learning Toolbox

**检查**：
```matlab
% 检查 MATLAB 版本
ver

% 检查工具箱
ver('deeplearning')
```

**如果没有 Deep Learning Toolbox**：
- 在 MATLAB 命令窗口：`Add-Ons` → `Get Add-Ons`
- 搜索 "Deep Learning Toolbox"
- 安装

---

## 📊 预期性能

### 推理速度

| 平台 | 典型延迟 | 备注 |
|------|----------|------|
| MATLAB (首次) | ~50 ms | 编译开销 |
| MATLAB (后续) | ~5-10 ms | 已编译 |

### 准确性

| 指标 | 期望值 | 说明 |
|------|--------|------|
| 重建误差 | < 5% | 训练好的模型 |
| 重建误差 (dB) | < -26 dB | 好的分离质量 |
| ONNX vs PyTorch | < 1e-6 | 数值误差 |

---

## 📝 测试报告模板

测试完成后，记录以下信息：

```
========================================
ONNX 导出和 MATLAB 测试报告
========================================
日期: 2025-12-05
操作员: [你的名字]

1. ONNX 导出
   - ✅ 成功 / ❌ 失败
   - 模型大小: ___ MB
   - Opset 版本: 9
   - 导出时间: ___ 秒

2. MATLAB 导入
   - ✅ 成功 / ❌ 失败
   - MATLAB 版本: ___
   - 导入警告: [有/无]

3. 推理测试
   - ✅ 成功 / ❌ 失败
   - 推理时间: ___ ms
   - 输出形状: (_, _, _)

4. 准确性验证
   - 重建误差: ____%
   - 重建误差 (dB): ___ dB
   - 质量评估: [优秀/良好/一般/差]

5. 问题和解决
   - [记录遇到的任何问题]
   - [记录解决方法]

6. 结论
   - ✅ 可以部署 / ❌ 需要改进
   - [其他备注]

========================================
```

---

## 🎯 成功标准

完成以下所有项，即可认为测试成功：

- [ ] ONNX 导出无错误
- [ ] ONNX 文件大小约 400 KB
- [ ] ONNX Runtime 测试误差 < 1e-5
- [ ] MATLAB 成功导入模型
- [ ] MATLAB 推理无错误
- [ ] 重建误差 < 20%（未训练）或 < 5%（已训练）
- [ ] 可视化图表正常显示

---

## 📚 参考文档

- **DEPLOYMENT_GUIDE.md** - 完整部署指南 ⭐
- **OPSET9_MODIFICATIONS_SUMMARY.md** - Opset 9 修改总结
- **test_onnx_model.m** - MATLAB 测试脚本

---

## 🚀 下一步（测试成功后）

1. 使用完整训练数据训练模型
2. 导出最佳模型
3. 在 MATLAB 中进行实际数据测试
4. 如需要，转换到 OpenVINO
5. 部署到生产环境

---

**准备好了吗？开始吧！** 🎉

```bash
# 第一步：导出 ONNX
python Model_AIIC_onnx/export_onnx.py \
  --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
  --output model.onnx \
  --opset 9
```

然后打开 MATLAB：
```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_model('model.onnx')
```

**祝测试顺利！** 🚀
