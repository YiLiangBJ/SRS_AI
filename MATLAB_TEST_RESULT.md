# ⚠️ MATLAB 测试结果 - 需要使用 importONNXFunction

## 📊 测试结果总结

### ❌ 失败 1: `importONNXNetwork` 不支持

```
✗ FAILED: Unable to import network because some network operators are not supported.

32 operators(s)  : Operator 'Gather' is not supported.
84 operators(s)  : Operator 'Slice' is not supported.
17 operators(s)  : Operator 'Unsqueeze' is not supported.
48 operators(s)  : The operator 'MatMul' is only supported when its inputs are one node and one constant.
26 operators(s)  : The operator 'Sub' is only supported when its inputs are one node and one constant.
```

**原因**：MATLAB 的 `importONNXNetwork` 对算子支持有限，无法导入我们的模型。

---

### ✅ 部分成功: `importONNXFunction` 可以导入

```
✓ SUCCESS: Model imported as function!
Function containing the imported ONNX network architecture was saved to the file model_func.m.
```

**但是推理失败**：
```
✗ Inference failed: Array indices must be positive integers or logical values.
```

**原因**：调用方式不正确，需要传递 `params` 参数。

---

## 🔧 解决方案

### 使用专门的测试脚本

我创建了新的测试脚本 `test_onnx_function.m`，专门用于 `importONNXFunction` 导入的模型。

在 MATLAB 中运行：

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_function
```

这个脚本会：
1. ✅ 正确导入模型（使用 `importONNXFunction`）
2. ✅ 提取模型参数
3. ✅ 正确调用推理（传递 `params`）
4. ✅ 自动处理维度变换
5. ✅ 验证重建质量

---

## 📋 当前状态

### 支持的方法

| 方法 | 状态 | 说明 |
|------|------|------|
| `importONNXNetwork` | ❌ 失败 | 算子不支持 |
| `importONNXFunction` | ✅ 成功 | 可以导入和使用 |

### 不支持的算子

当前模型包含 MATLAB `importONNXNetwork` 不支持的算子：

| 算子 | 数量 | 影响 |
|------|------|------|
| `Slice` | 84 | 来自索引操作 |
| `Gather` | 32 | 来自索引操作 |
| `Unsqueeze` | 17 | 来自 `.unsqueeze()` |
| `MatMul` (动态) | 48 | 需要一个输入是常量 |
| `Sub` (动态) | 26 | 需要一个输入是常量 |

**总计**: 207 个不支持的算子

---

## 🚀 立即测试

### 方法 1：使用新的测试脚本（推荐）⭐

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_function
```

### 方法 2：手动使用生成的函数

```matlab
% 1. 导入模型并获取参数（只需要一次）
params = importONNXFunction('model_onnx_mode.onnx', 'model_func');
% 这会生成 model_func.m 文件
% params 是 ONNXParameters 对象，包含模型权重

% 2. 准备数据
y = randn(1, 12) + 1i*randn(1, 12);
y_stacked = [real(y), imag(y)];
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% 3. 推理（必须传递 params）
[h_normalized, ~] = model_func(y_normalized, params, ...
                               'InputDataPermutation', 'none', ...
                               'OutputDataPermutation', 'none');

% 4. 处理输出
h_stacked = h_normalized * y_energy;
% 可能需要 permute 调整维度顺序
```

---

## 📊 期望结果

如果 `test_onnx_function` 成功，你会看到：

```
========================================
ONNX Function Test (importONNXFunction)
========================================

Step 1: Importing ONNX as function...
  ✓ Model imported as function!

Step 3: Running inference...
  ✓ Inference successful! (XX.XX ms)

Step 7: Verifying reconstruction...
  ✓ EXCELLENT reconstruction!

========================================
Summary
========================================
Import method:    importONNXFunction
Reconstruction:   ✓ EXCELLENT (X.XX%, XX.XX dB)
========================================

✓ Test PASSED!
```

---

## 🎯 下一步

### 如果 `test_onnx_function` 成功 ✅

**太好了！** 这意味着：
- ✅ 模型可以在 MATLAB 中使用
- ✅ 使用 `importONNXFunction` 方法
- ✅ 生成的 `model_func.m` 可以集成到你的应用

**后续步骤**：
1. 性能测试和优化
2. 集成到生产环境
3. 用真实数据测试

---

### 如果仍然失败 ❌

可能的问题：
1. **维度不匹配** - 需要调整 `InputDataPermutation` 和 `OutputDataPermutation`
2. **MATLAB 版本太旧** - `importONNXFunction` 在 R2020b+ 才有，R2022a+ 更好
3. **模型仍有兼容性问题** - 需要进一步改造

**进一步改造选项**：
1. 完全移除索引操作（避免 `Slice`/`Gather`）
2. 避免所有 `.unsqueeze()`（避免 `Unsqueeze`）
3. 确保所有 `MatMul` 和 `Sub` 的一个输入是常量

---

## 🔍 为什么 `importONNXNetwork` 失败？

MATLAB 的 `importONNXNetwork` 将 ONNX 模型转换为 MATLAB 的 `DAGNetwork` 或 `dlnetwork`，但只支持有限的算子集合。

我们的模型包含太多不支持的算子：
- **索引操作** → `Slice`/`Gather`
- **动态形状操作** → `Unsqueeze`
- **动态矩阵运算** → 需要常量输入

### 为什么 `importONNXFunction` 可能成功？

`importONNXFunction` 生成 MATLAB 代码实现 ONNX 算子，不依赖预定义的层，因此支持更多算子。

---

## 📝 重要说明

### `importONNXFunction` 的限制

虽然 `importONNXFunction` 支持更多算子，但：
- ⚠️ 性能可能不如 `importONNXNetwork`
- ⚠️ 生成的代码可能较长
- ⚠️ 不支持 GPU 加速（在某些 MATLAB 版本）

### 我们的选择

1. **优先使用** `importONNXFunction` - 当前唯一可行的方法
2. **如果需要更好性能** - 考虑进一步改造模型以支持 `importONNXNetwork`
3. **替代方案** - 考虑 OpenVINO（Intel）或 TensorRT（NVIDIA）

---

## 🚀 立即行动

**运行新的测试脚本**：

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_function
```

**然后告诉我结果**：
1. 是否成功导入？
2. 推理是否成功？
3. 重建误差是多少？
4. 遇到什么错误？

让我们一起解决问题！💪
