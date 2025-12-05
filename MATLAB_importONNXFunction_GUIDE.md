# importONNXFunction 快速参考

## ✅ 正确用法

### 导入模型（只需一次）

```matlab
% 方法 1：导入并获取参数
params = importONNXFunction('model_onnx_mode.onnx', 'model_func');
% ✓ 生成 model_func.m 文件
% ✓ 返回 ONNXParameters 对象

% 方法 2：捕获错误
try
    params = importONNXFunction('model_onnx_mode.onnx', 'model_func');
    fprintf('✓ Import successful!\n');
    fprintf('  Parameters class: %s\n', class(params));
catch ME
    fprintf('✗ Import failed: %s\n', ME.message);
end
```

### 使用生成的函数

```matlab
% 准备输入
y = randn(1, 24);  % (1, 24) 格式

% 推理（必须传递 params）
[output, ~] = model_func(y, params, ...
                         'InputDataPermutation', 'none', ...
                         'OutputDataPermutation', 'none');
```

---

## ❌ 常见错误

### 错误 1: Missing params argument

```matlab
% ❌ 错误写法（忘记传 params）
[output, ~] = model_func(input, ...
                         'InputDataPermutation', 'none');
```

**错误信息**：`The value of 'params' is invalid. It must satisfy the function: @(x)isa(x,'ONNXParameters').`

**原因**：生成的函数需要 `params` (ONNXParameters 对象)。

**正确写法**：
```matlab
% ✅ 正确（先获取 params）
params = importONNXFunction('model.onnx', 'func');
[output, ~] = func(input, params, ...
                   'InputDataPermutation', 'none');
```

---

### 错误 2: Undefined function 'model_func'

```matlab
% ❌ 错误：没有先导入
[output, ~] = model_func(input, params);
```

**原因**：必须先运行 `importONNXFunction`。

**正确写法**：
```matlab
% ✅ 正确
params = importONNXFunction('model.onnx', 'model_func');  % 先导入
[output, ~] = model_func(input, params, ...              % 再使用
                         'InputDataPermutation', 'none');
```

---

### 错误 3: 忘记保存 params

```matlab
% ❌ 错误：没有保存 params
importONNXFunction('model.onnx', 'func');
[output, ~] = func(input, params);  % params 未定义！
```

**错误信息**：`Undefined variable 'params'.`

**正确写法**：
```matlab
% ✅ 正确（保存 params）
params = importONNXFunction('model.onnx', 'func');
[output, ~] = func(input, params, ...
                   'InputDataPermutation', 'none');
```

---

### 错误 4: 维度不匹配

```matlab
% ❌ 可能错误：自动维度变换
[output, ~] = model_func(input);  % 使用默认 'auto'
```

**问题**：MATLAB 可能自动改变维度顺序。

**正确写法**：
```matlab
% ✅ 明确指定 'none'
[output, ~] = model_func(input, ...
                         'InputDataPermutation', 'none', ...
                         'OutputDataPermutation', 'none');
```

---

## 📋 完整工作流程

### 步骤 1: 导入（仅一次）

```matlab
cd('c:/GitRepo/SRS_AI')
params = importONNXFunction('model_onnx_mode.onnx', 'model_func');
```

**检查**：
- ✅ 确认 `model_func.m` 文件已生成
- ✅ `params` 是 `ONNXParameters` 类型

---

### 步骤 2: 使用

```matlab
% 2.1 准备数据
L = 12;
y_complex = randn(1, L) + 1i*randn(1, L);
y_stacked = [real(y_complex), imag(y_complex)];  % (1, 24)

% 2.2 能量归一化（必须！）
y_energy = sqrt(mean(abs(y_complex).^2));
y_normalized = y_stacked / y_energy;

% 2.3 推理（必须传递 params）
[h_normalized, ~] = model_func(y_normalized, params, ...
                               'InputDataPermutation', 'none', ...
                               'OutputDataPermutation', 'none');

% 2.4 恢复能量
h_stacked = h_normalized * y_energy;

% 2.5 处理输出维度
% 输出可能是 (24, 4, 1) 或其他格式
% 需要转换为 (1, 4, 24)
output_size = size(h_stacked);
fprintf('Raw output: %s\n', mat2str(output_size));

% 自动重排（如果需要）
if length(output_size) == 3
    % 找到各个维度
    [~, L2_dim] = max(output_size == 24);  % 特征维度
    [~, P_dim] = max(output_size == 4);    % 端口维度
    B_dim = setdiff([1 2 3], [L2_dim, P_dim]);  % batch 维度
    
    % 重排为 (B, P, L*2)
    h_stacked = permute(h_stacked, [B_dim, P_dim, L2_dim]);
end

fprintf('Final output: %s\n', mat2str(size(h_stacked)));

% 2.6 转换为复数
P = size(h_stacked, 2);
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);
h_complex = complex(h_real, h_imag);  % (1, P, L)
```

---

### 步骤 3: 验证

```matlab
% 重建信号
y_recon = squeeze(sum(h_complex, 2));

% 计算误差
recon_error = norm(y_complex - y_recon) / norm(y_complex);
fprintf('Reconstruction error: %.2f%%\n', recon_error * 100);

if recon_error < 0.05
    fprintf('✓ GOOD quality!\n');
else
    fprintf('⚠ Check normalization and dimensions\n');
end
```

---

## 🎯 使用测试脚本

不想手动写代码？使用我们的测试脚本：

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_function
```

这个脚本会自动：
- ✅ 导入模型
- ✅ 生成测试数据
- ✅ 处理归一化
- ✅ 运行推理
- ✅ 处理维度
- ✅ 验证结果
- ✅ 显示报告

---

## 📚 MATLAB 文档

想了解更多？查看 MATLAB 文档：

```matlab
help importONNXFunction
doc importONNXFunction
```

或在线：
https://www.mathworks.com/help/deeplearning/ref/importonnxfunction.html

---

## ⚡ 性能提示

### 批量处理

```matlab
% 处理多个样本
for i = 1:num_samples
    [h_norm, ~] = model_func(y_batch(i, :), params, ...
                             'InputDataPermutation', 'none', ...
                             'OutputDataPermutation', 'none');
    h_batch(:, :, :, i) = h_norm;
end
```

### 预编译（加速）

```matlab
% 第一次调用会较慢（JIT 编译）
[h1, ~] = model_func(y1, params, ...
                     'InputDataPermutation', 'none', ...
                     'OutputDataPermutation', 'none');

% 后续调用会快很多
tic;
for i = 1:1000
    [h, ~] = model_func(y, params, ...
                        'InputDataPermutation', 'none', ...
                        'OutputDataPermutation', 'none');
end
avg_time = toc / 1000;
fprintf('Average time: %.2f ms\n', avg_time * 1000);
```

---

**准备好了吗？重新运行测试！** 🚀

```matlab
cd('c:/GitRepo/SRS_AI')
test_onnx_function
```
