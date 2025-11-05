# 🔄 循环展开显示说明

## 🎯 新功能：自动展开循环

现在工具会**自动展开循环**，显示每次迭代的真实参数量！

## ✨ 改进对比

### 之前 ❌

```
┌─ Loop: for i in range(self.depth)
│    (注意：每次迭代参数量可能不同，以下显示首次迭代)
│    (循环总参数: 4.7K params)
│    1. x = self.enc_blocks[i](...)  # ComplexResidualBlock
│         Params: 624 params  ← 只显示i=0
│    2. x = self.down_samples[i](...)
│         Params: 272 params  ← 只显示i=0
└─
```

**问题**：看不出每次迭代的具体参数量

### 现在 ✅

```
┌─ Loop: for i in range(self.depth)
│    (展开显示每次迭代的实际参数量)
│
│  ─── 迭代 i=0 ───
│    1. x = self.enc_blocks[0](...)  # ComplexResidualBlock
│         624 params                  ← 清楚标注 i=0
│         → skips.append(x)
│    2. x = self.down_samples[0](...)  # ComplexConv1d
│         272 params
│
│  ─── 迭代 i=1 ───
│    3. x = self.enc_blocks[1](...)  # ComplexResidualBlock
│         2.8K params                 ← 清楚标注 i=1，参数量不同！
│         → skips.append(x)
│    4. x = self.down_samples[1](...)  # ComplexConv1d
│         1.1K params                 ← 1.1K ≠ 272!
└─
```

**优势**：
- ✅ 清楚显示每次迭代
- ✅ 真实参数量一目了然
- ✅ 可以直接累加验证总数
- ✅ 便于理解通道数变化

## 📊 完整示例：ComplexResidualUNet (depth=2)

```
【数据流路径 Data Flow】
  说明：forward()的实际执行顺序（自动从源码提取）

  ┌─ Loop: for i in range(self.depth)
  │    (展开显示每次迭代的实际参数量)
  │
  │  ─── 迭代 i=0 ───  ← 第1次循环
  │    1. x = self.enc_blocks[0](...)  # ComplexResidualBlock
  │         624 params
  │         → skips.append(x)
  │    2. x = self.down_samples[0](...)  # ComplexConv1d
  │         272 params
  │
  │  ─── 迭代 i=1 ───  ← 第2次循环
  │    3. x = self.enc_blocks[1](...)  # ComplexResidualBlock
  │         2.8K params   ← 参数量增加了！
  │         → skips.append(x)
  │    4. x = self.down_samples[1](...)  # ComplexConv1d
  │         1.1K params   ← 参数量增加了！
  └─
  
  5. x = self.bottleneck(...)  # ComplexResidualBlock
       Params: 10.8K params
  
  ┌─ Loop: for i in range(self.depth)
  │    (展开显示每次迭代的实际参数量)
  │
  │  ─── 迭代 i=0 ───
  │    6. x = self.up_samples[0](...)  # ComplexConvTranspose1d
  │         2.1K params
  │    7. x = self.dec_blocks[0](...)  # ComplexResidualBlock
  │         5.9K params
  │
  │  ─── 迭代 i=1 ───
  │    8. x = self.up_samples[1](...)  # ComplexConvTranspose1d
  │         528 params    ← 参数量减少了！
  │    9. x = self.dec_blocks[1](...)  # ComplexResidualBlock
  │         1.5K params   ← 参数量减少了！
  └─
  
  10. residual = self.final_conv(...)  # ComplexConv1d
       Params: 18 params
```

## 🔍 参数量验证

现在可以直接从展开的循环累加验证：

```
步骤1:  624 params    (enc_blocks[0])
步骤2:  272 params    (down_samples[0])
步骤3: 2,800 params   (enc_blocks[1])
步骤4: 1,100 params   (down_samples[1])
步骤5: 10,800 params  (bottleneck)
步骤6: 2,100 params   (up_samples[0])
步骤7: 5,900 params   (dec_blocks[0])
步骤8:  528 params    (up_samples[1])
步骤9: 1,500 params   (dec_blocks[1])
步骤10:  18 params    (final_conv)
────────────────────────────────
总计: ≈25,642 params

实际: 25,554 params
差异: 88 (显示时K级四舍五入导致)
```

## 💡 为什么参数量变化？

### 编码器：通道数递增

```
迭代 i=0:
  enc_blocks[0]:   2 → 8 channels  = 624 params
  down_samples[0]: 8 → 8           = 272 params

迭代 i=1:
  enc_blocks[1]:   8 → 16 channels = 2,784 params  ← 4.5倍！
  down_samples[1]: 16 → 16         = 1,056 params  ← 3.9倍！
```

**原因**：参数量 ∝ `in_channels × out_channels`
- i=0: 2 × 8 = 16
- i=1: 8 × 16 = 128  ← 8倍！

### 解码器：通道数递减

```
迭代 i=0:
  up_samples[0]:  32 → 16 channels = 2,080 params
  dec_blocks[0]:  32 → 16          = 5,856 params

迭代 i=1:
  up_samples[1]:  16 → 8 channels  = 528 params    ← 减少了
  dec_blocks[1]:  16 → 8           = 1,520 params  ← 减少了
```

## 🎨 不同depth的效果

### depth=3 (会展开3次)

```
┌─ Loop: for i in range(self.depth)
│    (展开显示每次迭代的实际参数量)
│
│  ─── 迭代 i=0 ───
│    1. x = self.enc_blocks[0](...)
│         624 params (2→8 ch)
│    2. x = self.down_samples[0](...)
│         272 params
│
│  ─── 迭代 i=1 ───
│    3. x = self.enc_blocks[1](...)
│         2.8K params (8→16 ch)
│    4. x = self.down_samples[1](...)
│         1.1K params
│
│  ─── 迭代 i=2 ───
│    5. x = self.enc_blocks[2](...)
│         10.7K params (16→32 ch)  ← 持续增长
│    6. x = self.down_samples[2](...)
│         4.1K params
└─
```

### depth > 10 (自动折叠)

如果循环次数太多（>10），工具会自动切换回简略显示模式，避免输出过长。

## 🔧 技术实现

### 自动识别循环次数

```python
# 从 range(self.depth) 中提取
if 'self.' in loop_range:
    attr_name = loop_range.replace('self.', '')
    if hasattr(module, attr_name):
        loop_count = getattr(module, attr_name)  # 获取实际值
```

### 展开条件

```python
if loop_count is not None and loop_count > 0 and loop_count <= 10:
    # 展开显示
    for iteration in range(loop_count):
        # 获取 module_list[iteration] 的实际参数
        target_module = module_list_obj[iteration]
        params = sum(p.numel() for p in target_module.parameters())
```

### 每次迭代显示

```python
print(f"  ─── 迭代 i={iteration} ───")
print(f"  {step_num}. x = self.{module_list}[{iteration}](...)")
print(f"       {params} params")
```

## 📈 优势总结

| 特性 | 之前 | 现在 |
|------|------|------|
| 显示方式 | 只显示首次迭代 | 展开所有迭代 |
| 参数可见性 | 需要推测 | 直接显示 |
| 验证便利性 | 难以验证 | 可直接累加 |
| 理解难度 | 需要深入思考 | 一目了然 |
| 模块索引 | 使用变量i | 显示实际数字 |

## 🎯 适用场景

### 适合展开
- ✅ depth ≤ 10
- ✅ 参数量变化大的模型
- ✅ 需要详细分析的场景

### 自动折叠
- ⚠️ depth > 10
- ⚠️ 循环次数未知
- ⚠️ 避免输出过长

## 📚 使用方法

### 运行分析

```bash
python AnalyzeModelStructure.py
```

### 查看报告

```bash
# 完整版
cat model_structure_analysis.txt

# 概览版
cat model_structure_summary.txt
```

### 自定义模型

```python
from AnalyzeModelStructure import analyze_model_structure

model = YourModel(depth=3)  # depth会被自动展开
analyze_model_structure(model, "YourModel")
```

## ✨ 总结

**改进前**：只看到循环的抽象描述，参数量需要推测  
**改进后**：每次迭代完全展开，参数量清晰可见

现在你可以：
1. ✅ 直接看到每个模块的真实参数量
2. ✅ 清楚知道这是第几次迭代（i=0, i=1...）
3. ✅ 轻松验证总参数量
4. ✅ 理解通道数如何变化

这让模型结构分析更加直观和准确！🎊
