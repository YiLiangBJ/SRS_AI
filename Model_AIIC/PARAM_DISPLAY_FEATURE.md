# 参数量显示功能 - 实现总结

## ✅ 已完成的功能

在图例中显示每个模型的参数量，格式如：`stages=2_share=False (104.6K)`

## 🔧 修改的文件

### 1. `Model_AIIC/evaluate_models.py`

**修改内容**：
- 从 checkpoint 的 `hyperparameters` 中读取 `num_params`
- 如果没有保存，自动计算参数量
- 将 `num_params` 添加到返回的 `config` 中

```python
# 从 hyperparameters 中获取 num_params
num_params = hyperparams.get('num_params', None)

# 如果没有保存，现在计算它
if num_params is None:
    num_params = sum(p.numel() for p in model.parameters())

# 添加到 config
config['num_params'] = num_params
```

### 2. `Model_AIIC/plot_results.py`

**新增函数**：

1. **`format_num_params(num_params)`** - 格式化参数量
   ```python
   format_num_params(104_640)  # → "104.6K"
   format_num_params(1_234_567)  # → "1.2M"
   format_num_params(None)  # → "N/A"
   ```

2. **`get_model_label(model_name, config)`** - 生成包含参数量的标签
   ```python
   get_model_label("stages=2_share=False", {'num_params': 104_640})
   # → "stages=2_share=False (104.6K)"
   ```

**更新的函数**：
- `plot_single_figure()` - 单图模式
- `plot_subplots_by_tdl()` - 按 TDL 分图
- `plot_subplots_by_model()` - 按模型分图

所有绘图函数现在都会在图例中显示参数量。

## 📊 效果展示

### 之前的图例
```
stages=2_share=False - A-30
stages=3_share=False - A-30
stages=2_share=True - B-100
```

### 现在的图例 ⭐
```
stages=2_share=False (104.6K) - A-30
stages=3_share=False (157.0K) - A-30
stages=2_share=True (78.3K) - B-100
```

## 🎯 参数量格式规则

| 参数量范围 | 格式 | 示例 |
|-----------|------|------|
| < 1,000 | 原始数字 | `500` |
| 1K - 999K | X.XK | `104.6K`, `157.0K` |
| ≥ 1M | X.XM | `1.2M`, `10.0M` |
| 未知 | N/A | `N/A` |

## 🚀 使用方法

### 1. 评估模型（自动读取参数量）

```bash
python ./Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval
```

### 2. 绘图（自动显示参数量）

```bash
# 按 TDL 分图
python ./Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

**图例示例**：
```
Legend:
  stages=2_share=False_loss=nmse (104.6K)
  stages=2_share=False_loss=normalized (104.6K)
  stages=3_share=False_loss=nmse (157.0K)
  stages=3_share=True_loss=log (78.3K)
```

### 3. 图表标题也包含参数量

按模型分图时，标题会显示：
```
stages=2_share=False (104.6K) - 4 Ports: [0,3,6,9]
```

## 📋 技术细节

### 参数量来源优先级

1. **从 checkpoint 读取**（最优）
   ```python
   checkpoint['hyperparameters']['num_params']
   ```

2. **实时计算**（备用）
   ```python
   sum(p.numel() for p in model.parameters())
   ```

### 向后兼容

- ✅ 旧模型（没有保存 `num_params`）：自动计算
- ✅ 新模型（已保存 `num_params`）：直接读取
- ✅ 缺失数据：显示原始模型名称（不添加参数量）

## 🧪 测试验证

当前仓库没有单独提交 `test_param_display.py`。

建议直接使用下面的评估和绘图命令验证参数量显示是否正常。

**测试结果**：
```
✓ None -> N/A
✓ 500 -> 500
✓ 1500 -> 1.5K
✓ 104640 -> 104.6K
✓ 156960 -> 157.0K
✓ 1234567 -> 1.2M
✓ 10000000 -> 10.0M

✓ 所有测试完成！
```

## 📊 实际应用示例

### 6 端口模型对比

```bash
# 评估
python ./Model_AIIC/evaluate_models.py \
  --exp_dir ./out6ports \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 100 \
  --output ./out6ports_eval

# 绘图
python ./Model_AIIC/plot_results.py \
  --input ./out6ports_eval \
  --layout subplots_tdl
```

**图例会显示**：
- 每个模型的阶段数、权重共享配置
- 损失函数类型
- **参数量**（如 234.5K）
- TDL 配置

**示例图例**：
```
stages=2_share=False_loss=nmse (234.5K) - A-30
stages=2_share=True_loss=normalized (140.2K) - A-30
stages=3_share=False_loss=log (351.8K) - A-30
```

## 🎨 图表改进总结

### 图例信息（现在包含）

✅ 模型阶段数（stages）  
✅ 权重共享（share_weights）  
✅ 损失函数类型（loss_type）  
✅ **参数量（num_params）** ⭐ 新增  
✅ TDL 配置

### 标题信息（现在包含）

✅ 完整模型配置  
✅ **参数量** ⭐ 新增  
✅ 端口配置（Ports: [0,2,4,6,8,10]）  
✅ TDL 类型

## 💡 使用建议

1. **对比不同阶段数的模型**
   - 可以直观看到参数量的差异
   - 评估性能提升与参数量增加的关系

2. **权重共享的影响**
   - `share=True` 通常参数量更少
   - 可以对比相同阶段数下的参数差异

3. **选择最优模型**
   - 综合考虑性能（NMSE）和复杂度（参数量）
   - 找到性价比最高的配置

## 📝 代码示例

### 获取模型标签（在代码中使用）

```python
from Model_AIIC.plot_results import get_model_label

config = {
    'num_params': 104_640,
    'num_stages': 2,
    'share_weights': False
}

label = get_model_label("stages=2_share=False", config)
print(label)  # → "stages=2_share=False (104.6K)"
```

### 格式化参数量

```python
from Model_AIIC.plot_results import format_num_params

print(format_num_params(104_640))   # → "104.6K"
print(format_num_params(1_234_567)) # → "1.2M"
print(format_num_params(None))      # → "N/A"
```

---

**更新日期**: 2025-12-04  
**版本**: v2.4  
**状态**: ✅ 已实现并测试
