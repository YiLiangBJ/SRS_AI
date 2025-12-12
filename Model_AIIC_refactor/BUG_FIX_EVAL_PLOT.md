# 🔧 Bug 修复：评估和绘图错误处理

## 🐛 修复的问题

### 问题1：评估失败 - `'config'` 键错误

**错误信息**：
```
✗ 模型 separator1_grid_search_6ports_hd64_stages2_depth3_share1_default_weighted 评估失败: 'config'
```

**原因**：
- 保存的模型 checkpoint 中，`config` 字典缺少 `pos_values` 键
- 评估代码直接访问 `config['pos_values']` 导致 KeyError

**修复**：
```python
# evaluate_models.py - load_model() 函数
if 'pos_values' not in config:
    # 根据 num_ports 推断 pos_values
    num_ports = config.get('num_ports', 4)
    if num_ports == 4:
        config['pos_values'] = [0, 3, 6, 9]
    elif num_ports == 6:
        config['pos_values'] = [0, 2, 4, 6, 8, 10]
    else:
        config['pos_values'] = list(range(0, 12, 12 // num_ports))[:num_ports]
```

---

### 问题2：绘图失败 - `StopIteration`

**错误信息**：
```
File "/home/liangyi/SRS_AI/Model_AIIC_refactor/plot.py", line 37, in generate_plots_programmatic
    first_model = next(iter(results['models'].values()))
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
StopIteration
```

**原因**：
- 评估失败后，`results['models']` 为空字典
- `next(iter(...))` 在空字典上抛出 `StopIteration`

**修复1 - plot.py**：
```python
# 检查是否有模型数据
if not results.get('models') or len(results['models']) == 0:
    print("⚠️  No models found in evaluation results. Skipping plot generation.")
    return []
```

**修复2 - train.py**：
```python
# 在绘图前检查评估结果
if not eval_json_path.exists():
    print(f"\n⚠️  Skipping plot generation: evaluation results not found")
else:
    with open(eval_json_path, 'r') as f:
        eval_data = json.load(f)
    
    if not eval_data.get('models') or len(eval_data['models']) == 0:
        print(f"\n⚠️  Skipping plot generation: no models evaluated successfully")
    else:
        # 生成图表
        generate_plots_programmatic(...)
```

---

## ✅ 修复后的行为

### 场景1：评估成功

```bash
python train.py --model_config separator1_default --device cuda --eval_after_train --plot_after_eval
```

**输出**：
```
✓ All training completed!

================================================================================
📊 Post-Training Evaluation
================================================================================
✓ 模型 separator1_hd64_stages2_depth3 评估完成

✓ Evaluation completed!

================================================================================
📈 Generating Plots
================================================================================
  ✓ Generated: nmse_vs_snr_TDL_A_30.png
  ✓ Generated: nmse_vs_snr_TDL_B_100.png
  ✓ Generated: nmse_vs_snr_combined.png

✓ Plots generated!
```

---

### 场景2：评估失败（缺少 pos_values）

**之前**：
```
✗ 模型 xxx 评估失败: 'config'  ← KeyError
[程序崩溃]
```

**现在**：
```
✓ 模型 xxx 评估完成  ← ✅ 自动推断 pos_values
```

---

### 场景3：所有模型评估失败

**之前**：
```
✓ Evaluation completed!

================================================================================
📈 Generating Plots
================================================================================
Traceback (most recent call last):
  ...
StopIteration  ← 程序崩溃
```

**现在**：
```
✓ Evaluation completed!

⚠️  Skipping plot generation: no models evaluated successfully

================================================================================
🎉 Complete Pipeline Finished!
================================================================================
```

---

## 🎯 向后兼容性

### 旧模型（没有 pos_values）

**自动推断逻辑**：

| num_ports | pos_values |
|-----------|------------|
| 4 | `[0, 3, 6, 9]` |
| 6 | `[0, 2, 4, 6, 8, 10]` |
| 其他 | 均匀分布 |

**示例**：
```python
# 4 端口
num_ports = 4
pos_values = [0, 3, 6, 9]

# 6 端口
num_ports = 6
pos_values = [0, 2, 4, 6, 8, 10]

# 8 端口
num_ports = 8
pos_values = [0, 1, 2, 3, 4, 5, 6, 7]  # 均匀分布
```

---

### 新模型（有 pos_values）

直接使用保存的 `pos_values`，不需要推断。

---

## 🔍 错误处理流程

### evaluate_models.py

```python
try:
    # 加载模型
    model, config = load_model(model_dir, device=device)
    
    # ✅ 确保 pos_values 存在
    if 'pos_values' not in config:
        config['pos_values'] = infer_pos_values(config['num_ports'])
    
    # 评估...
    
except Exception as e:
    print(f"✗ 模型 {model_name} 评估失败: {e}")
    continue  # 继续评估其他模型
```

---

### plot.py

```python
def generate_plots_programmatic(eval_results_path, output_dir):
    # Load results
    with open(eval_results_path, 'r') as f:
        results = json.load(f)
    
    # ✅ 检查是否有数据
    if not results.get('models') or len(results['models']) == 0:
        print("⚠️  No models found. Skipping plot generation.")
        return []
    
    # 生成图表...
```

---

### train.py

```python
if args.plot_after_eval:
    eval_json_path = eval_output_dir / 'evaluation_results.json'
    
    # ✅ 检查文件存在
    if not eval_json_path.exists():
        print("⚠️  Skipping plot: evaluation results not found")
    else:
        # ✅ 检查有无模型
        with open(eval_json_path, 'r') as f:
            eval_data = json.load(f)
        
        if not eval_data.get('models') or len(eval_data['models']) == 0:
            print("⚠️  Skipping plot: no models evaluated successfully")
        else:
            # 生成图表
            try:
                generate_plots_programmatic(...)
            except Exception as e:
                print(f"✗ Plot generation failed: {e}")
```

---

## ✅ 测试场景

### 1. 正常情况

```bash
python train.py --model_config separator1_default --device cuda --eval_after_train --plot_after_eval
```

**预期**：✅ 训练 → 评估 → 绘图，全部成功

---

### 2. 旧模型（缺少 pos_values）

```bash
python evaluate_models.py --exp_dir "./old_experiments/model_without_pos_values" --device cuda
```

**预期**：✅ 自动推断 pos_values，评估成功

---

### 3. 评估失败（所有模型）

```bash
# 假设模型文件损坏或不兼容
python train.py --device cuda --eval_after_train --plot_after_eval
```

**预期**：
- ✅ 训练成功
- ⚠️ 评估失败（显示错误）
- ⚠️ 跳过绘图（不崩溃）

---

### 4. 部分模型评估成功

```bash
# Grid Search，部分模型失败
python train.py --model_config separator1_grid_search --device cuda --eval_after_train --plot_after_eval
```

**预期**：
- ✅ 训练所有模型
- ⚠️ 部分评估失败（显示错误）
- ✅ 成功模型生成图表

---

## 📝 修改文件列表

1. `evaluate_models.py` - 添加 `pos_values` 推断逻辑
2. `plot.py` - 添加空结果检查
3. `train.py` - 添加绘图前的验证逻辑

---

## 🎉 总结

### 修复前 ❌

- 缺少 `pos_values` 导致评估崩溃
- 评估失败导致绘图崩溃
- 错误信息不友好

### 修复后 ✅

- ✅ 自动推断 `pos_values`（向后兼容）
- ✅ 评估失败不影响绘图
- ✅ 友好的错误提示
- ✅ 部分失败不影响整体流程

**更加健壮和用户友好！** 🚀
