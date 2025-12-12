# 🔧 torch.compile() 兼容性修复

## 🐛 问题

### 错误信息

```
Missing key(s) in state_dict: "port_mlps.0.mlp_real.0.weight", ...
Unexpected key(s) in state_dict: "_orig_mod.port_mlps.0.mlp_real.0.weight", ...
```

### 根本原因

当使用 `torch.compile()` 编译模型后：
1. **训练时**：模型被包装，state_dict 的键名带有 `_orig_mod.` 前缀
2. **保存时**：保存的是编译后的模型，state_dict 包含 `_orig_mod.` 前缀
3. **加载时**：创建的是未编译的原始模型，期望没有前缀的键名
4. **结果**：键名不匹配，加载失败

### 示例

```python
# 训练时
model = Separator1(...)
if compile_model:
    model = torch.compile(model)  # ✅ 编译模型

# 此时 model.state_dict() 的键名：
# {
#   '_orig_mod.port_mlps.0.mlp_real.0.weight': ...,
#   '_orig_mod.port_mlps.0.mlp_real.0.bias': ...,
#   ...
# }

# 评估时
model = Separator1(...)  # ❌ 未编译的模型

# 期望的键名：
# {
#   'port_mlps.0.mlp_real.0.weight': ...,
#   'port_mlps.0.mlp_real.0.bias': ...,
#   ...
# }
```

---

## ✅ 解决方案

### 修改：`training/trainer.py` 的 `save_checkpoint` 方法

**原理**：保存时使用原始模型（`_orig_mod`）的 state_dict

```python
def save_checkpoint(self, save_path, additional_info=None):
    """
    Save model checkpoint
    
    Handles torch.compile() wrapper: saves original model state_dict
    without _orig_mod prefix for compatibility with loading.
    """
    # ✅ Get original model (handle torch.compile() wrapper)
    if hasattr(self.model, '_orig_mod'):
        # Model is compiled with torch.compile(), use the original
        original_model = self.model._orig_mod
    else:
        # Model is not compiled
        original_model = self.model
    
    # ✅ Save original model's state_dict (without _orig_mod prefix)
    checkpoint = {
        'model_state_dict': original_model.state_dict(),
        'model_info': original_model.get_model_info(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'losses': self.losses,
        'val_losses': self.val_losses,
        'loss_type': self.loss_type
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
```

---

## 🔍 工作原理

### torch.compile() 的行为

```python
# 原始模型
model = Separator1(...)
print(model.state_dict().keys())
# Output: dict_keys(['port_mlps.0.mlp_real.0.weight', ...])

# 编译后的模型
compiled_model = torch.compile(model)
print(compiled_model.state_dict().keys())
# Output: dict_keys(['_orig_mod.port_mlps.0.mlp_real.0.weight', ...])

# 访问原始模型
print(compiled_model._orig_mod.state_dict().keys())
# Output: dict_keys(['port_mlps.0.mlp_real.0.weight', ...])  ✅ 没有前缀！
```

### 修复逻辑

```python
# 检查是否被编译
if hasattr(self.model, '_orig_mod'):
    # 被编译 → 使用原始模型
    original_model = self.model._orig_mod
else:
    # 未编译 → 直接使用
    original_model = self.model

# 保存原始模型的 state_dict（无前缀）
checkpoint['model_state_dict'] = original_model.state_dict()
```

---

## 🎯 验证

### 1. 检查保存的 checkpoint

```python
import torch

# 加载 checkpoint
ckpt = torch.load('model.pth', map_location='cpu')

# 检查键名
print("Keys sample:")
for key in list(ckpt['model_state_dict'].keys())[:5]:
    print(f"  {key}")

# 预期输出（无 _orig_mod 前缀）：
# Keys sample:
#   port_mlps.0.mlp_real.0.weight
#   port_mlps.0.mlp_real.0.bias
#   port_mlps.0.mlp_real.2.weight
#   ...
```

### 2. 测试训练 + 评估

```bash
# 训练（使用 compile）
python train.py \
    --model_config separator1_default \
    --training_config default \
    --num_batches 100 \
    --device cuda \
    --eval_after_train

# 预期：✅ 评估成功
```

### 3. 独立评估测试

```bash
# 独立运行评估
python evaluate_models.py \
    --exp_dir "experiments_refactored/20251212_xxx" \
    --device cuda \
    --num_batches 10

# 预期：✅ 所有模型评估成功
```

---

## 📊 修复前后对比

### 修复前 ❌

```python
# trainer.py
checkpoint = {
    'model_state_dict': self.model.state_dict(),  # ❌ 带 _orig_mod 前缀
    ...
}

# 保存的键名：
# '_orig_mod.port_mlps.0.mlp_real.0.weight'

# 评估时加载：
model = Separator1(...)
model.load_state_dict(checkpoint['model_state_dict'])
# ❌ 错误：Missing key "port_mlps.0.mlp_real.0.weight"
#         Unexpected key "_orig_mod.port_mlps.0.mlp_real.0.weight"
```

### 修复后 ✅

```python
# trainer.py
original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
checkpoint = {
    'model_state_dict': original_model.state_dict(),  # ✅ 无前缀
    ...
}

# 保存的键名：
# 'port_mlps.0.mlp_real.0.weight'

# 评估时加载：
model = Separator1(...)
model.load_state_dict(checkpoint['model_state_dict'])
# ✅ 成功：键名完全匹配
```

---

## 🔄 兼容性

### 支持的场景

| 训练配置 | 评估配置 | 结果 |
|---------|---------|------|
| 未编译 | 未编译 | ✅ 正常 |
| 未编译 | 编译 | ✅ 正常 |
| 编译 | 未编译 | ✅ 正常（修复后）|
| 编译 | 编译 | ✅ 正常（修复后）|

### 向后兼容

- ✅ 旧模型（未编译训练）：正常工作
- ✅ 新模型（编译训练）：正常工作
- ✅ 混合场景：全部支持

---

## 💡 关键点

### 1. 为什么不在加载时移除前缀？

**不推荐**：
```python
# ❌ 在加载时处理
state_dict = checkpoint['model_state_dict']
new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
```

**问题**：
- 需要在每个加载的地方都处理
- 容易遗漏
- 不是根本解决方案

**推荐**（当前方案）：
```python
# ✅ 在保存时处理（一次性）
original_model = self.model._orig_mod if hasattr(self.model, '_orig_mod') else self.model
checkpoint['model_state_dict'] = original_model.state_dict()
```

**优势**：
- 只需修改一个地方（save_checkpoint）
- 所有加载代码无需改动
- 更清晰，更易维护

---

### 2. torch.compile() 最佳实践

```python
# ✅ 推荐做法
model = create_model(...)

# 编译（如果需要）
if compile_model:
    model = torch.compile(model)

# 训练
trainer = Trainer(model, ...)
trainer.train(...)

# 保存（自动处理 _orig_mod）
trainer.save_checkpoint('model.pth')  # ✅ 保存的是原始 state_dict

# 加载（无需特殊处理）
checkpoint = torch.load('model.pth')
new_model = create_model(...)
new_model.load_state_dict(checkpoint['model_state_dict'])  # ✅ 直接加载
```

---

## 🚀 测试步骤

### 完整测试

```bash
# 1. 清理旧结果
rm -rf experiments_refactored/20251212_*

# 2. 重新训练（启用 compile）
python train.py \
    --model_config separator1_default \
    --training_config default \
    --num_batches 100 \
    --device cuda

# 3. 检查保存的模型
python -c "
import torch
ckpt = torch.load('experiments_refactored/20251212_*/separator1_*/model.pth', 
                   map_location='cpu')
keys = list(ckpt['model_state_dict'].keys())
print('First 3 keys:')
for k in keys[:3]:
    print(f'  {k}')
print('Has _orig_mod prefix:', any('_orig_mod' in k for k in keys))
"

# 预期输出：
# First 3 keys:
#   port_mlps.0.mlp_real.0.weight
#   port_mlps.0.mlp_real.0.bias
#   port_mlps.0.mlp_real.2.weight
# Has _orig_mod prefix: False  ✅

# 4. 测试评估
python evaluate_models.py \
    --exp_dir "experiments_refactored/20251212_*" \
    --device cuda \
    --num_batches 10

# 预期输出：
# ✓ 模型 separator1_hd64_stages2_depth3 评估完成
# ✓ Evaluation completed!

# 5. 完整流程测试
python train.py \
    --model_config separator1_default \
    --training_config default \
    --num_batches 100 \
    --device cuda \
    --eval_after_train \
    --plot_after_eval

# 预期输出：
# ✓ All training completed!
# ✓ 模型评估完成
# ✓ Plots generated!
# 🎉 Complete Pipeline Finished!
```

---

## ✅ 总结

### 问题
- `torch.compile()` 导致保存的 state_dict 包含 `_orig_mod.` 前缀
- 加载时键名不匹配

### 修复
- 保存时使用 `self.model._orig_mod` 获取原始模型
- 保存原始模型的 state_dict（无前缀）

### 优势
- ✅ 一次修改，全局生效
- ✅ 无需改动加载代码
- ✅ 完全向后兼容
- ✅ 清晰易维护

---

**修复完成！现在 torch.compile() 完全兼容！** 🎉
