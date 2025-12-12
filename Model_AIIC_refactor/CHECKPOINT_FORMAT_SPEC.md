# 🏗️ 统一的 Checkpoint 格式规范

## 📋 设计原则

1. **单一真相来源**：所有信息统一保存在 checkpoint 中
2. **明确的键名约定**：避免混淆（config vs model_config）
3. **完整性**：保存所有需要的信息用于评估和复现
4. **可读性**：结构清晰，易于理解和维护

---

## 📦 标准 Checkpoint 格式

### 核心结构

```python
checkpoint = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. 模型相关（必需）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'model_state_dict': model.state_dict(),          # 模型权重
    'model_info': model.get_model_info(),            # 模型架构信息
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. 配置信息（必需）- 用于重建模型和评估
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'config': {
        # 模型架构参数
        'model_type': 'separator1',
        'hidden_dim': 64,
        'num_stages': 2,
        'mlp_depth': 3,
        'share_weights': False,
        'activation_type': 'relu',
        
        # 数据参数
        'seq_len': 12,
        'num_ports': 4,
        'pos_values': [0, 3, 6, 9],
        
        # 元信息
        'num_params': 156032,
    },
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. 训练配置（必需）- 用于了解训练过程
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'training_config': {
        'loss_type': 'nmse',
        'learning_rate': 0.01,
        'num_batches': 10000,
        'batch_size': 4096,
        'snr_config': {'type': 'range', 'min': 0, 'max': 30},
        'tdl_config': 'A-30',
    },
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. 训练状态（可选）- 用于恢复训练
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': [0.5, 0.3, 0.2, ...],
    'val_losses': [0.6, 0.4, 0.25, ...],
    'loss_type': 'nmse',
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. 元信息（可选但推荐）- 用于追溯和管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'metadata': {
        'model_config_name': 'separator1_default',
        'config_instance_name': 'separator1_hd64_stages2_depth3',
        'training_config_name': 'default',
        'training_duration': 1234.5,
        'timestamp': '20251212_103045',
    },
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 6. 评估结果（可选）- 训练时的快速评估
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    'eval_results': {
        'nmse': 0.001234,
        'nmse_db': -29.08,
        'per_port_nmse_db': [-30.1, -28.5, -29.2, -28.8],
    }
}
```

---

## 🔄 数据流

### Train → Save

```python
# train.py
checkpoint = {
    'model_state_dict': model.state_dict(),
    'model_info': model.get_model_info(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses,
    'val_losses': val_losses,
    'loss_type': loss_type,
    
    # ✅ 统一键名：config（包含所有模型配置）
    'config': {
        'model_type': config['model_type'],
        'hidden_dim': config['hidden_dim'],
        'num_stages': config['num_stages'],
        'mlp_depth': config['mlp_depth'],
        'share_weights': config['share_weights'],
        'activation_type': config.get('activation_type', 'relu'),
        'seq_len': config.get('seq_len', 12),
        'num_ports': len(config.get('pos_values', [0, 3, 6, 9])),
        'pos_values': config.get('pos_values', [0, 3, 6, 9]),
        'num_params': num_params,
    },
    
    # ✅ 统一键名：training_config（包含所有训练配置）
    'training_config': {
        'loss_type': training_config['loss_type'],
        'learning_rate': training_config['learning_rate'],
        'num_batches': training_config['num_batches'],
        'batch_size': training_config['batch_size'],
        'snr_config': training_config['snr_config'],
        'tdl_config': training_config.get('tdl_config', 'A-30'),
    },
    
    # ✅ 元信息
    'metadata': {
        'model_config_name': model_config_name,
        'config_instance_name': config_instance_name,
        'training_config_name': training_config_name,
        'training_duration': training_duration,
        'timestamp': timestamp,
    },
    
    # ✅ 评估结果
    'eval_results': eval_results,
}

torch.save(checkpoint, save_path)
```

---

### Load → Eval

```python
# evaluate_models.py
def load_model(model_dir, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    
    # ✅ 读取 config（标准键名）
    config = checkpoint['config']
    
    # ✅ 使用 config 创建模型
    model = create_model(
        model_type=config['model_type'],
        config=config
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

# ✅ 评估时直接使用 config 中的参数
nmse, nmse_db, port_nmse, port_nmse_db = evaluate_model_at_snr(
    model, snr_db, tdl_config, 
    pos_values=config['pos_values'],  # ✅ 从 config 读取
    num_batches=num_batches,
    batch_size=batch_size,
    device=device
)
```

---

### Load → Plot

```python
# plot.py
with open(eval_results_path, 'r') as f:
    results = json.load(f)

# ✅ 评估结果已经包含所有需要的信息
for model_name, model_data in results['models'].items():
    config = model_data['config']  # ✅ 标准键名
    tdl_results = model_data['tdl_results']
    
    # 绘图...
```

---

## 🔑 关键设计决策

### 1. 统一键名

| 键名 | 用途 | 内容 |
|------|------|------|
| `config` | 模型配置 | 所有重建模型需要的参数 |
| `training_config` | 训练配置 | 所有训练相关的参数 |
| `metadata` | 元信息 | 追溯和管理信息 |
| `eval_results` | 评估结果 | 快速评估的结果 |

**不再使用**：
- ❌ `model_config`（容易混淆）
- ❌ 分散的配置项

---

### 2. 必需字段保证

```python
# config 必需字段
REQUIRED_CONFIG_FIELDS = {
    'model_type',      # 模型类型
    'hidden_dim',      # 隐藏层维度
    'num_stages',      # stage数量
    'mlp_depth',       # MLP深度
    'share_weights',   # 是否共享权重
    'seq_len',         # 序列长度
    'num_ports',       # 端口数量
    'pos_values',      # 端口位置
}

# training_config 必需字段
REQUIRED_TRAINING_FIELDS = {
    'loss_type',       # 损失函数类型
    'learning_rate',   # 学习率
    'num_batches',     # 批次数
    'batch_size',      # 批大小
}
```

---

### 3. 向后兼容策略

**不需要！** 重新训练所有模型，使用新格式。

---

## 📝 实现清单

### 修改文件

1. ✅ `train.py` - 统一保存格式
   - 使用标准键名 `config` 和 `training_config`
   - 确保 `config` 包含所有必需字段
   
2. ✅ `evaluate_models.py` - 统一读取格式
   - 读取 `config` 键
   - 移除 `pos_values` 推断逻辑（不再需要）
   
3. ✅ `plot.py` - 已经正确使用 `config`

4. ✅ `training/trainer.py` - 无需修改
   - `save_checkpoint` 已经支持 `additional_info`

---

## ✅ 验证点

### 训练后检查

```python
import torch

checkpoint = torch.load('model.pth')

# ✅ 检查必需键
assert 'config' in checkpoint
assert 'training_config' in checkpoint
assert 'metadata' in checkpoint

# ✅ 检查 config 必需字段
assert 'model_type' in checkpoint['config']
assert 'pos_values' in checkpoint['config']
assert 'num_ports' in checkpoint['config']

# ✅ 检查一致性
assert len(checkpoint['config']['pos_values']) == checkpoint['config']['num_ports']

print("✅ Checkpoint format valid!")
```

---

### 评估前检查

```python
# evaluate_models.py
config = checkpoint['config']

# ✅ 验证必需字段
required = ['model_type', 'pos_values', 'num_ports', 'seq_len']
for field in required:
    assert field in config, f"Missing required field: {field}"

print("✅ Config valid for evaluation!")
```

---

## 🎯 好处

1. **清晰性** ⭐⭐⭐
   - 键名明确，不会混淆
   - 结构层次清晰

2. **完整性** ⭐⭐⭐
   - 包含所有需要的信息
   - 可以完全复现训练和评估

3. **可维护性** ⭐⭐⭐
   - 统一的格式，易于维护
   - 添加新字段容易

4. **可测试性** ⭐⭐
   - 容易验证格式正确性
   - 容易发现缺失字段

5. **易用性** ⭐⭐⭐
   - 不需要推断和猜测
   - 所有信息都在 checkpoint 中

---

## 🚀 下一步

1. 修改 `train.py` 使用新格式保存
2. 修改 `evaluate_models.py` 使用新格式读取
3. 删除所有推断和兼容性代码
4. 重新训练一个模型验证
5. 完整测试 train → eval → plot 流程

---

**统一、清晰、可维护！** 🎉
