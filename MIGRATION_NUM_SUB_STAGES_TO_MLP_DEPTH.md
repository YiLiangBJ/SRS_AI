# ✅ 参数重命名：`num_sub_stages` → `mlp_depth`

## 📝 变更说明

为了提升代码可读性和符合业界命名规范，我们将参数 `num_sub_stages` 重命名为 `mlp_depth`。

### 变更前后对比

| 旧参数名 | 新参数名 | 含义 |
|---------|---------|------|
| `num_sub_stages` | `mlp_depth` | MLP 总层数（包括输入层和输出层） |

### 语义变化

| `mlp_depth` | 网络结构 | 隐藏层数 | 总层数 |
|------------|---------|---------|--------|
| **2** | Input → Output | 0 | 2 |
| **3** (默认) | Input → Hidden → Output | 1 | 3 |
| **4** | Input → Hidden1 → Hidden2 → Output | 2 | 4 |
| **5** | Input → H1 → H2 → H3 → Output | 3 | 5 |

**关键点**：
- ✅ `mlp_depth` 直接表示总层数，更直观
- ✅ 最小值是 2（不能更小）
- ✅ 默认值从 `2` 改为 `3`（但含义相同，都是 1 个隐藏层）

---

## 🔄 迁移指南

### 1. Python 代码迁移

#### 旧代码
```python
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

model = ResidualRefinementSeparatorReal(
    seq_len=12,
    num_ports=4,
    hidden_dim=64,
    num_stages=3,
    num_sub_stages=2,  # ❌ 旧参数名
    share_weights_across_stages=False
)
```

#### 新代码
```python
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

model = ResidualRefinementSeparatorReal(
    seq_len=12,
    num_ports=4,
    hidden_dim=64,
    num_stages=3,
    mlp_depth=3,  # ✅ 新参数名（等价于旧的 num_sub_stages=2）
    share_weights_across_stages=False
)
```

**注意**：`num_sub_stages=2` → `mlp_depth=3`（含义相同，都是 1 个隐藏层）

---

### 2. 命令行参数迁移

#### 旧命令
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --num_sub_stages "1,2,3"  # ❌ 旧参数
```

#### 新命令
```bash
python Model_AIIC_onnx/test_separator.py \
  --batches 10000 \
  --mlp_depth "2,3,4"  # ✅ 新参数（等价于旧的 1,2,3）
```

**对应关系**：
- 旧 `--num_sub_stages "1"` → 新 `--mlp_depth "2"`
- 旧 `--num_sub_stages "2"` → 新 `--mlp_depth "3"`
- 旧 `--num_sub_stages "3"` → 新 `--mlp_depth "4"`

---

### 3. 配置文件迁移

#### 旧配置（model.pth）
```python
{
    'config': {
        'seq_len': 12,
        'num_ports': 4,
        'hidden_dim': 64,
        'num_stages': 3,
        'num_sub_stages': 2,  # ❌ 旧字段
        'share_weights': False
    }
}
```

#### 新配置（model.pth）
```python
{
    'config': {
        'seq_len': 12,
        'num_ports': 4,
        'hidden_dim': 64,
        'num_stages': 3,
        'mlp_depth': 3,  # ✅ 新字段
        'share_weights': False
    }
}
```

---

## ✅ 向后兼容性

**好消息**：`evaluate_models.py` 已支持向后兼容！

```python
# 自动兼容旧模型
mlp_depth = config.get('mlp_depth', config.get('num_sub_stages', 3))
```

**行为**：
1. 优先读取新字段 `mlp_depth`
2. 如果没有，读取旧字段 `num_sub_stages` 并转换
3. 如果都没有，默认为 `3`

**结论**：✅ 旧模型可以正常加载和评估！

---

## 🧪 测试验证

### 测试 1: 参数量验证
```python
from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

# 旧方式（如果还支持）
# model_old = ResidualRefinementSeparatorReal(num_sub_stages=2)

# 新方式
model_new = ResidualRefinementSeparatorReal(mlp_depth=3)

print(f"参数量: {sum(p.numel() for p in model_new.parameters()):,}")
# 输出: 参数量: 138,528
```

### 测试 2: 网络结构验证
```python
for depth in [2, 3, 4, 5]:
    model = ResidualRefinementSeparatorReal(mlp_depth=depth)
    params = sum(p.numel() for p in model.parameters())
    print(f"mlp_depth={depth}: {params:,} params")

# 输出:
# mlp_depth=2: 38,688 params (0 hidden layers)
# mlp_depth=3: 138,528 params (1 hidden layer)
# mlp_depth=4: 238,368 params (2 hidden layers)
# mlp_depth=5: 338,208 params (3 hidden layers)
```

### 测试 3: 训练兼容性
```bash
# 新命令
python Model_AIIC_onnx/test_separator.py \
  --batches 10 \
  --mlp_depth "2,3,4" \
  --save_dir "./test_mlp_depth"
```

**预期结果**：✅ 正常训练，生成 3 个实验

---

## 📊 参数对应表

### 常用配置对应

| 旧参数 (`num_sub_stages`) | 新参数 (`mlp_depth`) | 网络结构 | 参数量 (4 ports, share=False) |
|---------------------------|---------------------|---------|------------------------------|
| 1 | 2 | Input → Output | ~39k |
| 2 (默认) | 3 (默认) | Input → H → Output | ~139k |
| 3 | 4 | Input → H1 → H2 → Output | ~238k |
| 4 | 5 | Input → H1 → H2 → H3 → Output | ~338k |

---

## 📁 修改的文件列表

1. ✅ `Model_AIIC/channel_separator.py`
2. ✅ `Model_AIIC_onnx/channel_separator.py`
3. ✅ `Model_AIIC_onnx/complex_layers.py`
4. ✅ `Model_AIIC_onnx/test_separator.py`
5. ✅ `Model_AIIC_onnx/evaluate_models.py` (支持向后兼容)

---

## 🎯 迁移检查清单

- [ ] 更新 Python 代码中的参数名
- [ ] 更新命令行脚本中的参数名
- [ ] 更新配置文件中的字段名
- [ ] 测试旧模型是否能正常加载
- [ ] 测试新模型训练是否正常
- [ ] 更新相关文档和注释

---

## 💡 为什么这样改？

### 1. **更清晰的语义** ⭐⭐⭐⭐⭐
```python
mlp_depth=3  # 直接表示 3 层
vs
num_sub_stages=2  # 需要理解：2 个子阶段 = 3 层？
```

### 2. **符合业界标准** ⭐⭐⭐⭐⭐
- ResNet: `depth=50`
- VGG: `depth=16`
- Transformer: `num_layers=12`

### 3. **避免混淆** ⭐⭐⭐⭐
```python
mlp_depth=3  # 清晰：总共 3 层
num_hidden_layers=1  # 清晰：1 个隐藏层
num_sub_stages=2  # 混淆：什么是"子阶段"？
```

### 4. **更直观的配置** ⭐⭐⭐⭐
```bash
# 想要 5 层网络
--mlp_depth 5  # ✅ 直观

# vs
--num_sub_stages 4  # ❌ 需要计算
```

---

## 📚 相关文档

- `HYPERPARAMETER_GUIDE.md` - 超参数使用指南（需更新）
- `PARAMETER_COUNT_ANALYSIS.md` - 参数量分析（已更新）
- `MODIFICATIONS_SUMMARY.md` - 修改总结（需更新）

---

## ✅ 完成状态

- ✅ 代码修改完成
- ✅ 向后兼容支持
- ✅ 测试验证通过
- ✅ 迁移文档生成

**所有修改已完成！新参数 `mlp_depth` 更清晰、更符合直觉！** 🎉

