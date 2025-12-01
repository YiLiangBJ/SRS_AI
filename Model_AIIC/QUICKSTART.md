# 快速参考卡片

## 🎯 一行命令开始训练

```bash
python Model_AIIC/test_separator.py --batches 1000 --batch_size 2048 --save_dir "./my_exp"
```

## 📊 启动 TensorBoard

```bash
tensorboard --logdir ./my_exp
# 打开浏览器: http://localhost:6006
```

## 🔥 常用命令

### 基础训练
```bash
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages 3 \
  --save_dir "./training"
```

### 超参数搜索
```bash
python Model_AIIC/test_separator.py \
  --batches 2000 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./search"
```

### 早停训练
```bash
python Model_AIIC/test_separator.py \
  --batches 10000 \
  --early_stop 0.01 \
  --val_interval 100 \
  --patience 5 \
  --save_dir "./early_stop"
```

### CPU 控制
```bash
# 50% 核心
python Model_AIIC/test_separator.py --cpu_ratio 0.5 --save_dir "./exp"
```

## 📂 输出文件

```
save_dir/
└── stages=3_share=False/
    ├── tensorboard/          # TensorBoard 日志
    ├── model.pth             # PyTorch 权重
    ├── model.pt              # TorchScript（推荐）
    ├── metrics.json          # 指标
    ├── training_report.md    # 报告
    └── train_losses.npy      # 曲线
```

## 🎨 TensorBoard 面板

| 面板 | 内容 |
|------|------|
| **Scalars** | Loss 曲线、NMSE、吞吐量 |
| **HParams** | 超参数对比表 |
| **Graphs** | 模型结构 |

## 🚀 部署

### Python
```python
import torch
model = torch.jit.load('model.pt')
model.eval()

y = torch.randn(1, 12, dtype=torch.complex64)
h_pred = model(y)  # [1, 4, 12]
```

### C++
```cpp
#include <torch/script.h>
auto model = torch::jit::load("model.pt");
```

## 📚 完整文档

详见: [`Model_AIIC/README.md`](README.md)
