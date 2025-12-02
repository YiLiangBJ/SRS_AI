# 评估与绘图 - 快速参考

## 🎯 一行命令

### 评估所有模型
```bash
python Model_AIIC/evaluate_models.py --exp_dir ./experiments --output ./results
```

### 绘制曲线
```bash
python Model_AIIC/plot_results.py --input ./results
```

---

## 📊 常用命令

### 1. 评估特定模型
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./experiments \
  --models "stages=2_share=False,stages=3_share=False" \
  --output ./results
```

### 2. 自定义 SNR 范围
```bash
# 30, 27, 24, ..., 3, 0 dB
python Model_AIIC/evaluate_models.py \
  --exp_dir ./experiments \
  --snr_range "30:-3:0" \
  --output ./results
```

### 3. 选择 TDL 配置
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./experiments \
  --tdl "A-30,B-100" \
  --output ./results
```

### 4. 快速评估（少样本）
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./experiments \
  --num_samples 500 \
  --batch_size 100 \
  --output ./results
```

---

## 🎨 绘图布局

### 单图（默认）
```bash
python Model_AIIC/plot_results.py \
  --input ./results \
  --layout single
```

### 按 TDL 分图
```bash
python Model_AIIC/plot_results.py \
  --input ./results \
  --layout subplots_tdl
```

### 按模型分图
```bash
python Model_AIIC/plot_results.py \
  --input ./results \
  --layout subplots_model
```

### 过滤绘制
```bash
python Model_AIIC/plot_results.py \
  --input ./results \
  --models "stages=2_share=False" \
  --tdl "A-30,B-100"
```

---

## 📁 输出文件

```
results/
├── evaluation_results.json    # 评估数据
├── evaluation_results.npy     # NumPy 格式
├── nmse_vs_snr_single.png     # 单图
├── nmse_vs_snr_single.pdf     # 单图 PDF
├── nmse_vs_snr_subplots.png   # 子图
└── nmse_vs_snr_subplots.pdf   # 子图 PDF
```

---

## ⚡ 快速演示

```bash
# 运行完整演示（训练-评估-绘图）
# Linux/Mac:
bash Model_AIIC/demo_workflow.sh

# Windows:
Model_AIIC\demo_workflow.bat
```

---

## 🔧 参数速查

### evaluate_models.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp_dir` | - | 实验目录 **（必需）** |
| `--models` | 所有 | 模型列表 |
| `--tdl` | A-30,B-100,C-300 | TDL 配置 |
| `--snr_range` | 30:-3:0 | SNR 范围 |
| `--num_samples` | 1000 | 样本数 |
| `--batch_size` | 100 | 批大小 |
| `--output` | ./evaluation_results | 输出目录 |

### plot_results.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | - | 结果目录 **（必需）** |
| `--models` | 所有 | 模型过滤 |
| `--tdl` | 所有 | TDL 过滤 |
| `--layout` | single | 布局风格 |
| `--no_show` | False | 不显示图像 |

---

## 📚 完整文档

详见: [`Model_AIIC/EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md)
