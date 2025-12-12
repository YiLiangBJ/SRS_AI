# 🚀 快速开始：Train → Eval → Plot

## 一键命令

```bash
# 最简单：训练 + 评估 + 绘图
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

---

## 常用命令

### 1. 只训练
```bash
python train.py --model_config separator1_default --training_config default --device cuda
```

### 2. 训练 + 评估
```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train
```

### 3. 训练 + 评估 + 绘图 ⭐
```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --plot_after_eval
```

### 4. 自定义评估参数
```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda \
    --eval_after_train \
    --eval_snr_range "20:-2:0" \
    --eval_tdl "A-30,C-300" \
    --eval_num_batches 200 \
    --plot_after_eval
```

---

## 参数说明

### 训练参数
- `--model_config`: 模型配置名称
- `--training_config`: 训练配置名称  
- `--device`: 设备 (auto/cpu/cuda)
- `--save_dir`: 保存目录

### 评估参数 ⭐ NEW
- `--eval_after_train`: 训练后自动评估
- `--eval_snr_range`: SNR范围 (默认 "30:-3:0")
- `--eval_tdl`: TDL配置 (默认 "A-30,B-100,C-300")
- `--eval_num_batches`: 批次数 (默认 100)
- `--eval_batch_size`: 批大小 (默认 2048)
- `--plot_after_eval`: 评估后自动绘图

---

## 输出目录结构

```
./experiments_refactored/
└── separator1_default_xxx/
    ├── model.pth                    # 训练好的模型
    ├── training_history.png         # 训练曲线
    ├── TRAINING_REPORT.md          # 训练报告
    ├── evaluation_results/          # 评估结果 ✅
    │   ├── evaluation_results.json
    │   └── evaluation_results.npy
    └── plots/                       # 图表 ✅
        ├── nmse_vs_snr_TDL_A_30.png
        ├── nmse_vs_snr_TDL_B_100.png
        ├── nmse_vs_snr_TDL_C_300.png
        └── nmse_vs_snr_combined.png
```

---

## 详细文档

查看 [`COMPLETE_PIPELINE.md`](COMPLETE_PIPELINE.md) 获取完整说明。
