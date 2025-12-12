# 📁 目录结构快速参考

## 🎯 自动时间戳（避免冲突）

### ✅ 每次训练自动创建唯一目录

```bash
python train.py --model_config separator1_default --training_config default
```

**自动创建**：
```
./experiments_refactored/20251212_103045_separator1_default_default/
```

**格式**：`{timestamp}_{model_config}_{training_config}/`

---

## 📂 目录结构

```
./experiments_refactored/
│
├── 20251212_103045_separator1_default_default/  ← 实验1（带时间戳）
│   ├── separator1_hd64_stages2_depth3/          ← 模型1
│   │   ├── model.pth
│   │   ├── config.yaml
│   │   └── training_history.png
│   ├── separator1_hd128_stages3_depth4/         ← 模型2
│   │   └── model.pth
│   ├── evaluation_results/                      ← 评估
│   │   ├── evaluation_results.json
│   │   └── evaluation_results.npy
│   ├── plots/                                   ← 图表
│   │   └── nmse_vs_snr_combined.png
│   └── TRAINING_REPORT.md                       ← 报告
│
└── 20251212_105123_separator1_small_quick/      ← 实验2（不同时间）
    └── ...
```

---

## 🔄 并发训练（无冲突）

```bash
# 用户A（10:30:45）
python train.py --model_config separator1_default --device cuda:0

# 用户B（10:51:23，同时）
python train.py --model_config separator1_default --device cuda:1

# ✅ 结果：两个独立目录
# - 20251212_103045_separator1_default_default/
# - 20251212_105123_separator1_default_default/
```

---

## 💡 优势

- ✅ **自动避免冲突**：时间戳保证唯一性
- ✅ **可追溯**：时间戳记录训练时间
- ✅ **并发安全**：多个训练互不干扰
- ✅ **便于管理**：按时间排序，易于对比

---

详细说明见 [`EXPERIMENT_DIRECTORY_STRUCTURE.md`](EXPERIMENT_DIRECTORY_STRUCTURE.md)
