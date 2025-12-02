# 模型性能评估与绘图指南

## 📋 工作流程

```
1. 训练模型 (test_separator.py)
      ↓
2. 评估性能 (evaluate_models.py) → 生成 JSON/NPY 结果
      ↓
3. 绘制曲线 (plot_results.py) → 生成 PNG/PDF 图像
```

---

## 1️⃣ 评估模型性能

### 基本用法

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --output "./evaluation_results"
```

这会：
- 扫描 SNR: 30, 27, 24, ..., 3, 0 dB（共11个点）
- 评估所有训练好的模型
- 在所有 TDL 配置下测试（默认: A-30, B-100, C-300）
- 保存结果到 `./evaluation_results/`

### 选择特定模型

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --models "stages=2_share=False,stages=3_share=False" \
  --output "./results"
```

### 自定义 SNR 范围

```bash
# SNR: 25, 20, 15, 10, 5, 0 dB
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "25:-5:0" \
  --output "./results"

# SNR: 30, 20, 10, 0 dB
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "30:-10:0" \
  --output "./results"

# 特定 SNR 点
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "0,10,20,30" \
  --output "./results"
```

### 自定义 TDL 配置

```bash
# 只评估 TDL-A 和 TDL-B
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --tdl "A-30,B-100" \
  --output "./results"

# 不同的 delay spread
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --tdl "A-30,A-100,A-300" \
  --output "./results"
```

### 调整评估样本数

```bash
# 更多样本（更准确但更慢）
# 总样本数 = 50 batches × 100 samples = 5000
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --num_batches 50 \
  --batch_size 100 \
  --output "./results"

# 更少样本（更快但不太准确）
# 总样本数 = 5 batches × 100 samples = 500
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --num_batches 5 \
  --batch_size 100 \
  --output "./results"
```

### 完整示例

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir "./grid_search" \
  --models "stages=2_share=False,stages=3_share=False,stages=3_share=True" \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output "./evaluation_results"

# 总样本数 = 10 batches × 200 samples = 2000 per SNR point
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--exp_dir` | 训练实验目录（包含 model.pth） | **必需** |
| `--models` | 要评估的模型（逗号分隔）| 所有模型 |
| `--tdl` | TDL 配置（逗号分隔） | A-30,B-100,C-300 |
| `--snr_range` | SNR 范围 (start:step:end) | 30:-3:0 |
| `--num_batches` | 每个 SNR 点的批次数 | 10 |
| `--batch_size` | 评估批大小 | 100 |
| `--output` | 结果保存目录 | ./evaluation_results |

**注意**: 总样本数 = `num_batches` × `batch_size`

---

## 2️⃣ 绘制性能曲线

### 基本用法

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results"
```

这会生成单个图像，包含所有模型和 TDL 的曲线。

### 选择特定模型

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --models "stages=2_share=False,stages=3_share=False"
```

### 选择特定 TDL

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --tdl "A-30,B-100"
```

### 不同的布局风格

#### 风格 1: 单图（默认）
所有曲线在一个图中

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --layout single
```

#### 风格 2: 按 TDL 分子图
每个 TDL 一个子图，比较不同模型

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --layout subplots_tdl
```

#### 风格 3: 按模型分子图
每个模型一个子图，比较不同 TDL

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --layout subplots_model
```

### 只保存不显示

```bash
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --no_show
```

### 完整示例

```bash
# 单图：比较 stage=2 和 stage=3，只看 TDL-A 和 TDL-B
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --models "stages=2_share=False,stages=3_share=False" \
  --tdl "A-30,B-100" \
  --layout single

# 按 TDL 分图：查看所有模型在不同 TDL 下的表现
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --layout subplots_tdl

# 按模型分图：查看每个模型在不同 TDL 下的表现
python Model_AIIC/plot_results.py \
  --input "./evaluation_results" \
  --layout subplots_model
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 评估结果目录 | **必需** |
| `--models` | 要绘制的模型（逗号分隔） | 所有模型 |
| `--tdl` | 要绘制的 TDL（逗号分隔） | 所有 TDL |
| `--layout` | 布局风格 | single |
| `--no_show` | 不显示图像（只保存） | False |

**布局选项**:
- `single`: 所有曲线在一个图中
- `subplots_tdl`: 每个 TDL 一个子图
- `subplots_model`: 每个模型一个子图

---

## 3️⃣ 输出文件

### 评估输出

```
evaluation_results/
├── evaluation_results.json    # JSON 格式（可读）
└── evaluation_results.npy     # NumPy 格式（快速加载）
```

**JSON 结构**:
```json
{
  "timestamp": "2025-12-02T...",
  "config": {
    "snr_list": [30, 27, 24, ...],
    "tdl_list": ["A-30", "B-100", "C-300"],
    "num_samples": 1000
  },
  "models": {
    "stages=2_share=False": {
      "config": {...},
      "tdl_results": {
        "A-30": {
          "snr": [30, 27, ...],
          "nmse": [...],
          "nmse_db": [...],
          "port_nmse": [[...], ...],
          "port_nmse_db": [[...], ...]
        },
        ...
      }
    },
    ...
  }
}
```

### 绘图输出

```
evaluation_results/
├── nmse_vs_snr_single.png      # 单图 PNG
├── nmse_vs_snr_single.pdf      # 单图 PDF
├── nmse_vs_snr_subplots.png    # 子图 PNG
├── nmse_vs_snr_subplots.pdf    # 子图 PDF
├── nmse_vs_snr_by_model.png    # 按模型分图 PNG
└── nmse_vs_snr_by_model.pdf    # 按模型分图 PDF
```

---

## 4️⃣ 完整工作流示例

### 场景 1: 快速评估

```bash
# 1. 训练模型
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --stages "2,3" \
  --save_dir "./quick_exp"

# 2. 评估（快速）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./quick_exp" \
  --num_samples 500 \
  --output "./quick_results"

# 3. 绘图
python Model_AIIC/plot_results.py \
  --input "./quick_results"
```

### 场景 2: 全面评估

```bash
# 1. 训练多个配置
python Model_AIIC/test_separator.py \
  --batches 2000 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./full_exp"

# 2. 评估（详细）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./full_exp" \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_samples 2000 \
  --output "./full_results"

# 3. 绘制多种图
python Model_AIIC/plot_results.py \
  --input "./full_results" \
  --layout single

python Model_AIIC/plot_results.py \
  --input "./full_results" \
  --layout subplots_tdl

python Model_AIIC/plot_results.py \
  --input "./full_results" \
  --layout subplots_model
```

### 场景 3: 对比特定配置

```bash
# 评估
python Model_AIIC/evaluate_models.py \
  --exp_dir "./grid_search" \
  --models "stages=2_share=False,stages=3_share=False" \
  --tdl "A-30,B-100" \
  --snr_range "30:-3:0" \
  --output "./comparison"

# 绘图
python Model_AIIC/plot_results.py \
  --input "./comparison" \
  --models "stages=2_share=False,stages=3_share=False" \
  --tdl "A-30,B-100" \
  --layout subplots_tdl
```

---

## 5️⃣ 结果解读

### NMSE vs SNR 曲线

**X 轴**: SNR (dB) - 信噪比  
**Y 轴**: NMSE (dB) - 归一化均方误差

**典型趋势**:
- SNR 越高 → NMSE 越低（性能越好）
- NMSE 越低越好（负值表示很好）
- 曲线斜率反映模型对 SNR 的敏感度

**对比要点**:
1. **不同模型**: 哪个模型 NMSE 更低？
2. **不同 TDL**: 哪个信道条件下性能更好？
3. **不同 SNR**: 低 SNR 下哪个模型更鲁棒？

### 性能指标

**查看摘要**:
```bash
python Model_AIIC/plot_results.py --input ./results
```

输出示例:
```
模型: stages=3_share=False
  TDL-A-30:
    SNR 范围: 30.0 ~ 0.0 dB
    最佳性能: -18.5 dB @ SNR=30.0 dB
    最差性能: -2.3 dB @ SNR=0.0 dB
    性能提升: 16.2 dB
```

---

## 6️⃣ 故障排查

### 问题 1: 找不到模型

```
FileNotFoundError: Model not found: ...
```

**解决**:
- 确认 `--exp_dir` 路径正确
- 确认目录下有 `model.pth` 文件
- 使用 `--models` 指定正确的模型名称

### 问题 2: 评估太慢

**优化**:
- 减少 `--num_samples`（如 500）
- 增大 `--batch_size`（如 500）
- 减少 SNR 点数（如 `--snr_range "30:-5:0"`）

### 问题 3: 内存不足

**解决**:
- 减小 `--batch_size`（如 50）
- 减少 `--num_samples`

### 问题 4: 图像不显示

**解决**:
- Windows: 确保有图形界面
- 远程服务器: 使用 `--no_show`，只保存图像
- 查看保存的 PNG/PDF 文件

---

## 7️⃣ 高级技巧

### 批量评估

```bash
# 创建评估脚本
cat > evaluate_all.sh << 'EOF'
#!/bin/bash

# 评估多个实验目录
for exp in exp1 exp2 exp3; do
    python Model_AIIC/evaluate_models.py \
        --exp_dir "./${exp}" \
        --output "./${exp}_results"
    
    python Model_AIIC/plot_results.py \
        --input "./${exp}_results" \
        --layout subplots_tdl
done
EOF

chmod +x evaluate_all.sh
./evaluate_all.sh
```

### 自定义绘图

修改 `plot_results.py` 中的：
- 颜色: `colors = [...]`
- 线型: `linestyles = [...]`
- 标记: `markers = [...]`
- 字体大小: `fontsize=...`

---

## 📚 参考

- **训练指南**: `Model_AIIC/README.md`
- **快速开始**: `Model_AIIC/QUICKSTART.md`
- **TensorBoard**: `Model_AIIC/README.md` - TensorBoard 章节

---

**最后更新**: 2025-12-02  
**版本**: v1.0
