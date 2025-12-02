# 模型评估与绘图系统 - 完成总结

## ✅ 已创建的文件

### 1. 核心脚本

| 文件 | 功能 | 代码行数 |
|------|------|---------|
| **evaluate_models.py** | 模型性能评估 | ~280 行 |
| **plot_results.py** | 性能曲线绘制 | ~420 行 |

### 2. 文档

| 文件 | 内容 |
|------|------|
| **EVALUATION_GUIDE.md** | 完整使用指南（7000+ 字） |
| **EVALUATION_QUICKREF.md** | 快速参考卡片 |

### 3. 示例脚本

| 文件 | 用途 |
|------|------|
| **demo_workflow.sh** | Linux/Mac 演示脚本 |
| **demo_workflow.bat** | Windows 演示脚本 |

---

## 🎯 核心功能

### evaluate_models.py

**功能清单**:
- ✅ 扫描 SNR 范围（默认 30:-3:0 dB）
- ✅ 支持多个 TDL 配置（A/B/C）
- ✅ 选择性加载训练好的模型
- ✅ 批量评估 NMSE 性能
- ✅ 支持每端口独立评估
- ✅ 保存 JSON 和 NPY 格式结果
- ✅ 进度条显示
- ✅ 详细日志输出

**关键特性**:
```python
# 1. 灵活的 SNR 配置
--snr_range "30:-3:0"   # 30, 27, 24, ..., 0 dB
--snr_range "30:-5:0"   # 30, 25, 20, 15, 10, 5, 0 dB
--snr_range "0,10,20,30"  # 特定点

# 2. 模型选择
--models "stages=2_share=False,stages=3_share=False"

# 3. 信道配置
--tdl "A-30,B-100,C-300"

# 4. 性能调优
--num_samples 1000  # 样本数
--batch_size 100    # 批大小
```

### plot_results.py

**功能清单**:
- ✅ 读取评估结果（JSON/NPY）
- ✅ 三种布局风格
  - 单图：所有曲线
  - 按 TDL 分图：比较模型
  - 按模型分图：比较 TDL
- ✅ 选择性绘制（过滤模型/TDL）
- ✅ 自动生成 PNG 和 PDF
- ✅ 打印性能摘要
- ✅ 中文支持
- ✅ 专业配色和标记

**关键特性**:
```python
# 1. 布局选择
--layout single          # 单图
--layout subplots_tdl    # 按 TDL 分
--layout subplots_model  # 按模型分

# 2. 过滤
--models "stages=2_share=False"
--tdl "A-30,B-100"

# 3. 输出控制
--no_show  # 只保存不显示
```

---

## 📊 使用流程

### 完整工作流

```bash
# 步骤 1: 训练模型
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --stages "2,3" \
  --save_dir "./experiments"

# 步骤 2: 评估性能
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --output "./results"

# 步骤 3: 绘制曲线
python Model_AIIC/plot_results.py \
  --input "./results"
```

### 快速演示

```bash
# Linux/Mac
bash Model_AIIC/demo_workflow.sh

# Windows
Model_AIIC\demo_workflow.bat
```

---

## 📁 输出结构

### 评估结果

```
results/
├── evaluation_results.json    # 主结果文件
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
          "snr": [...],
          "nmse": [...],
          "nmse_db": [...],
          "port_nmse": [...],
          "port_nmse_db": [...]
        }
      }
    }
  }
}
```

### 图像文件

```
results/
├── nmse_vs_snr_single.png      # 单图 PNG
├── nmse_vs_snr_single.pdf      # 单图 PDF
├── nmse_vs_snr_subplots.png    # 按 TDL 分图
├── nmse_vs_snr_subplots.pdf    # 按 TDL 分图 PDF
├── nmse_vs_snr_by_model.png    # 按模型分图
└── nmse_vs_snr_by_model.pdf    # 按模型分图 PDF
```

---

## 🎨 绘图效果

### 1. 单图 (single)
- 所有模型 × 所有 TDL 的曲线在一个图中
- 适合快速对比
- 颜色区分模型，线型区分 TDL

### 2. 按 TDL 分图 (subplots_tdl)
- 每个 TDL 一个子图
- 在同一信道条件下比较不同模型
- 横向排列

### 3. 按模型分图 (subplots_model)
- 每个模型一个子图
- 查看单个模型在不同信道下的表现
- 网格排列（最多 3 列）

---

## 🔧 高级用法

### 1. 选择性评估

```bash
# 只评估 stage=2 和 stage=3
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --models "stages=2_share=False,stages=3_share=False" \
  --output "./comparison"

# 只看 TDL-A 和 TDL-B
python Model_AIIC/plot_results.py \
  --input "./comparison" \
  --tdl "A-30,B-100"
```

### 2. 不同 SNR 范围

```bash
# 宽范围，粗间隔（快速）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "30:-6:0" \
  --output "./quick_results"

# 细间隔（精确）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "30:-1:0" \
  --output "./detailed_results"

# 特定点
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --snr_range "0,5,10,15,20,25,30" \
  --output "./specific_results"
```

### 3. 性能优化

```bash
# 快速评估（少样本）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --num_samples 500 \
  --batch_size 100 \
  --output "./quick"

# 精确评估（多样本）
python Model_AIIC/evaluate_models.py \
  --exp_dir "./experiments" \
  --num_samples 5000 \
  --batch_size 500 \
  --output "./accurate"
```

### 4. 批量处理

```bash
# 评估多个实验目录
for exp in exp1 exp2 exp3; do
  python Model_AIIC/evaluate_models.py \
    --exp_dir "./${exp}" \
    --output "./${exp}_results"
  
  python Model_AIIC/plot_results.py \
    --input "./${exp}_results" \
    --layout subplots_tdl
done
```

---

## 📈 性能基准

### 评估速度

| 配置 | 时间 | 说明 |
|------|------|------|
| 1 模型, 1 TDL, 11 SNR 点, 1000 样本 | ~2-3 分钟 | 标准 |
| 3 模型, 3 TDL, 11 SNR 点, 1000 样本 | ~15-20 分钟 | 中等规模 |
| 6 模型, 3 TDL, 11 SNR 点, 2000 样本 | ~60 分钟 | 全面评估 |

**优化建议**:
- 增大 `--batch_size`（如 200-500）
- 减少 `--num_samples`（如 500）
- 使用粗 SNR 间隔（如 `-6` 而非 `-3`）

---

## 🎓 使用场景

### 场景 1: 快速验证
```bash
# 训练完成后快速看看效果
python Model_AIIC/evaluate_models.py \
  --exp_dir ./exp \
  --num_samples 500 \
  --snr_range "30:-6:0" \
  --output ./quick_check

python Model_AIIC/plot_results.py --input ./quick_check
```

### 场景 2: 论文图表
```bash
# 精确评估，生成出版质量图表
python Model_AIIC/evaluate_models.py \
  --exp_dir ./final_exp \
  --num_samples 5000 \
  --snr_range "30:-3:0" \
  --output ./paper_results

python Model_AIIC/plot_results.py \
  --input ./paper_results \
  --layout subplots_tdl
```

### 场景 3: 模型对比
```bash
# 对比两个最佳模型
python Model_AIIC/evaluate_models.py \
  --exp_dir ./grid_search \
  --models "stages=3_share=False,stages=4_share=False" \
  --output ./best_comparison

python Model_AIIC/plot_results.py \
  --input ./best_comparison \
  --layout single
```

---

## 🔍 故障排查

### 问题 1: 找不到模型
```
FileNotFoundError: Model not found
```
**解决**: 检查 `--exp_dir` 路径，确认有 `model.pth`

### 问题 2: 评估太慢
**解决**: 减少 `--num_samples` 或增大 `--batch_size`

### 问题 3: 内存不足
**解决**: 减小 `--batch_size`

### 问题 4: 图像乱码（Windows）
**解决**: 系统安装 SimHei 字体或修改 `plot_results.py` 中的字体设置

---

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| `EVALUATION_GUIDE.md` | 完整使用指南（推荐） |
| `EVALUATION_QUICKREF.md` | 快速参考卡片 |
| `README.md` | 项目总览 |
| `QUICKSTART.md` | 快速开始 |

---

## ✨ 特色功能

1. **灵活的 SNR 配置**: 支持范围、列表、自定义间隔
2. **多种 TDL 信道**: A/B/C 及自定义 delay spread
3. **选择性评估**: 只评估需要的模型
4. **多种绘图布局**: 单图、按 TDL 分、按模型分
5. **双格式输出**: JSON（可读）+ NPY（快速）
6. **双格式图像**: PNG（Web）+ PDF（出版）
7. **进度显示**: tqdm 进度条
8. **性能摘要**: 自动打印最佳/最差性能
9. **错误处理**: 友好的错误提示
10. **批量处理**: 支持脚本化批量评估

---

## 🚀 下一步

1. **运行演示**:
   ```bash
   bash Model_AIIC/demo_workflow.sh  # Linux/Mac
   Model_AIIC\demo_workflow.bat      # Windows
   ```

2. **查看文档**:
   ```bash
   cat Model_AIIC/EVALUATION_GUIDE.md
   ```

3. **开始评估**:
   ```bash
   python Model_AIIC/evaluate_models.py --exp_dir ./your_exp --output ./results
   python Model_AIIC/plot_results.py --input ./results
   ```

---

**创建日期**: 2025-12-02  
**版本**: v1.0  
**状态**: ✅ 完成
