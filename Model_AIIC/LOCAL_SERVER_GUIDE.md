# 本地测试与服务器运行指南

## ✅ 修改说明

**改进**: 将 `--num_samples` 改为 `--num_batches`，更加直观！

```
总样本数 = num_batches × batch_size
```

---

## 🖥️ 本地测试（Windows）

### 1. 修正你的命令

**❌ 你的原命令（有错误）**:
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir .Model_AIIC/test_func \     # ❌ 路径错误（少了斜杠）
  --num_samples 20 \                    # ❌ 参数已改名
  --batch_size 200 \
  --output ./test_func_results
```

**✅ 修正后的命令**:
```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./Model_AIIC/test_func \    # ✅ 路径正确
  --num_batches 10 \                    # ✅ 新参数
  --batch_size 200 \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --output ./test_func_results
```

**总样本数**: 10 batches × 200 samples = 2000 per SNR point

### 2. 快速测试（推荐）

先用少量数据测试是否正常：

```bash
# 只测试 1 个 TDL，3 个 SNR 点，2 个 batch
python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_tensorboard \
  --tdl "A-30" \
  --snr_range "30,15,0" \
  --num_batches 2 \
  --batch_size 50 \
  --output ./quick_test

# 总样本数: 2 × 50 = 100 per SNR point
```

### 3. 完整评估

测试通过后，运行完整评估：

```bash
python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./test_func_results

# 总样本数: 10 × 200 = 2000 per SNR point
```

---

## 🖥️ 服务器运行（Linux）

### 1. 基础命令

```bash
cd /path/to/SRS_AI

python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./test_func_results
```

### 2. 后台运行（推荐）

```bash
nohup python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./test_func_results \
  > evaluate.log 2>&1 &

# 查看进度
tail -f evaluate.log

# 查看任务
jobs
ps aux | grep evaluate_models
```

### 3. 使用 screen（推荐）

```bash
# 创建 session
screen -S eval

# 运行评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./test_func_results

# 断开 (Ctrl+A+D)
# 重新连接
screen -r eval
```

---

## 📊 参数说明

### 新参数: num_batches

| 参数 | 说明 | 示例 |
|------|------|------|
| `--num_batches` | 批次数 | 10 |
| `--batch_size` | 每批样本数 | 200 |
| **总样本数** | `num_batches × batch_size` | 10 × 200 = 2000 |

### 为什么改成 num_batches？

**更直观**:
```bash
# ✅ 清晰：10 个 batch，每个 200 样本
--num_batches 10 --batch_size 200  # 总共 2000 样本

# ❌ 不够直观：需要心算
--num_samples 2000 --batch_size 200  # 多少个 batch？10 个
```

**与训练一致**:
```bash
# 训练时
python test_separator.py --batches 1000 --batch_size 2048

# 评估时（现在一致了）
python evaluate_models.py --num_batches 10 --batch_size 200
```

---

## 🎯 典型配置

### 快速测试（~1分钟）
```bash
--num_batches 2 \
--batch_size 50 \
--snr_range "30,15,0" \
--tdl "A-30"

# 总样本: 2 × 50 = 100 per point
```

### 标准评估（~15分钟）
```bash
--num_batches 10 \
--batch_size 200 \
--snr_range "30:-3:0" \
--tdl "A-30,B-100,C-300"

# 总样本: 10 × 200 = 2000 per point
```

### 高精度评估（~1小时）
```bash
--num_batches 50 \
--batch_size 200 \
--snr_range "30:-3:0" \
--tdl "A-30,B-100,C-300"

# 总样本: 50 × 200 = 10000 per point
```

---

## 🔧 故障排查

### 问题 1: 路径错误
```
FileNotFoundError: [WinError 3] The system cannot find the path
```

**原因**: 路径写错了
```bash
--exp_dir .Model_AIIC/test_func  # ❌ 错误
--exp_dir ./Model_AIIC/test_func # ✅ 正确
```

### 问题 2: 找不到模型
```
ValueError: 在 ... 中没有找到任何训练好的模型
```

**检查**:
```bash
# 列出目录内容
ls -la ./test_func/

# 确认有 model.pth
ls ./test_func/*/model.pth
```

### 问题 3: 参数错误
```
error: unrecognized arguments: --num_samples
```

**解决**: 使用新参数 `--num_batches`
```bash
# ❌ 旧参数（已移除）
--num_samples 2000

# ✅ 新参数
--num_batches 10 --batch_size 200
```

---

## 📈 性能优化

### 调整批大小

```bash
# 小批（内存友好）
--num_batches 20 --batch_size 100  # 总样本 2000

# 大批（更快）
--num_batches 10 --batch_size 200  # 总样本 2000

# 超大批（最快，需要大内存）
--num_batches 4 --batch_size 500   # 总样本 2000
```

### 减少 SNR 点

```bash
# 密集采样（11个点）
--snr_range "30:-3:0"  # [30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0]

# 稀疏采样（6个点）
--snr_range "30:-6:0"  # [30, 24, 18, 12, 6, 0]

# 关键点（4个点）
--snr_range "30,20,10,0"  # [30, 20, 10, 0]
```

---

## 📋 完整示例

### 本地 Windows

```bash
# 1. 快速测试
python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_tensorboard \
  --tdl "A-30" \
  --snr_range "30,15,0" \
  --num_batches 2 \
  --batch_size 50 \
  --output ./quick_test

# 2. 完整评估
python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./results

# 3. 绘图
python Model_AIIC/plot_results.py --input ./results
```

### 服务器 Linux

```bash
# 后台运行
nohup python Model_AIIC/evaluate_models.py \
  --exp_dir ./test_func \
  --tdl "A-30,B-100,C-300" \
  --snr_range "30:-3:0" \
  --num_batches 10 \
  --batch_size 200 \
  --output ./results \
  > eval.log 2>&1 &

# 监控
tail -f eval.log

# 完成后绘图
python Model_AIIC/plot_results.py --input ./results
```

---

## 📚 相关文档

- `EVALUATION_GUIDE.md` - 完整使用指南
- `EVALUATION_QUICKREF.md` - 快速参考

---

**最后更新**: 2025-12-02  
**版本**: v1.1 (改进参数命名)
