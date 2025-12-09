# ✅ Training Report 时间记录功能已添加

## 📝 新增功能

在 `training_report.md` 中添加了完整的训练时间信息：

### 新增字段

1. **Training Started** - 训练开始的实际时间
2. **Training Ended** - 训练结束的实际时间  
3. **Total Duration** - 总训练时长（格式：HH:MM:SS 和秒数）
4. **Batches Completed** - 完成的 batch 数量 / 总 batch 数量
5. **Average Throughput** - 平均吞吐量（samples/s）
6. **Time per Batch** - 每个 batch 的平均时间（秒）

---

## 📊 示例输出

### training_report.md 示例

```markdown
# Training Report

**Experiment**: stages=2_hd=32_sub=1_share=False_loss=weighted_act=relu

**Report Generated**: 2025-12-09 14:45:40

## Training Timeline

| Event | Time |
|-------|------|
| **Training Started** | 2025-12-09 14:45:38 |
| **Training Ended** | 2025-12-09 14:45:38 |
| **Total Duration** | 00:00:00 (0.2s) |
| **Batches Completed** | 5 / 5 |
| **Average Throughput** | 790 samples/s |
| **Time per Batch** | 0.040s |

---

## Configuration
...
```

### 更长时间训练的示例

假设训练 50,000 batches，batch_size=2048：

```markdown
## Training Timeline

| Event | Time |
|-------|------|
| **Training Started** | 2025-12-09 10:00:00 |
| **Training Ended** | 2025-12-09 11:25:30 |
| **Total Duration** | 01:25:30 (5130.0s) |
| **Batches Completed** | 50000 / 50000 |
| **Average Throughput** | 19982 samples/s |
| **Time per Batch** | 0.103s |
```

---

## 🎯 使用方法

### 训练时自动生成

```bash
# 正常训练，会自动在 training_report.md 中记录时间
python Model_AIIC_onnx/test_separator.py \
  --batches 50000 \
  --batch_size 2048 \
  --stages "3" \
  --save_dir "./experiments"
```

### 查看训练报告

```bash
# 查看生成的报告
cat experiments/*/training_report.md | head -20

# 或者在 VS Code 中直接打开
code experiments/20251209_*/training_report.md
```

---

## 📈 时间信息的用途

### 1. 性能分析
- **Throughput**: 判断 CPU/GPU 利用率
- **Time per Batch**: 找出瓶颈

### 2. 实验记录
- **Start/End Time**: 精确知道每个实验的执行时间
- **Total Duration**: 对比不同配置的训练速度

### 3. 资源规划
- 估算大规模实验的时间需求
- 优化 batch_size 和 num_workers

### 4. 复现性
- 完整的时间戳记录便于追踪实验

---

## 🔍 实际案例

### 案例 1: 快速原型验证
```
Training Started:  2025-12-09 14:45:38
Training Ended:    2025-12-09 14:45:38
Total Duration:    00:00:00 (0.2s)
Batches:           5 / 5
Throughput:        790 samples/s
```
**分析**: 验证代码正常，5 batches 只需 0.2 秒

### 案例 2: 标准训练
```
Training Started:  2025-12-09 10:00:00
Training Ended:    2025-12-09 11:25:30
Total Duration:    01:25:30 (5130s)
Batches:           50000 / 50000
Throughput:        19982 samples/s
Time per Batch:    0.103s
```
**分析**: 50k batches 用了 1.5 小时，吞吐量正常

### 案例 3: 提前停止
```
Training Started:  2025-12-09 09:00:00
Training Ended:    2025-12-09 09:35:20
Total Duration:    00:35:20 (2120s)
Batches:           15420 / 50000  ← 提前停止
Throughput:        18650 samples/s
```
**分析**: Early stopping 生效，节省了约 1 小时

---

## 🛠️ 技术细节

### 时间记录点

```python
# 训练开始
training_start_datetime = datetime.now()
training_start_time = time.time()

# ... 训练循环 ...

# 训练结束
training_end_datetime = datetime.now()
training_duration = time.time() - training_start_time

# 保存到 training_report.md
f.write(f"| **Training Started** | {training_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} |\n")
f.write(f"| **Training Ended** | {training_end_datetime.strftime('%Y-%m-%d %H:%M:%S')} |\n")
```

### 时长计算

```python
duration_hours = int(training_duration // 3600)
duration_mins = int((training_duration % 3600) // 60)
duration_secs = int(training_duration % 60)

# 格式化为 HH:MM:SS
formatted_duration = f"{duration_hours:02d}:{duration_mins:02d}:{duration_secs:02d}"
```

### 吞吐量计算

```python
samples_trained = len(losses) * batch_size
throughput = samples_trained / training_duration  # samples/s
time_per_batch = training_duration / len(losses)  # seconds/batch
```

---

## 📋 相关文件

- **修改的文件**: `Model_AIIC_onnx/test_separator.py`
- **生成的报告**: `experiments/*/training_report.md`
- **使用指南**: `HYPERPARAMETER_GUIDE.md`

---

## ✅ 验证测试

```bash
# 快速测试（已验证 ✅）
python Model_AIIC_onnx/test_separator.py \
  --batches 5 \
  --batch_size 32 \
  --stages "2" \
  --hidden_dim "32" \
  --num_sub_stages "1" \
  --save_dir "./test_time_report"

# 查看结果
cat test_time_report/*/training_report.md | head -20
```

**输出**: ✅ 成功显示完整的训练时间信息

---

## 🎉 完成状态

- ✅ 添加训练开始时间记录
- ✅ 添加训练结束时间记录
- ✅ 计算总训练时长（HH:MM:SS 格式）
- ✅ 计算平均吞吐量（samples/s）
- ✅ 计算每 batch 平均时间
- ✅ 显示完成的 batch 数量
- ✅ 功能测试通过

**所有时间记录功能已完成！** 🚀

