# 📊 TensorBoard 快速参考

## 🚀 快速开始

### 1. 训练（自动启用）
```bash
python train.py --model_config separator1_default --device cuda
```

### 2. 启动 TensorBoard
```bash
tensorboard --logdir experiments_refactored
```

### 3. 打开浏览器
```
http://localhost:6006
```

---

## 📈 记录的指标

| 指标 | 说明 | 预期行为 |
|------|------|---------|
| **Loss/train** | 训练损失 | 持续下降 |
| **NMSE/train** | 归一化均方误差 | 持续下降 |
| **NMSE_dB/train** | NMSE (dB) | 下降至负值（越低越好）|
| **Loss/validation** | 验证损失 | 跟随训练损失下降 |
| **SNR/train** | 信噪比 | 在设定范围内 |
| **Learning_Rate** | 学习率 | 固定或按计划变化 |
| **Throughput/samples_per_sec** | 训练速度 | GPU: >10K samples/s |

---

## 🎯 使用场景

### 单个模型监控
```bash
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default/separator1_hd64_stages2_depth3/tensorboard
```

### 对比多个模型
```bash
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default
```

### 对比多个实验
```bash
tensorboard --logdir experiments_refactored
```

### 远程访问
```bash
tensorboard --logdir experiments_refactored --host 0.0.0.0 --port 6006
```

---

## 🔍 常见问题诊断

### Loss 不下降
- ✅ 检查学习率是否过小
- ✅ 检查模型是否合适
- ✅ 检查数据是否正确

### Loss 震荡
- ✅ 降低学习率
- ✅ 增加 batch_size
- ✅ 检查 SNR 配置

### 训练太慢
- ✅ Throughput 很低？启用 `--use_amp`
- ✅ 使用 GPU：`--device cuda`
- ✅ 增加 batch_size

### 过拟合
- ✅ Loss/train 下降，Loss/validation 上升
- ✅ 减少模型复杂度
- ✅ 增加训练数据

---

## 💡 最佳实践

### 训练前
1. 小规模测试（100 batches）
2. 检查 Loss 是否下降
3. 确认 Throughput 正常

### 训练中
1. 定期查看 TensorBoard（每 5-10 分钟）
2. 监控 Loss 曲线
3. 检查异常情况

### 训练后
1. 完整查看训练曲线
2. 对比不同配置
3. 选择最佳模型

---

## 🛠️ 命令速查

```bash
# 基本使用
tensorboard --logdir <path>

# 自定义端口
tensorboard --logdir <path> --port 8888

# 远程访问
tensorboard --logdir <path> --host 0.0.0.0

# 后台运行
nohup tensorboard --logdir <path> --host 0.0.0.0 > tb.log 2>&1 &

# 停止
pkill tensorboard
```

---

**详细说明见 [`TENSORBOARD_GUIDE.md`](TENSORBOARD_GUIDE.md)**
