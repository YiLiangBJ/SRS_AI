# 📊 TensorBoard 集成指南

## 🎯 功能说明

TensorBoard 可以可视化训练过程中的各种指标，帮助你：
1. **监控训练进度**：实时查看 Loss 曲线
2. **调整超参数**：对比不同配置的效果
3. **诊断问题**：发现过拟合、欠拟合等问题
4. **性能分析**：查看吞吐量和训练速度

---

## 🚀 快速开始

### 1. 训练模型（自动启用 TensorBoard）

```bash
python train.py \
    --model_config separator1_default \
    --training_config default \
    --device cuda
```

**输出**：
```
📊 TensorBoard logging enabled: experiments_refactored/20251212_103045_separator1_default_default/separator1_hd64_stages2_depth3/tensorboard
   Run: tensorboard --logdir experiments_refactored/20251212_103045_separator1_default_default/separator1_hd64_stages2_depth3/tensorboard
```

---

### 2. 启动 TensorBoard

```bash
# 方法1：查看单个模型
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default_default/separator1_hd64_stages2_depth3/tensorboard

# 方法2：查看整个实验（对比多个模型）
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default_default

# 方法3：查看所有实验
tensorboard --logdir experiments_refactored
```

**输出**：
```
TensorBoard 2.15.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

---

### 3. 打开浏览器

访问：http://localhost:6006

---

## 📈 可视化的指标

### 1. Loss/train
训练损失曲线
- **用途**：监控训练是否收敛
- **预期**：逐渐下降并趋于稳定

### 2. NMSE/train 和 NMSE_dB/train
归一化均方误差（线性和dB）
- **用途**：评估模型性能
- **预期**：逐渐降低

### 3. Loss/validation
验证损失（如果启用验证）
- **用途**：检测过拟合
- **预期**：应该跟随训练损失下降

### 4. SNR/train
训练时的信噪比
- **用途**：查看 SNR 变化范围
- **预期**：如果使用动态 SNR，会在范围内变化

### 5. Learning_Rate
学习率
- **用途**：确认学习率设置正确
- **预期**：固定值或按计划变化

### 6. Throughput/samples_per_sec
训练吞吐量（样本/秒）
- **用途**：评估训练速度
- **预期**：GPU 训练应该很高（数万样本/秒）

---

## 🔍 使用场景

### 场景1：监控单个训练

```bash
# 训练
python train.py --model_config separator1_default --device cuda

# 启动 TensorBoard（在另一个终端）
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default_default/separator1_hd64_stages2_depth3/tensorboard

# 打开浏览器：http://localhost:6006
```

**查看**：
- Loss 是否下降
- NMSE 是否改善
- Throughput 是否稳定

---

### 场景2：对比不同损失函数

```bash
# 训练多个模型（不同损失函数）
python train.py \
    --model_config separator1_default \
    --training_config loss_comparison \
    --device cuda

# 启动 TensorBoard（对比整个实验）
tensorboard --logdir experiments_refactored/20251212_103045_separator1_default_loss_comparison

# 打开浏览器
```

**对比**：
- 哪个损失函数收敛更快？
- 哪个达到更低的 NMSE？
- 哪个训练更稳定？

---

### 场景3：Grid Search 对比

```bash
# Grid Search 训练
python train.py \
    --model_config separator1_grid_search \
    --training_config default \
    --device cuda

# 启动 TensorBoard（查看所有配置）
tensorboard --logdir experiments_refactored/20251212_103045_separator1_grid_search_default

# 打开浏览器
```

**分析**：
- 哪个 hidden_dim 最好？
- 哪个 num_stages 最好？
- 是否需要更多层数？

---

### 场景4：长期训练监控

```bash
# 启动长时间训练
nohup python train.py \
    --model_config separator1_default \
    --training_config long_training \
    --device cuda > train.log 2>&1 &

# 在另一台机器上启动 TensorBoard（远程访问）
tensorboard --logdir experiments_refactored --host 0.0.0.0 --port 6006

# 从其他电脑访问：http://server_ip:6006
```

**好处**：
- 实时监控远程训练
- 无需 SSH 登录
- 随时查看进度

---

## 🎨 TensorBoard 界面导航

### SCALARS 标签页

**左侧边栏**：
- 选择要显示的指标
- 平滑曲线（Smoothing 滑块）
- 显示/隐藏曲线

**主界面**：
- 多条曲线对比
- 缩放和平移
- 悬停查看具体值

**常用操作**：
1. **平滑曲线**：调整 Smoothing 滑块（0.6-0.9 较好）
2. **对比运行**：勾选多个 run
3. **缩放**：鼠标滚轮或拖拽
4. **下载数据**：点击左上角下载按钮

---

## 💡 高级技巧

### 1. 对比多次实验

```bash
# 结构化目录
experiments_refactored/
├── 20251212_103045_exp1/
│   ├── model1/tensorboard/
│   └── model2/tensorboard/
├── 20251212_105000_exp2/
│   ├── model1/tensorboard/
│   └── model2/tensorboard/

# 启动 TensorBoard（自动对比）
tensorboard --logdir experiments_refactored
```

**TensorBoard 会自动识别所有 runs 并允许对比！**

---

### 2. 自定义端口

```bash
# 默认端口（6006）
tensorboard --logdir experiments_refactored

# 自定义端口
tensorboard --logdir experiments_refactored --port 8888
```

---

### 3. 远程访问

```bash
# 服务器上启动（允许外部访问）
tensorboard --logdir experiments_refactored --host 0.0.0.0 --port 6006

# 从本地机器访问
http://server_ip:6006
```

---

### 4. 后台运行

```bash
# 后台运行 TensorBoard
nohup tensorboard --logdir experiments_refactored --host 0.0.0.0 > tensorboard.log 2>&1 &

# 查看日志
tail -f tensorboard.log

# 停止
pkill tensorboard
```

---

## 📊 实际示例

### 示例1：正常训练曲线

**Loss/train**：
```
初始：0.5 → 0.1 → 0.05 → 0.03 → 0.02 （稳定）
```
✅ **解读**：收敛良好

**NMSE_dB/train**：
```
初始：-3 dB → -10 dB → -15 dB → -20 dB （稳定）
```
✅ **解读**：性能持续改善

---

### 示例2：过拟合

**Loss/train**：持续下降
**Loss/validation**：开始上升

⚠️ **解读**：模型过拟合，需要：
- 减少模型复杂度
- 增加训练数据
- 使用正则化

---

### 示例3：学习率过大

**Loss/train**：剧烈震荡，不收敛

⚠️ **解读**：学习率太大，需要：
- 降低学习率（0.01 → 0.001）
- 使用学习率衰减

---

### 示例4：学习率过小

**Loss/train**：下降极慢

⚠️ **解读**：学习率太小，需要：
- 提高学习率（0.0001 → 0.001）
- 增加训练批次

---

## 🛠️ 调试技巧

### 1. 训练不收敛

**查看**：
- Loss/train 是否持续震荡？
- NMSE 是否毫无改善？

**尝试**：
- 降低学习率
- 检查数据生成是否正确
- 简化模型（减少 hidden_dim）

---

### 2. 训练太慢

**查看**：
- Throughput/samples_per_sec 是否很低？

**尝试**：
- 启用 `--use_amp`（混合精度）
- 启用 `--compile_model`（模型编译）
- 增加 batch_size
- 检查是否 CPU 训练（应该用 GPU）

---

### 3. Loss 震荡

**查看**：
- Loss 曲线是否剧烈上下波动？

**尝试**：
- 降低学习率
- 增加 batch_size
- 使用平滑的 SNR 变化

---

## 🎯 最佳实践

### 1. 训练前

```bash
# 先测试小规模训练
python train.py \
    --model_config separator1_small \
    --training_config quick_test \
    --num_batches 100 \
    --device cuda

# 查看 TensorBoard
tensorboard --logdir experiments_refactored
```

**确认**：
- Loss 是否下降
- 没有明显错误

---

### 2. 训练中

**定期检查**（每 5-10 分钟）：
- Loss 是否持续下降
- Throughput 是否稳定
- 有无异常曲线

---

### 3. 训练后

**完整分析**：
- 查看完整训练曲线
- 对比不同配置
- 选择最佳模型

---

## 📝 常见问题

### Q1: TensorBoard 无法启动

**错误**：`⚠️ TensorBoard not available`

**解决**：
```bash
pip install tensorboard
```

---

### Q2: 无法访问 TensorBoard

**问题**：浏览器打不开 http://localhost:6006

**解决**：
```bash
# 检查 TensorBoard 是否运行
ps aux | grep tensorboard

# 检查端口是否被占用
netstat -tuln | grep 6006

# 尝试其他端口
tensorboard --logdir experiments_refactored --port 8888
```

---

### Q3: 没有看到曲线

**问题**：TensorBoard 界面空白

**原因**：
- 训练还未开始记录
- 目录路径不对

**解决**：
```bash
# 确认目录存在
ls -R experiments_refactored/*/tensorboard

# 使用正确的路径
tensorboard --logdir <correct_path>
```

---

### Q4: 远程访问被拒绝

**问题**：从其他机器无法访问

**解决**：
```bash
# 服务器上启动时指定 --host 0.0.0.0
tensorboard --logdir experiments_refactored --host 0.0.0.0

# 检查防火墙
sudo ufw allow 6006/tcp
```

---

## 🎉 总结

### TensorBoard 让你能够：

1. ✅ **实时监控训练**：无需等训练结束
2. ✅ **对比实验**：快速找出最佳配置
3. ✅ **诊断问题**：及时发现训练异常
4. ✅ **优化超参数**：数据驱动的决策
5. ✅ **远程查看**：随时随地监控训练

---

**立即开始使用 TensorBoard，让训练过程可视化！** 📊🚀
