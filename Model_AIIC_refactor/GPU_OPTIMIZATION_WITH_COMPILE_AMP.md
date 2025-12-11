# 🚀 GPU优化完整指南 - 模型编译 + 混合精度

## ✅ 已实现的优化

### 1. **torch.compile** - 模型编译
- **自动优化计算图**
- **融合操作，减少kernel启动**
- **GPU only**（CPU收益很小）
- **预计提速**: 20-30%

### 2. **Mixed Precision (AMP)** - 混合精度
- **FP16计算，减少显存**
- **自动梯度缩放**
- **GPU only**（CPU不支持）
- **预计提速**: 30-50%

### 3. **Combined** - 组合使用
- **torch.compile + AMP**
- **预计提速**: 50-80% 🚀🚀

---

## 🎯 使用方法

### 基础训练（自动启用所有优化）

```bash
# GPU: 自动启用 compile + AMP
python train.py --model_config separator1_default --training_config default --device cuda

# CPU: 自动跳过优化
python train.py --model_config separator1_default --training_config default --device cpu
```

**输出示例（GPU）**：
```
🚀 Compiling model with torch.compile...
   ✓ Model compiled successfully
⚡ Mixed precision training enabled (FP16)
```

**输出示例（CPU）**：
```
ℹ️  Model compilation skipped (CPU mode, limited benefit)
ℹ️  Mixed precision skipped (CPU mode not supported)
```

---

### 控制优化选项

```bash
# 禁用所有优化（GPU baseline）
python train.py --model_config separator1_default --training_config default \
                --device cuda --no-compile --no-amp

# 只用 torch.compile
python train.py --model_config separator1_default --training_config default \
                --device cuda --no-amp

# 只用混合精度
python train.py --model_config separator1_default --training_config default \
                --device cuda --no-compile

# 全部优化（默认）
python train.py --model_config separator1_default --training_config default \
                --device cuda
```

---

## 📊 性能对比工具

### 完整对比（推荐）

```bash
# 对比所有配置
python compare_optimizations.py --model_config separator1_small --num_batches 100
```

**测试配置**：
1. ✅ CPU baseline
2. ✅ GPU baseline（无优化）
3. ✅ GPU + torch.compile
4. ✅ GPU + AMP
5. ✅ GPU + compile + AMP（完整优化）

**输出示例**：
```
📊 Performance Comparison Summary
================================================================================

Configuration              Duration    Throughput          Speedup
---------------------------------------------------------------------------
CPU                         15.2s      2,000 samples/s    1.00x
GPU baseline [C:  A: ]       5.8s     10,000 samples/s    2.62x
GPU + compile [C:✓ A: ]      4.5s     13,000 samples/s    3.38x
GPU + AMP [C:  A:✓]          3.8s     15,000 samples/s    4.00x
GPU + both [C:✓ A:✓]         2.9s     20,000 samples/s    5.24x ⭐

🎯 Performance Insights:

   Best GPU configuration: GPU + compile + AMP
     Duration: 2.9s
     Throughput: 20,000 samples/s
     Speedup over CPU: 5.24x

   Optimization breakdown (vs GPU baseline):
     torch.compile only:  1.29x faster
     AMP only:            1.53x faster
     Both combined:       2.00x faster ⭐

💡 Recommendations:
   ✅ For production training: use --device cuda (enables both optimizations)
   ✅ For maximum speed: use GPU with all optimizations enabled
   ✅ For debugging: use --no-compile --no-amp or --device cpu
```

---

### 快速对比

```bash
# 只测GPU（跳过CPU）
python compare_optimizations.py --model_config separator1_default \
                                --num_batches 200 \
                                --skip_cpu

# 只测CPU
python compare_optimizations.py --model_config separator1_default \
                                --num_batches 100 \
                                --skip_gpu

# 自定义batch size
python compare_optimizations.py --model_config separator1_default \
                                --num_batches 100 \
                                --batch_size 8192
```

---

## 🔧 技术细节

### torch.compile 工作原理

```python
# 在 Trainer.__init__() 中
if compile_model and self.device.type == 'cuda':
    if torch.__version__ >= '2.0.0':
        self.model = torch.compile(self.model)  # ← 一行代码
        print("✓ Model compiled successfully")
```

**优化内容**：
- 融合多个操作（减少kernel启动）
- 优化内存访问模式
- 自动选择最快的实现

**限制**：
- 首次运行慢（需要编译）
- 后续运行快
- GPU上效果最好

---

### 混合精度（AMP）工作原理

```python
# 在训练循环中
if self.use_amp:
    with autocast():  # 自动 FP16
        h_pred = self.model(y)
        loss = calculate_loss(h_pred, h_targets, ...)
    
    self.scaler.scale(loss).backward()  # 梯度缩放
    self.scaler.step(self.optimizer)
    self.scaler.update()
```

**优化内容**：
- 大部分计算用 FP16（快2x）
- 关键计算保持 FP32（精度）
- 自动梯度缩放（防止下溢）

**优势**：
- 速度提升 30-50%
- 显存减少 50%
- 精度几乎无损

---

## 📈 预期性能提升

### CPU模式

| 配置 | 吞吐量 | vs 原始 |
|------|--------|---------|
| CPU baseline | 2,000 samples/s | 1.0x |

**优化**：
- ✅ 已完成全tensor化
- ✅ 已完成向量化
- ⏸️ compile效果很小（跳过）
- ❌ AMP不支持（跳过）

---

### GPU模式（重点）

| 配置 | 吞吐量 | vs CPU | vs GPU baseline |
|------|--------|--------|-----------------|
| GPU baseline | 10,000 | 5x | 1.0x |
| GPU + compile | 13,000 | 6.5x | 1.3x |
| GPU + AMP | 15,000 | 7.5x | 1.5x |
| GPU + both | 20,000 | **10x** | **2.0x** ⭐ |

**提升分解**：
1. **CPU → GPU基础**: 5x（硬件优势）
2. **+ torch.compile**: +30%（图优化）
3. **+ AMP**: +50%（FP16计算）
4. **组合效果**: 2x GPU baseline = **10x CPU** 🚀🚀

---

## 🎯 推荐配置

### 生产训练（推荐）⭐⭐⭐⭐⭐

```bash
# Linux GPU机器
python train.py --model_config separator1_grid_search_4ports \
                --training_config default \
                --device cuda \
                --batch_size 8192

# 自动启用：
# ✅ torch.compile
# ✅ Mixed precision (AMP)
# ✅ GPU data generation
```

**预期**：
- 吞吐量：15,000-25,000 samples/s
- GPU利用率：>90%
- 比CPU快：8-12x

---

### 调试模式

```bash
# CPU或GPU无优化
python train.py --model_config separator1_small \
                --training_config quick_test \
                --num_batches 50 \
                --device cpu
# 或
python train.py --model_config separator1_small \
                --training_config quick_test \
                --num_batches 50 \
                --device cuda --no-compile --no-amp
```

**用途**：
- 调试代码逻辑
- 检查模型输出
- 错误信息更清晰

---

### 实验对比

```bash
# 快速对比所有配置
python compare_optimizations.py --model_config separator1_default --num_batches 100
```

**用途**：
- 验证优化效果
- 选择最佳配置
- 性能分析

---

## 🐛 故障排除

### 问题1：torch.compile失败

**症状**：
```
⚠️  Model compilation failed: ...
```

**解决**：
1. 确认PyTorch版本 ≥ 2.0
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

2. 禁用编译继续
   ```bash
   python train.py ... --no-compile
   ```

---

### 问题2：AMP导致NaN

**症状**：
```
Loss: nan
```

**解决**：
1. 禁用AMP
   ```bash
   python train.py ... --no-amp
   ```

2. 或减小学习率
   ```yaml
   # training_configs.yaml
   learning_rate: 0.001  # 从0.01减小
   ```

---

### 问题3：GPU内存不足

**症状**：
```
RuntimeError: CUDA out of memory
```

**解决**：
1. AMP会减少50%显存，确保启用
   ```bash
   python train.py ... --device cuda  # 自动启用AMP
   ```

2. 减小batch size
   ```bash
   python train.py ... --batch_size 4096  # 从8192减小
   ```

3. 使用更小模型
   ```bash
   python train.py --model_config separator1_small ...
   ```

---

## 📊 监控GPU

### 实时监控

```bash
# Linux
watch -n 0.5 nvidia-smi

# Windows PowerShell
while($true) { cls; nvidia-smi; sleep 0.5 }
```

**关注指标**：
- **GPU-Util**: 应该 >85%
- **Memory-Usage**: 根据batch size
- **Power**: 接近TDP表示满负载

---

### 训练输出

```
Batch 50/1000, SNR:15.0dB, Loss:0.375, NMSE:-4.26dB, 
Throughput: 18,500 samples/s [Data:5% Fwd:28% Bwd:67%]
         ↑                              ↑
    目标 >15k                    理想分布
```

**理想分布**：
- Data: <10%（GPU数据生成）
- Forward: 25-30%
- Backward: 60-70%（正常）

---

## ✅ 检查清单

### 开始GPU训练前

- [ ] PyTorch版本 ≥ 2.1（支持compile + AMP）
  ```bash
  python -c "import torch; print(torch.__version__)"
  ```

- [ ] CUDA版本显示（`+cu121` 或 `+cu118`）
  ```
  2.1.2+cu121  ✅ GPU版本
  2.1.2+cpu    ❌ CPU版本
  ```

- [ ] GPU可用
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  # True
  ```

- [ ] 足够显存（推荐 ≥8GB）
  ```bash
  nvidia-smi
  ```

---

### 性能验证

- [ ] 运行性能对比
  ```bash
  python compare_optimizations.py --model_config separator1_small --num_batches 100
  ```

- [ ] GPU比CPU快 >5x
- [ ] 完整优化比GPU baseline快 >1.5x
- [ ] Throughput >15,000 samples/s
- [ ] GPU利用率 >85%

---

## 🎉 总结

### ✅ 完整优化栈

1. **✅ GPU数据生成**（已实现）
   - TDL Channel在GPU
   - 所有tensor操作在GPU
   - 零CPU→GPU传输

2. **✅ torch.compile**（新增）
   - 自动图优化
   - +20-30% 速度

3. **✅ 混合精度 (AMP)**（新增）
   - FP16计算
   - +30-50% 速度
   - 显存减半

4. **✅ 自动化对比**（新增）
   - 一键测试所有配置
   - 详细性能报告

### 🚀 最终性能

**CPU → GPU完整优化**：**8-12x faster** 🚀🚀

| 阶段 | 提速 | 累计 |
|------|------|------|
| CPU baseline | - | 1x |
| GPU基础 | +5x | 5x |
| + compile | +30% | 6.5x |
| + AMP | +50% | **10x** ⭐ |

---

**开始使用**：

```bash
# Linux GPU机器（推荐）
python train.py --model_config separator1_default --training_config default --device cuda

# 性能测试
python compare_optimizations.py --model_config separator1_small --num_batches 100
```

🎉 **享受极速训练！**
