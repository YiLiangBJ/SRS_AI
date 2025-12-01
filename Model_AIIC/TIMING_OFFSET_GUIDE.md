# 随机时间偏移功能说明

## 📍 时间偏移概述

**功能**: 每个 port 自动添加独立的随机时间偏移  
**范围**: ±256Tc，其中 Tc = 1/(480e3*4096) ≈ 0.509 ns  
**目的**: 模拟真实场景中的定时误差，提高模型鲁棒性

---

## 🔧 实现原理

### 时间偏移计算

```python
# 3GPP 基本时间单位
Tc = 1 / (480e3 * 4096)  # ≈ 0.509 ns

# 采样间隔
Ts = 1 / (scs * Ktc * seq_len)  # 例如: 30kHz * 4 * 12

# 每个 port 随机偏移（Tc 单位）
timing_offset_Tc = random.uniform(-256, 256)  # 每个 sample, 每个 port

# 转换为采样点单位
timing_offset_samples = timing_offset_Tc * Tc / Ts
```

### 频域相位旋转

时间偏移通过频域相位旋转实现：

```python
# 时域信号
h(t) -> 频域: H(f)

# 应用时间偏移 delta
H_shifted(f) = H(f) * exp(j * 2π * f * delta)

# 转回时域
h_offset(t) = IFFT(H_shifted(f))
```

**具体实现**:
```python
H_fft = FFT(h)  # 频域
k = [0, 1, 2, ..., L-1]  # 频率索引
phase_shift = exp(j * 2π * k * delta / L)
H_shifted = H_fft * phase_shift
h_offset = IFFT(H_shifted)
```

---

## 📊 特性详解

### 独立性
- ✅ **每个 sample** 有独立的随机偏移
- ✅ **每个 port** 有独立的随机偏移
- ✅ 每个 batch 重新生成偏移

### 随机范围

| 参数 | 值 | 说明 |
|------|-----|------|
| 最大偏移（Tc 单位） | ±256 | 3GPP 标准范围 |
| Tc（秒） | ~0.509 ns | 3GPP 基本时间单位 |
| 最大时间偏移 | ±130 ns | 256 × 0.509 ns |
| 等效采样点偏移 | 取决于 Ts | delta = 256Tc / Ts |

### 示例

**配置**: SCS=30kHz, Ktc=4, seq_len=12

```
Ts = 1 / (30e3 * 4 * 12) ≈ 694 ns
Tc ≈ 0.509 ns
delta_max = 256 * 0.509 / 694 ≈ 0.188 samples
```

即使是 256Tc 的偏移，在这个采样率下只对应 ~0.2 个采样点。

---

## 🎯 训练效果

### 无时间偏移 vs 有时间偏移

| 场景 | 无偏移 | 有偏移 |
|------|--------|--------|
| 训练数据 | 理想对齐 | 真实定时误差 |
| 模型泛化 | 较差 | 更好 |
| 实际部署 | 性能下降 | 鲁棒 |
| NMSE (训练) | 可能更低 | 稍高（更难） |
| NMSE (实际) | 下降明显 | 稳定 |

### 训练建议

**训练时**:
- ✅ 开启随机偏移（已默认开启）
- ✅ 使用多种 TDL 配置
- ✅ 使用 SNR 范围

**测试时**:
- 可以固定偏移或设为 0 来测试理想性能
- 使用真实数据验证鲁棒性

---

## 🔍 验证时间偏移

### 方法 1: 运行测试脚本

```bash
python Model_AIIC/test_timing_offset.py
```

输出示例:
```
✅ 随机时间偏移已添加：
  - 每个 port 有独立的随机时间偏移
  - 范围: ±256*Tc，其中 Tc ≈ 0.509 ns
  - 实现方式: 频域相位旋转
```

### 方法 2: 代码验证

```python
from Model_AIIC.test_separator import generate_training_data

# 生成数据
y, h_targets, pos_values, h_true = generate_training_data(
    batch_size=10,
    snr_db=20.0,
    seq_len=12,
    num_ports=4,
    tdl_config='A-30'
)

print(f"生成的数据已包含随机时间偏移")
print(f"h_true shape: {h_true.shape}")  # (10, 4, 12)
# 每个样本的每个 port 都有独立的随机偏移
```

---

## 📝 技术细节

### 为什么是 ±256Tc？

- **3GPP 标准**: TR 38.901 定义的定时偏移范围
- **实际场景**: 覆盖基站间同步误差、传播延迟变化
- **合理范围**: 既有挑战性又不至于过度困难

### 为什么用频域相位旋转？

1. **精度**: 可以实现亚采样点精度的时间偏移
2. **效率**: FFT/IFFT 高效实现
3. **数学等价**: 时域卷积 = 频域相乘

### 与固定位置偏移的关系

```python
pos_values = [0, 3, 6, 9]  # 固定的 comb 位置
timing_offset = random.uniform(-256, 256)  # 随机微调

# 最终效果 = 固定位置 + 随机偏移
total_shift = pos_values[i] + timing_offset
```

---

## 🆚 对比：有无时间偏移

### 示例训练

```bash
# 训练 1: 有时间偏移（默认）
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --save_dir "./with_timing_offset"

# 训练 2: 无时间偏移（需要修改代码临时禁用）
# 修改 generate_training_data 函数，设置:
# timing_offset_samples = np.zeros((batch_size, num_ports))
```

**预期结果**:
- **有偏移**: NMSE 稍高，但模型更鲁棒
- **无偏移**: 训练 NMSE 可能更低，但泛化能力差

---

## 💡 最佳实践

### 推荐配置

```bash
python Model_AIIC/test_separator.py \
  --batches 2000 \
  --batch_size 2048 \
  --stages 3 \
  --snr "0,30" \
  --tdl "A-30,B-100,C-300" \
  --save_dir "./robust_training"
```

这个配置结合了：
- ✅ 随机时间偏移（自动）
- ✅ 多种 SNR
- ✅ 多种 TDL 配置

### 调试建议

如果模型难以收敛：
1. 先用固定 SNR 训练（如 `--snr 20.0`）
2. 再逐步增加难度（`--snr "10,30"`）
3. 最后添加多种 TDL（`--tdl "A-30,B-100"`）

---

## 🔗 相关代码

**实现位置**: `Model_AIIC/test_separator.py` 中的 `generate_training_data()` 函数

**关键代码段**:
```python
# 生成随机偏移
timing_offset_Tc = np.random.uniform(-256, 256, (batch_size, num_ports))
timing_offset_samples = timing_offset_Tc * Tc / Ts

# 应用偏移
H_fft = torch.fft.fft(h_base, dim=-1)
phase_shift = torch.exp(1j * 2 * np.pi * k * delta / seq_len)
H_shifted = H_fft * phase_shift
h_offset = torch.fft.ifft(H_shifted, dim=-1)
```

---

## 📚 参考文献

- 3GPP TR 38.901: Study on channel model for frequencies from 0.5 to 100 GHz
- Section 7.7: TDL channel models
- Timing offset specifications

---

**最后更新**: 2025-12-01  
**版本**: v1.0
