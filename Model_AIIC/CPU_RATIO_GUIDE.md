# CPU 核心控制参数使用说明

## 新增参数: `--cpu_ratio`

控制使用物理 CPU 核心的比例。

### 参数说明

```bash
--cpu_ratio RATIO
```

- **类型**: float
- **范围**: 0.0 ~ 1.0
- **默认值**: 1.0 (使用所有物理核心)

### 工作原理

1. **检测物理核心数**
   - 自动检测系统可用的逻辑 CPU 数
   - 除以 2 得到物理核心数（假设 2-way SMT/超线程）

2. **应用比例**
   - `num_threads = int(physical_cores * cpu_ratio)`
   - 最少使用 1 个核心

3. **设置线程数**
   - 应用到 PyTorch、OpenMP、MKL 等所有并行库

### 使用示例

#### 使用 50% 核心
```bash
python Model_AIIC/test_separator.py \
  --batches 100 \
  --batch_size 128 \
  --cpu_ratio 0.5
```

**输出示例**:
```
🚀 CPU Optimization:
   Available CPUs: 14
   Physical cores: 7
   CPU ratio: 0.50 (50%)
   Using threads: 3
```

#### 使用 25% 核心
```bash
python Model_AIIC/test_separator.py \
  --batches 100 \
  --batch_size 128 \
  --cpu_ratio 0.25
```

**输出示例**:
```
🚀 CPU Optimization:
   Available CPUs: 14
   Physical cores: 7
   CPU ratio: 0.25 (25%)
   Using threads: 1
```

#### 使用全部核心（默认）
```bash
python Model_AIIC/test_separator.py \
  --batches 100 \
  --batch_size 128
  # 或显式指定
  --cpu_ratio 1.0
```

**输出示例**:
```
🚀 CPU Optimization:
   Available CPUs: 14
   Physical cores: 7
   CPU ratio: 1.00 (100%)
   Using threads: 7
```

### 与环境变量的关系

如果设置了 `OMP_NUM_THREADS` 环境变量，它会**覆盖** `--cpu_ratio` 设置：

```bash
# 环境变量优先级更高
export OMP_NUM_THREADS=4
python Model_AIIC/test_separator.py --cpu_ratio 0.5  # 实际使用 4 个线程
```

### 应用场景

#### 1. 多任务并行
在同一台机器上运行多个实验时：

```bash
# 终端 1: 使用 50% 核心
python Model_AIIC/test_separator.py --cpu_ratio 0.5 --save_dir exp1 &

# 终端 2: 使用另外 50% 核心
taskset -c 8-15 python Model_AIIC/test_separator.py --cpu_ratio 0.5 --save_dir exp2 &
```

#### 2. 资源限制
避免占用所有 CPU 影响其他服务：

```bash
# 只使用 30% 核心
python Model_AIIC/test_separator.py --cpu_ratio 0.3 --batches 10000
```

#### 3. 性能测试
测试不同核心数对性能的影响：

```bash
# 测试 1 核
python Model_AIIC/test_separator.py --cpu_ratio 0.14 --batches 100

# 测试 4 核
python Model_AIIC/test_separator.py --cpu_ratio 0.57 --batches 100

# 测试全核
python Model_AIIC/test_separator.py --cpu_ratio 1.0 --batches 100
```

### 完整示例：超参数搜索 + CPU 控制

```bash
python Model_AIIC/test_separator.py \
  --batches 1000 \
  --batch_size 2048 \
  --stages "2,3,4" \
  --share_weights "True,False" \
  --snr "10,30" \
  --tdl "A-30,B-100,C-300" \
  --early_stop 0.01 \
  --save_dir "./experiments" \
  --cpu_ratio 0.8
```

这会：
- 使用 80% 的物理核心
- 测试 6 种超参数组合
- 每个组合独立训练
- 自动早停和保存结果

### 性能建议

| 场景 | 推荐 cpu_ratio | 说明 |
|------|---------------|------|
| 单任务训练 | 1.0 | 充分利用所有核心 |
| 多任务并行 | 0.5 | 为其他任务预留资源 |
| 后台训练 | 0.3-0.5 | 不影响交互式使用 |
| 性能测试 | 变化 | 测试扩展性 |

### 注意事项

1. **物理核心检测**：代码假设 2-way SMT（超线程），即逻辑核心数 = 2 × 物理核心数
2. **最小值**：至少使用 1 个核心（即使 `cpu_ratio < 1/physical_cores`）
3. **优先级**：`OMP_NUM_THREADS` 环境变量 > `--cpu_ratio` 参数
4. **生效时机**：必须在导入 torch 之前设置，所以代码会先解析这个参数

---

**更新日期**: 2025-12-01  
**版本**: v1.0
