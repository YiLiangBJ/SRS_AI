# 单NUMA节点CPU使用率测试指南

## 问题背景

在双NUMA节点的Linux机器上，您遇到了以下问题：
- 可以看到很多线程，但大多数处于S状态（睡眠状态）
- CPU使用率很低
- 怀疑是跨NUMA节点通信导致的性能问题

## 解决方案

我们添加了强制单NUMA节点训练的功能，让您可以测试在单个NUMA节点上的CPU使用率，避免跨NUMA通信开销。

## 新增参数

### `--force-single-numa`
强制使用单个NUMA节点进行训练，避免跨NUMA节点的进程间通信。

### `--numa-node-id <id>`
指定要使用的NUMA节点ID（默认为0）。

## 使用方法

### 1. Windows系统测试
```bash
# 测试单进程训练（Windows不支持NUMA绑定，但可以测试单进程性能）
python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 2 --batch-size 32

# 使用Windows批处理脚本
test_single_numa.bat
```

### 2. Linux系统测试

#### 方法1：直接命令行测试
```bash
# 测试NUMA节点0
python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 2 --batch-size 32 --debug

# 测试NUMA节点1
python train_distributed.py --force-single-numa --numa-node-id 1 --num-epochs 2 --batch-size 32 --debug
```

#### 方法2：使用专用测试脚本
```bash
# 赋予脚本执行权限
chmod +x test_numa_single_node.sh

# 运行交互式测试菜单
./test_numa_single_node.sh

# 或直接测试特定NUMA节点
./test_numa_single_node.sh test 0    # 测试NUMA节点0
./test_numa_single_node.sh test 1    # 测试NUMA节点1
./test_numa_single_node.sh info      # 显示NUMA拓扑信息
```

#### 方法3：使用资源监控脚本
```bash
# 启动训练并实时监控资源使用
python monitor_training.py --train-command "python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 3 --batch-size 32"
```

## 预期结果

### 成功的单NUMA节点训练应该显示：

1. **NUMA绑定确认**：
   ```
   🔒 FORCE SINGLE NUMA: Process 0 forced to NUMA node 0
   🎯 Process 0 bound to NUMA node 0:
      - CPU cores: 0-11 (例如)
      - PyTorch threads: 12
   ```

2. **高CPU使用率**：
   - 通过`htop`或`top -H`应该能看到线程处于"R"（运行）状态
   - CPU使用率应该接近100%（在指定的核心上）

3. **正常的训练进度**：
   ```
   批次 [001/1000] - 损失: 50706.265625, NMSE: 1.39 dB
   批次 [002/1000] - 损失: 49627.027344, NMSE: 1.53 dB
   ...
   ```

## 性能诊断步骤

### 第1步：测试单NUMA节点性能
```bash
# 在Linux上测试NUMA节点0
python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 2 --batch-size 32 --debug
```

在训练过程中，使用另一个终端监控：
```bash
# 监控CPU使用率
htop

# 监控线程状态
ps -eLf | grep python

# 监控NUMA统计
watch -n 1 numastat

# 检查CPU亲和性
taskset -cp <training_pid>
```

### 第2步：对比不同NUMA节点
如果NUMA节点0表现良好，测试节点1：
```bash
python train_distributed.py --force-single-numa --numa-node-id 1 --num-epochs 2 --batch-size 32 --debug
```

### 第3步：分析结果
- **如果单NUMA节点CPU使用率高**：说明计算本身没问题，之前的低使用率是由跨NUMA通信引起的
- **如果单NUMA节点CPU使用率仍然低**：说明问题可能在于：
  - 数据生成瓶颈
  - I/O等待
  - PyTorch线程配置问题
  - 算法实现效率问题

## 进一步优化建议

### 如果单NUMA节点性能良好：
1. **继续使用单NUMA节点训练**：
   ```bash
   python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 100
   ```

2. **或者优化跨NUMA通信**：
   - 增加batch size来减少通信频率
   - 使用更高效的DDP后端
   - 调整通信模式

### 如果单NUMA节点性能仍然不理想：
1. **检查数据生成器**：
   ```bash
   # 减小batch size测试
   python train_distributed.py --force-single-numa --batch-size 16
   
   # 增加PyTorch线程数
   export OMP_NUM_THREADS=24
   python train_distributed.py --force-single-numa
   ```

2. **使用性能分析工具**：
   ```bash
   # 使用perf分析CPU热点
   perf top -p <training_pid>
   
   # 或使用py-spy
   pip install py-spy
   py-spy top --pid <training_pid>
   ```

## 监控命令参考

### 实时监控CPU使用率：
```bash
# 总体CPU使用率
vmstat 1

# 每个核心的使用率
mpstat -P ALL 1

# 进程级监控
top -H -p <training_pid>

# 详细的线程状态
ps -eLo pid,tid,stat,pcpu,comm | grep <training_pid>
```

### 监控NUMA状态：
```bash
# NUMA内存使用统计
numastat

# 特定进程的NUMA信息
numactl --show --pid <training_pid>

# NUMA硬件拓扑
numactl --hardware
```

## 自动化测试脚本

使用提供的脚本可以自动化测试过程：

```bash
# Linux完整测试
./test_numa_single_node.sh

# 仅显示NUMA信息
./test_numa_single_node.sh info

# 快速测试NUMA节点0
./test_numa_single_node.sh test 0

# Windows测试
test_single_numa.bat
```

这些脚本会自动收集系统信息、启动训练、监控资源使用情况，并提供诊断建议。
