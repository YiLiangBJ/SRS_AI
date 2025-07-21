# NUMA-Aware Distributed Training Implementation

## 概述
根据您的要求，对 `train_distributed.py` 进行了 NUMA 感知的分布式训练优化，主要实现了以下功能：

## 主要修改

### 1. NUMA 拓扑检测
- **函数**: `detect_numa_topology()`
- **功能**: 
  - Linux: 使用 `lscpu` 命令检测 NUMA 节点数和物理核心数
  - Windows: 自动检测并设置合理的默认值
  - 返回 NUMA 节点数、每节点核心数、总核心数等信息

### 2. 进程 NUMA 绑定
- **函数**: `bind_process_to_numa_node(rank, numa_info)`
- **功能**:
  - Linux: 使用 `taskset` 命令将进程绑定到特定 NUMA 节点的物理核心
  - 使用 `torch.set_num_threads()` 设置 PyTorch 线程数为该 NUMA 节点的物理核心数
  - Windows: 自动设置合理的线程数

### 3. 智能 World Size 确定
- **函数**: `determine_optimal_world_size(numa_info, enable_ddp)`
- **逻辑**:
  - Linux 多 NUMA 节点 + DDP 启用: `world_size = numa_nodes`
  - Windows 或单 NUMA 节点: `world_size = 1`

### 4. 错误处理优化
- **移除**: 所有 `try/except` 错误掩盖
- **改进**: 错误直接传播，便于调试和问题定位

### 5. 平台兼容性
- **Windows**: 单进程训练，自动线程优化
- **Linux**: NUMA 感知多进程 DDP 训练

## 代码结构

### 核心函数
```python
# NUMA 拓扑检测
detect_numa_topology() -> Dict

# 进程绑定到 NUMA 节点
bind_process_to_numa_node(rank: int, numa_info: Dict)

# 智能 world_size 确定
determine_optimal_world_size(numa_info: Dict, enable_ddp: bool) -> int
```

### DistributedTrainer 类更新
- 添加了 `numa_info` 参数
- 初始化时自动调用 NUMA 绑定
- 训练信息中显示 NUMA 相关信息

## 使用方式

### Windows PC (推荐)
```bash
# 单进程训练
python train_distributed.py

# 使用 PowerShell 脚本
.\launch_training.ps1
```

### Linux 服务器 (2+ NUMA 节点)
```bash
# NUMA 感知 DDP 训练
python train_distributed.py --enable-ddp

# 使用 Shell 脚本
./launch_training_numa.sh --enable-ddp
```

### 测试 NUMA 功能
```bash
# 运行 NUMA 测试
python test_numa_optimization.py

# PowerShell 测试
.\launch_training.ps1 -TestNuma

# Linux 测试
./launch_training_numa.sh --test-numa
```

## 预期行为

### Linux 服务器 (假设 2 个 NUMA 节点，每节点 56 物理核心)
- **检测结果**: 2 个 NUMA 节点，每节点 56 核心
- **训练配置**: `world_size=2`, rank 0 和 1
- **进程绑定**:
  - Rank 0: 绑定到 NUMA 节点 0 的 0-55 核心
  - Rank 1: 绑定到 NUMA 节点 1 的 56-111 核心
- **PyTorch 线程**: 每进程 56 线程

### Windows PC
- **检测结果**: 1 个 NUMA 节点，总核心数
- **训练配置**: `world_size=1`, 单进程
- **线程设置**: PyTorch 线程数 = 总核心数

## 启动脚本

### PowerShell (Windows) - `launch_training.ps1`
```powershell
# 基本训练
.\launch_training.ps1

# NUMA 测试
.\launch_training.ps1 -TestNuma

# 自定义参数
.\launch_training.ps1 -NumEpochs 200 -BatchSize 128
```

### Bash (Linux) - `launch_training_numa.sh`
```bash
# 基本训练
./launch_training_numa.sh

# NUMA 感知 DDP
./launch_training_numa.sh --enable-ddp

# NUMA 测试
./launch_training_numa.sh --test-numa
```

## 测试验证

### 已验证功能
1. ✅ Windows 平台 NUMA 检测 (1 节点, 14 核心)
2. ✅ 进程线程绑定 (设置为 14 线程)
3. ✅ 智能 world_size 确定 (Windows: 1)
4. ✅ 错误传播 (KeyboardInterrupt 正确传播)
5. ✅ 训练启动 (模型初始化、数据生成、损失下降)

### 测试工具
- `test_numa_optimization.py`: 完整的 NUMA 功能测试套件
- 验证 NUMA 检测、绑定、配置、PyTorch 集成

## 关键优势

1. **性能优化**: NUMA 感知进程绑定避免跨节点内存访问
2. **自动配置**: 根据硬件拓扑自动确定最优配置
3. **平台兼容**: Windows 和 Linux 都有优化支持
4. **错误透明**: 移除错误掩盖，便于调试
5. **简单使用**: 保持原有 API，自动应用优化

## 注意事项

1. **Linux 依赖**: 需要 `lscpu` 和 `taskset` 命令
2. **权限要求**: NUMA 绑定可能需要特定权限
3. **硬件要求**: 多 NUMA 节点才能发挥 DDP 优势
4. **批处理大小**: 建议根据 world_size 调整总批处理大小

## 下一步建议

1. 在实际的 Linux 双 NUMA 节点服务器上测试
2. 根据实际硬件调整 NUMA 绑定策略
3. 监控性能指标验证优化效果
4. 考虑添加更细粒度的 NUMA 控制选项
