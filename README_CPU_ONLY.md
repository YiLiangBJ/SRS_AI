# CPU-Only Training Configuration

## 概述

本项目已完全配置为仅使用CPU进行训练，彻底移除了所有CUDA/GPU依赖。这确保代码可以在任何环境下运行，无论是否有GPU可用。

## 主要修改

### 1. 环境变量设置

在所有主要模块的开头都设置了以下环境变量：

```python
import os

# Force CPU-only execution - disable all CUDA/GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow CUDA warnings
```

这些设置的作用：
- `CUDA_VISIBLE_DEVICES=''`: 隐藏所有CUDA设备
- `CUDA_LAUNCH_BLOCKING='1'`: 阻止CUDA内核启动
- `XLA_FLAGS`: 强制XLA使用CPU
- `TF_CPP_MIN_LOG_LEVEL='2'`: 抑制TensorFlow CUDA警告

### 2. 修改的文件列表

#### 核心训练文件：
- `train_distributed.py`: 主分布式训练脚本
- `trainMLPmmse.py`: MLP MMSE训练器
- `model_Traditional.py`: SRS信道估计器模型
- `model_AIpart.py`: AI增强模块（保持现有逻辑）

#### 数据生成文件：
- `data_generator_refactored.py`: 重构的数据生成器
- `professional_channels.py`: 专业信道模型

#### 系统检测文件：
- `system_detection.py`: 系统硬件检测
- `evaluate_performance.py`: 性能评估

### 3. 设备选择修改

所有默认设备选择都从：
```python
device: str = "cuda" if torch.cuda.is_available() else "cpu"
```

修改为：
```python
device: str = "cpu"  # Force CPU-only execution
```

### 4. DDP（分布式数据并行）修改

在`train_distributed.py`中，DDP配置从：
```python
self.mmse_module = DDP(
    self.mmse_module,
    device_ids=[self.rank],
    output_device=self.rank,
    find_unused_parameters=True
)
```

修改为（CPU-only）：
```python
self.mmse_module = DDP(
    self.mmse_module,
    find_unused_parameters=True
)
```

### 5. 系统检测修改

在`system_detection.py`中，GPU检测从：
```python
self.has_cuda = torch.cuda.is_available()
self.gpu_count = torch.cuda.device_count() if self.has_cuda else 0
```

修改为：
```python
self.has_cuda = False
self.gpu_count = 0
```

## 使用方法

### 单进程训练（推荐）
```bash
python train_distributed.py --num-epochs 100 --batch-size 64
```

### 强制单NUMA节点训练
```bash
python train_distributed.py --force-single-numa --numa-node-id 0 --num-epochs 50
```

### 测试CPU-only配置
```bash
python test_cpu_only.py
```

## 验证结果

运行`test_cpu_only.py`应该显示：
- ✅ CUDA available: False
- ✅ GPU count: 0
- ✅ 所有张量操作在CPU设备上执行
- ✅ 模型初始化成功且使用CPU设备

## 性能特点

### CPU优化：
- 使用NUMA感知的进程绑定
- 物理核心优化的PyTorch线程数
- 内存局部性优化

### 预期性能：
- 训练速度：纯CPU（无GPU加速）
- 内存使用：相对较低（无GPU内存）
- 可扩展性：基于CPU核心数

## 故障排除

### 如果仍然看到CUDA警告：
1. 确保所有环境变量都在模块导入之前设置
2. 重启Python进程
3. 检查是否有其他进程或库尝试初始化CUDA

### 性能优化：
1. 使用`--force-single-numa`在单NUMA节点上测试
2. 调整批大小以适应CPU内存
3. 监控CPU使用率确保充分利用

## 兼容性

- ✅ Windows PC
- ✅ Linux服务器（单/多NUMA节点）
- ✅ 任何无GPU环境
- ✅ Docker容器
- ✅ 云CPU实例

## 注意事项

1. **训练速度**：CPU训练比GPU训练慢，但更稳定可靠
2. **内存使用**：注意调整批大小以适应系统内存
3. **线程优化**：代码已自动检测并优化CPU线程数
4. **NUMA感知**：在多NUMA节点系统上自动优化内存访问

## 下一步

建议在生产环境中：
1. 使用较小的批大小开始测试
2. 监控CPU和内存使用情况
3. 根据硬件配置调整超参数
4. 考虑使用分布式训练（如果有多个NUMA节点）
