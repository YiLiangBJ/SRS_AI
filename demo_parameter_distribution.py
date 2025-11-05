"""
参数类型分布可视化示例
帮助理解"80个weight, 24,976参数"的含义
"""

import torch
import torch.nn as nn
from collections import defaultdict

print("\n" + "="*100)
print(" " * 30 + "📊 参数类型分布详解")
print("="*100)

# 创建一个简单的示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 8, 3)      # 有 weight 和 bias
        self.bn1 = nn.BatchNorm1d(8)         # 有 weight 和 bias
        self.conv2 = nn.Conv1d(8, 16, 3)     # 有 weight 和 bias
        self.bn2 = nn.BatchNorm1d(16)        # 有 weight 和 bias
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x

model = SimpleModel()

print("\n【示例模型结构】")
print("  conv1: Conv1d(2, 8, kernel_size=3)")
print("  bn1:   BatchNorm1d(8)")
print("  conv2: Conv1d(8, 16, kernel_size=3)")
print("  bn2:   BatchNorm1d(16)")

# 统计参数
print("\n" + "-"*100)
print("【逐个参数分析】")
print("-"*100)
print(f"{'参数名称':<30s} {'形状':<20s} {'元素个数':>15s} {'类型':>10s}")
print("-"*100)

param_types = defaultdict(lambda: {'tensors': [], 'total': 0})

for name, param in model.named_parameters():
    param_type = name.split('.')[-1]  # 提取 'weight' 或 'bias'
    numel = param.numel()
    
    param_types[param_type]['tensors'].append((name, param.shape, numel))
    param_types[param_type]['total'] += numel
    
    print(f"{name:<30s} {str(tuple(param.shape)):<20s} {numel:>15,d} {param_type:>10s}")

print("-"*100)

# 汇总统计
print("\n" + "="*100)
print("【参数类型分布汇总】")
print("="*100)
print(f"{'类型':<15s} {'张量数量':>15s} {'参数总数':>20s} {'占比':>12s}")
print("-"*100)

total_params = sum(p.numel() for p in model.parameters())

for ptype, info in sorted(param_types.items(), key=lambda x: x[1]['total'], reverse=True):
    tensor_count = len(info['tensors'])
    param_total = info['total']
    percentage = (param_total / total_params * 100) if total_params > 0 else 0
    
    print(f"{ptype:<15s} {tensor_count:>15,d} {param_total:>20,d} {percentage:>11.1f}%")

print("-"*100)
print(f"{'总计':<15s} {sum(len(info['tensors']) for info in param_types.values()):>15,d} {total_params:>20,d} {'100.0%':>12s}")

# 详细解释
print("\n" + "="*100)
print("【详细说明】")
print("="*100)

for ptype, info in sorted(param_types.items()):
    print(f"\n🔹 {ptype.upper()}")
    print(f"   张量数量: {len(info['tensors'])} 个")
    print(f"   参数总数: {info['total']:,} 个")
    print(f"\n   具体组成：")
    
    for i, (name, shape, numel) in enumerate(info['tensors'], 1):
        print(f"     {i}. {name:<30s} {str(tuple(shape)):<20s} = {numel:>6,d} 参数")
    
    print(f"\n   计算: {' + '.join(str(t[2]) for t in info['tensors'])} = {info['total']:,}")

# 可视化
print("\n" + "="*100)
print("【参数占比可视化】")
print("="*100)

for ptype, info in sorted(param_types.items(), key=lambda x: x[1]['total'], reverse=True):
    param_total = info['total']
    percentage = (param_total / total_params * 100) if total_params > 0 else 0
    bar_length = int(percentage)
    
    print(f"\n{ptype:<10s} [{percentage:>5.1f}%]  {'█' * bar_length}")
    print(f"{'':10s} {param_total:>7,d} 参数 / {len(info['tensors'])} 个张量")

# 关键理解
print("\n" + "="*100)
print("【关键理解】")
print("="*100)
print("""
✨ 两个维度的统计：

1️⃣  张量数量（个数统计）
   - 计数有多少个参数张量
   - 示例：4个 weight 张量（conv1.weight, bn1.weight, conv2.weight, bn2.weight）
   - 用途：了解模型有多少个参数层

2️⃣  参数总数（元素统计）  
   - 统计所有参数张量包含的标量总数
   - 示例：这4个weight张量共包含 48+8+384+16 = 456 个参数
   - 用途：了解模型的总参数规模（影响内存、计算量）

📊 类比理解：
   - 张量数量 = 你有几个钱包
   - 参数总数 = 这些钱包里总共有多少钱
   
   "4个钱包，总共456元"
   ↓
   "4个weight张量，总共456个参数"
""")

print("\n" + "="*100)
print("✓ 示例完成")
print("="*100)
