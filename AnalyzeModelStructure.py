"""
通用模型结构分析工具
完全基于模型实例自动分析，无任何硬编码
"""
# -*- coding: utf-8 -*-

import sys
import io
import torch
import torch.nn as nn
from collections import defaultdict

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_parameter(name, param):
    """分析单个参数的详细信息"""
    return {
        'name': name,
        'shape': tuple(param.shape),
        'numel': param.numel(),
        'dtype': str(param.dtype),
        'requires_grad': param.requires_grad,
        'device': str(param.device)
    }


def get_module_parameters_detail(module, module_name=""):
    """获取模块的所有参数详细信息"""
    params_info = []
    buffers_info = []
    
    # 分析参数（可训练和不可训练）
    for name, param in module.named_parameters(recurse=False):
        params_info.append(analyze_parameter(name, param))
    
    # 分析缓冲区（如BatchNorm的running_mean, running_var等）
    for name, buffer in module.named_buffers(recurse=False):
        buffers_info.append({
            'name': name,
            'shape': tuple(buffer.shape),
            'numel': buffer.numel(),
            'dtype': str(buffer.dtype)
        })
    
    return params_info, buffers_info


def print_module_tree(module, prefix="", is_last=True, parent_name="", depth=0, max_depth=None):
    """
    递归打印模型的树状结构，显示每个模块的参数详情
    
    Args:
        module: PyTorch模块
        prefix: 打印前缀（用于树状结构）
        is_last: 是否是最后一个子模块
        parent_name: 父模块名称
        depth: 当前深度
        max_depth: 最大递归深度（None表示无限制）
    """
    
    if max_depth is not None and depth > max_depth:
        return
    
    # 获取当前模块的类型
    module_type = type(module).__name__
    
    # 获取当前模块的参数信息（不递归）
    params_info, buffers_info = get_module_parameters_detail(module)
    
    # 计算当前模块的直接参数总数
    direct_params_count = sum(p['numel'] for p in params_info)
    direct_buffers_count = sum(b['numel'] for b in buffers_info)
    
    # 计算包含子模块的总参数
    total_params = sum(p.numel() for p in module.parameters())
    total_buffers = sum(b.numel() for b in module.buffers())
    
    # 打印当前模块
    connector = "└─ " if is_last else "├─ "
    if depth == 0:
        print(f"\n{'='*120}")
        print(f"📦 模型: {module_type}")
        print(f"{'='*120}")
    else:
        print(f"{prefix}{connector}{parent_name}: {module_type}")
    
    # 打印参数统计
    if direct_params_count > 0 or len(params_info) > 0:
        param_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{param_prefix}")
        print(f"{param_prefix}【参数统计】")
        
        if direct_params_count > 0:
            print(f"{param_prefix}  直接参数: {direct_params_count:,} 个")
            
            # 按类型分组显示参数
            param_groups = defaultdict(list)
            for p in params_info:
                param_groups[p['name'].split('.')[-1]].append(p)
            
            for param_type, params in sorted(param_groups.items()):
                for p in params:
                    grad_status = "✓可训练" if p['requires_grad'] else "✗冻结"
                    print(f"{param_prefix}    • {p['name']}: {p['shape']} = {p['numel']:,} ({grad_status})")
        
        if total_params > direct_params_count:
            print(f"{param_prefix}  子模块参数: {total_params - direct_params_count:,} 个")
        
        print(f"{param_prefix}  总参数: {total_params:,} 个")
    
    # 打印缓冲区信息
    if len(buffers_info) > 0:
        param_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{param_prefix}【缓冲区】(非可训练)")
        for b in buffers_info:
            print(f"{param_prefix}  • {b['name']}: {b['shape']} = {b['numel']:,}")
    
    # 获取子模块
    children = list(module.named_children())
    
    if len(children) > 0:
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        for idx, (name, child) in enumerate(children):
            is_last_child = (idx == len(children) - 1)
            print_module_tree(
                child, 
                prefix=new_prefix, 
                is_last=is_last_child,
                parent_name=name,
                depth=depth + 1,
                max_depth=max_depth
            )


def analyze_model_structure(model, model_name="Model", max_depth=None):
    """
    分析模型结构的主函数
    
    Args:
        model: PyTorch模型实例
        model_name: 模型名称
        max_depth: 最大分析深度（None表示分析到最底层）
    """
    
    print("\n" + "█" * 120)
    print(f" " * 45 + f"🔍 {model_name} 结构分析")
    print("█" * 120)
    
    # 整体统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())
    
    print(f"\n【整体统计】")
    print(f"  总参数量:     {total_params:,}")
    print(f"  可训练参数:   {trainable_params:,}")
    print(f"  冻结参数:     {total_params - trainable_params:,}")
    print(f"  缓冲区元素:   {total_buffers:,}")
    print(f"  总模块数:     {len(list(model.modules()))}")
    
    # 参数类型分布统计
    print(f"\n【参数类型分布】")
    param_types = defaultdict(lambda: {'count': 0, 'numel': 0})
    
    for name, param in model.named_parameters():
        # 提取参数类型（如weight, bias等）
        param_type = name.split('.')[-1]
        param_types[param_type]['count'] += 1
        param_types[param_type]['numel'] += param.numel()
    
    for ptype, info in sorted(param_types.items(), key=lambda x: x[1]['numel'], reverse=True):
        print(f"  {ptype:20s}: {info['count']:3d} 个, {info['numel']:10,} 参数")
    
    # 打印详细的树状结构
    print(f"\n{'='*120}")
    print(f" " * 45 + "📊 详细层级结构")
    print(f"{'='*120}")
    
    if max_depth is not None:
        print(f"\n（显示深度限制为 {max_depth} 层）\n")
    else:
        print(f"\n（显示所有层级，深度不限）\n")
    
    print_module_tree(model, max_depth=max_depth)
    
    print("\n" + "=" * 120)
    print(f" " * 40 + "✓ 分析完成")
    print("=" * 120)


if __name__ == "__main__":
    from complexUnet import ComplexResidualUNet
    
    print("\n" + "█" * 120)
    print(" " * 40 + "测试配置 1: 小型网络")
    print("█" * 120)
    
    model1 = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=8,
        depth=2,
        attention_flag=True,
        activation='modrelu'
    )
    
    # 分析到所有层级（max_depth=None）
    analyze_model_structure(model1, "ComplexResidualUNet (depth=2, base=8)", max_depth=None)
    
    # # 可以限制深度，只看主要结构
    # print("\n\n" + "█" * 120)
    # print(" " * 40 + "主要结构概览（深度限制=2）")
    # print("█" * 120)
    # analyze_model_structure(model1, "ComplexResidualUNet 概览", max_depth=2)
