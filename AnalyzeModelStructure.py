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


def infer_tensor_shape_meaning(shape, module_type, param_name=""):
    """
    推断张量形状每个维度的含义
    
    Args:
        shape: 张量形状 tuple
        module_type: 模块类型名称
        param_name: 参数名称（weight, bias等）
    
    Returns:
        带标注的形状字符串
    """
    if not shape or len(shape) == 0:
        return "()"
    
    # 参数张量的维度含义
    if param_name:
        if 'Conv1d' in module_type or 'ConvTranspose1d' in module_type:
            if param_name == 'weight':
                if len(shape) == 3:
                    return f"({shape[0]}, {shape[1]}, {shape[2]})  # (out_channels, in_channels, kernel_size)"
                elif len(shape) == 1:
                    return f"({shape[0]},)  # (out_channels,)"
            elif param_name == 'bias':
                return f"({shape[0]},)  # (out_channels,)" if len(shape) == 1 else str(tuple(shape))
        
        elif 'Conv2d' in module_type or 'ConvTranspose2d' in module_type:
            if param_name == 'weight' and len(shape) == 4:
                return f"({shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})  # (out_ch, in_ch, kH, kW)"
            elif param_name == 'bias' and len(shape) == 1:
                return f"({shape[0]},)  # (out_channels,)"
        
        elif 'BatchNorm' in module_type or 'LayerNorm' in module_type:
            if param_name in ['weight', 'bias'] and len(shape) == 1:
                return f"({shape[0]},)  # (num_features,)"
        
        elif 'Linear' in module_type:
            if param_name == 'weight' and len(shape) == 2:
                return f"({shape[0]}, {shape[1]})  # (out_features, in_features)"
            elif param_name == 'bias' and len(shape) == 1:
                return f"({shape[0]},)  # (out_features,)"
        
        elif 'Embedding' in module_type:
            if param_name == 'weight' and len(shape) == 2:
                return f"({shape[0]}, {shape[1]})  # (num_embeddings, embedding_dim)"
        
        elif 'ModReLU' in module_type or 'ReLU' in module_type:
            if param_name == 'bias' and len(shape) == 1:
                return f"({shape[0]},)  # (num_features,)"
    
    # 对于任何一维bias，如果还没处理，尝试通用标注
    if param_name == 'bias' and len(shape) == 1:
        return f"({shape[0]},)  # (num_features,)"
    
    # 默认返回原始形状
    return str(tuple(shape))


def get_module_io_shape_info(module, module_type):
    """
    尝试推断模块的输入输出形状信息
    
    Returns:
        dict with 'input_shape', 'output_shape', 'shape_note'
    """
    info = {
        'input_shape': None,
        'output_shape': None,
        'shape_note': None
    }
    
    try:
        # 从模块属性推断
        if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
            # Conv layers
            in_ch = module.in_channels
            out_ch = module.out_channels
            if 'Conv1d' in module_type or 'ConvTranspose1d' in module_type:
                info['input_shape'] = f"(B, {in_ch}, L)"
                info['output_shape'] = f"(B, {out_ch}, L')"
                info['shape_note'] = "B=batch, L=length"
            elif 'Conv2d' in module_type or 'ConvTranspose2d' in module_type:
                info['input_shape'] = f"(B, {in_ch}, H, W)"
                info['output_shape'] = f"(B, {out_ch}, H', W')"
                info['shape_note'] = "B=batch, H=height, W=width"
        
        elif hasattr(module, 'num_features'):
            # BatchNorm layers
            nf = module.num_features
            if 'BatchNorm1d' in module_type:
                info['input_shape'] = f"(B, {nf}, L)"
                info['output_shape'] = f"(B, {nf}, L)"
                info['shape_note'] = "B=batch, L=length"
            elif 'BatchNorm2d' in module_type:
                info['input_shape'] = f"(B, {nf}, H, W)"
                info['output_shape'] = f"(B, {nf}, H, W)"
                info['shape_note'] = "B=batch, H=height, W=width"
        
        elif hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            # Linear layers
            in_f = module.in_features
            out_f = module.out_features
            info['input_shape'] = f"(B, {in_f})"
            info['output_shape'] = f"(B, {out_f})"
            info['shape_note'] = "B=batch"
        
        elif hasattr(module, 'num_embeddings') and hasattr(module, 'embedding_dim'):
            # Embedding layers
            num_emb = module.num_embeddings
            emb_dim = module.embedding_dim
            info['input_shape'] = f"(B, L)"
            info['output_shape'] = f"(B, L, {emb_dim})"
            info['shape_note'] = "B=batch, L=sequence_length"
        
        elif 'Pool' in module_type:
            # Pooling layers
            if 'AdaptiveAvgPool1d' in module_type:
                info['input_shape'] = "(B, C, L)"
                info['output_shape'] = f"(B, C, {module.output_size})"
                info['shape_note'] = "B=batch, C=channels, L=length"
            elif 'Pool1d' in module_type:
                info['input_shape'] = "(B, C, L)"
                info['output_shape'] = "(B, C, L')"
                info['shape_note'] = "B=batch, C=channels"
            elif 'Pool2d' in module_type:
                info['input_shape'] = "(B, C, H, W)"
                info['output_shape'] = "(B, C, H', W')"
                info['shape_note'] = "B=batch, C=channels"
        
        elif 'ModReLU' in module_type:
            # Complex activation - 尝试从bias推断特征数 (必须在ReLU之前检查！)
            num_features = None
            if hasattr(module, 'bias'):
                if isinstance(module.bias, torch.nn.Parameter):
                    num_features = module.bias.shape[0]
                elif module.bias is not None and hasattr(module.bias, 'shape'):
                    num_features = module.bias.shape[0]
            
            if num_features is not None:
                info['input_shape'] = f"(B, {num_features}, L)"
                info['output_shape'] = f"(B, {num_features}, L)"
                info['shape_note'] = f"B=batch, {num_features} features, L=length (complex tensor)"
            else:
                info['input_shape'] = "(*, ...)"
                info['output_shape'] = "(*, ...)"
                info['shape_note'] = "complex tensor, shape unchanged"
        
        elif 'ReLU' in module_type or 'Dropout' in module_type or 'Identity' in module_type:
            # Element-wise operations
            info['input_shape'] = "(*, ...)"
            info['output_shape'] = "(*, ...)"
            info['shape_note'] = "element-wise operation, shape unchanged"
    
    except:
        pass
    
    return info


def print_module_tree(module, prefix="", is_last=True, parent_name="", depth=0, max_depth=None, show_forward_order=True):
    """
    递归打印模型的树状结构，显示每个模块的参数详情
    
    Args:
        module: PyTorch模块
        prefix: 打印前缀（用于树状结构）
        is_last: 是否是最后一个子模块
        parent_name: 父模块名称
        depth: 当前深度
        max_depth: 最大递归深度（None表示无限制）
        show_forward_order: 是否尝试显示forward执行顺序
    """
    
    if max_depth is not None and depth > max_depth:
        return
    
    # 获取当前模块的类型
    module_type = type(module).__name__
    
    # 获取输入输出形状信息
    io_info = get_module_io_shape_info(module, module_type)
    
    # 获取当前模块的参数信息（不递归）
    params_info, buffers_info = [], []
    
    # 分析参数
    for name, param in module.named_parameters(recurse=False):
        param_info = analyze_parameter(name, param)
        param_type = name.split('.')[-1]
        param_info['shape_with_meaning'] = infer_tensor_shape_meaning(
            param.shape, module_type, param_type
        )
        params_info.append(param_info)
    
    # 分析缓冲区
    for name, buffer in module.named_buffers(recurse=False):
        buffers_info.append({
            'name': name,
            'shape': tuple(buffer.shape),
            'numel': buffer.numel(),
            'dtype': str(buffer.dtype)
        })
    
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
    
    # 显示输入输出形状信息
    if io_info['input_shape'] or io_info['output_shape']:
        param_prefix = prefix + ("    " if is_last else "│   ")
        print(f"{param_prefix}")
        print(f"{param_prefix}【张量形状 Tensor Shapes】")
        if io_info['input_shape']:
            print(f"{param_prefix}  Input:  {io_info['input_shape']}")
        if io_info['output_shape']:
            print(f"{param_prefix}  Output: {io_info['output_shape']}")
        if io_info['shape_note']:
            print(f"{param_prefix}  说明: {io_info['shape_note']}")
    
    # 尝试提取forward执行顺序（如果有）
    if show_forward_order and depth > 0:
        forward_order = extract_forward_order(module)
        if forward_order:
            param_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{param_prefix}")
            print(f"{param_prefix}【执行顺序 Forward Flow】")
            for idx, step in enumerate(forward_order, 1):
                print(f"{param_prefix}  {idx}. {step}")
    
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
                    # 使用带维度含义的形状
                    shape_str = p.get('shape_with_meaning', str(p['shape']))
                    print(f"{param_prefix}    • {p['name']}: {shape_str} = {p['numel']:,} ({grad_status})")
        
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
                max_depth=max_depth,
                show_forward_order=show_forward_order
            )


def extract_forward_order(module):
    """
    尝试从模块的forward方法源码中提取执行顺序
    返回执行步骤列表，如果无法提取则返回None
    """
    try:
        import inspect
        
        # 检查是否有forward方法
        if not hasattr(module, 'forward'):
            return None
        
        # 获取forward方法源码（设置超时保护）
        try:
            source = inspect.getsource(module.forward)
        except (OSError, TypeError):
            # 无法获取源码（可能是内置模块或C扩展）
            return None
        
        lines = source.split('\n')
        
        # 解析执行顺序
        steps = []
        for line in lines:
            line = line.strip()
            
            # 跳过定义行、注释、空行、return语句
            if (not line or 
                line.startswith('def ') or 
                line.startswith('#') or 
                line.startswith('return') or
                line.startswith('"""') or
                line.startswith("'''")):
                continue
            
            # 提取有用的执行语句
            if '=' in line and 'self.' in line:
                # 提取模块调用，如: out = self.conv1(x)
                parts = line.split('=')
                if len(parts) >= 2:
                    left = parts[0].strip()
                    right = parts[1].strip()
                    
                    # 提取模块名称
                    if 'self.' in right:
                        import re
                        matches = re.findall(r'self\.(\w+)', right)
                        if matches:
                            module_name = matches[0]
                            steps.append(f"{left} = self.{module_name}(...)")
        
        # 只有在有实质内容时才返回
        if len(steps) > 2:  # 至少要有几个步骤才有意义
            return steps
        
    except Exception as e:
        # 任何异常都静默处理
        pass
    
    return None

def analyze_model_structure(model, model_name="Model", max_depth=None, show_forward_order=True, output_file=None):
    """
    分析模型结构的主函数
    
    Args:
        model: PyTorch模型实例
        model_name: 模型名称
        max_depth: 最大分析深度（None表示分析到最底层）
        show_forward_order: 是否尝试显示forward执行顺序
        output_file: 输出文件路径（None表示输出到终端）
    """
    
    # 如果指定了输出文件，重定向输出
    original_stdout = None
    file_handle = None
    
    if output_file:
        import os
        # 确保目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 打开文件并重定向stdout
        original_stdout = sys.stdout
        file_handle = open(output_file, 'w', encoding='utf-8')
        sys.stdout = file_handle
        
        print(f"# 模型结构分析报告")
        print(f"# 生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"# 模型名称: {model_name}")
        print()
    
    try:
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
        
        print(f"  {'类型':<20s}  {'张量数量':>10s}  {'参数总数':>12s}  {'占比':>8s}")
        print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*8}")
        
        for ptype, info in sorted(param_types.items(), key=lambda x: x[1]['numel'], reverse=True):
            percentage = (info['numel'] / total_params * 100) if total_params > 0 else 0
            print(f"  {ptype:<20s}  {info['count']:>10,d}  {info['numel']:>12,d}  {percentage:>7.1f}%")
        
        print(f"  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*8}")
        print(f"  {'总计':<20s}  {sum(p['count'] for p in param_types.values()):>10,d}  {total_params:>12,d}  {'100.0%':>8s}")
        print()
        print(f"  💡 说明：")
        print(f"     - 张量数量：该类型参数在模型中出现的次数（如有80个名为'weight'的参数张量）")
        print(f"     - 参数总数：这些张量包含的标量参数总和（如这80个weight张量共包含24,976个标量）")
        
        # 打印详细的树状结构
        print(f"\n{'='*120}")
        print(f" " * 45 + "📊 详细层级结构")
        print(f"{'='*120}")
        
        if max_depth is not None:
            print(f"\n（显示深度限制为 {max_depth} 层）\n")
        else:
            print(f"\n（显示所有层级，深度不限）\n")
        
        if show_forward_order:
            print(f"💡 【执行顺序 Forward Flow】显示模块在forward()中的调用顺序\n")
        
        print_module_tree(model, max_depth=max_depth, show_forward_order=show_forward_order)
        
        print("\n" + "=" * 120)
        print(f" " * 40 + "✓ 分析完成")
        print("=" * 120)
        
        if output_file:
            print(f"\n📄 报告已保存到: {output_file}")
    
    finally:
        # 恢复stdout并关闭文件
        if original_stdout:
            sys.stdout = original_stdout
            if file_handle:
                file_handle.close()
            print(f"\n✓ 分析完成！报告已保存到: {output_file}")
            print(f"  文件大小: {os.path.getsize(output_file):,} 字节")


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
    
    # 分析到所有层级并保存到文件
    output_path = "model_structure_analysis.txt"
    print(f"\n开始分析模型结构...")
    print(f"输出文件: {output_path}")
    
    analyze_model_structure(
        model1, 
        "ComplexResidualUNet (depth=2, base=8)", 
        max_depth=None,
        output_file=output_path
    )
    
    # 也可以生成一个简化版本（限制深度）
    output_path_short = "model_structure_summary.txt"
    print(f"\n生成简化版本...")
    print(f"输出文件: {output_path_short}")
    
    analyze_model_structure(
        model1, 
        "ComplexResidualUNet - 概览", 
        max_depth=2,
        output_file=output_path_short
    )
    
    print("\n" + "="*120)
    print("✓ 完成！生成了以下报告文件：")
    print(f"  1. {output_path} - 完整详细分析（所有层级）")
    print(f"  2. {output_path_short} - 简化概览（depth=2）")
    print("="*120)
