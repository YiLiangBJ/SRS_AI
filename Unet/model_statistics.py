"""
Model Parameter Statistics for Complex U-Net
打印模型的详细参数统计信息
"""

import torch
from complexUnet import ComplexResidualUNet


def count_parameters(model, verbose=True):
    """
    统计模型参数
    
    Args:
        model: PyTorch 模型
        verbose: 是否打印详细信息
    
    Returns:
        dict: 参数统计信息
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # 按层统计
    layer_stats = []
    
    for name, parameter in model.named_parameters():
        param_count = parameter.numel()
        total_params += param_count
        
        if parameter.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
        
        layer_stats.append({
            'name': name,
            'shape': tuple(parameter.shape),
            'params': param_count,
            'trainable': parameter.requires_grad,
            'dtype': str(parameter.dtype)
        })
    
    stats = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'layer_stats': layer_stats
    }
    
    if verbose:
        print_parameter_stats(stats)
    
    return stats


def print_parameter_stats(stats):
    """打印参数统计信息"""
    
    print("=" * 100)
    print("MODEL PARAMETER STATISTICS")
    print("=" * 100)
    
    # 总体统计
    print("\n【Total Parameters】")
    print(f"  Total:          {stats['total_params']:>15,} parameters")
    print(f"  Trainable:      {stats['trainable_params']:>15,} parameters")
    print(f"  Non-trainable:  {stats['non_trainable_params']:>15,} parameters")
    
    # 按模块分组
    print("\n【Parameters by Module】")
    print(f"{'Module':<40} {'Shape':<25} {'Parameters':>15} {'Trainable':<10}")
    print("-" * 100)
    
    module_groups = {}
    for layer in stats['layer_stats']:
        # 提取模块名称（第一级）
        parts = layer['name'].split('.')
        if len(parts) > 1:
            module_name = parts[0]
        else:
            module_name = 'root'
        
        if module_name not in module_groups:
            module_groups[module_name] = []
        module_groups[module_name].append(layer)
    
    # 打印每个模块
    for module_name in sorted(module_groups.keys()):
        layers = module_groups[module_name]
        module_params = sum(l['params'] for l in layers)
        
        print(f"\n{module_name.upper()}: {module_params:,} parameters")
        print("-" * 100)
        
        for layer in layers:
            name = layer['name']
            shape_str = str(layer['shape'])
            params = layer['params']
            trainable = 'Yes' if layer['trainable'] else 'No'
            
            # 缩短名称显示
            if len(name) > 38:
                display_name = '...' + name[-35:]
            else:
                display_name = name
            
            print(f"  {display_name:<38} {shape_str:<25} {params:>13,}   {trainable:<10}")
    
    print("\n" + "=" * 100)


def compare_model_configs(configs):
    """
    比较不同配置的模型参数量
    
    Args:
        configs: list of dict, 每个dict包含模型配置
    """
    print("\n" + "=" * 100)
    print("MODEL CONFIGURATION COMPARISON")
    print("=" * 100)
    
    results = []
    
    for config in configs:
        model = ComplexResidualUNet(**config['params'])
        total_params = sum(p.numel() for p in model.parameters())
        
        results.append({
            'name': config['name'],
            'params': total_params,
            'config': config['params']
        })
    
    # 打印表格
    print(f"\n{'Configuration':<30} {'Parameters':>15} {'Config Details'}")
    print("-" * 100)
    
    for result in results:
        name = result['name']
        params = result['params']
        config = result['config']
        
        # 构建配置字符串
        config_str = f"depth={config.get('depth', 3)}, base_ch={config.get('base_channels', 32)}, "
        config_str += f"attn={config.get('attention_flag', False)}"
        
        print(f"{name:<30} {params:>13,}   {config_str}")
    
    print("\n" + "=" * 100)


def analyze_by_layer_type(model):
    """按层类型分析参数"""
    
    print("\n" + "=" * 100)
    print("PARAMETERS BY LAYER TYPE")
    print("=" * 100)
    
    type_stats = {}
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        
        if module_type not in type_stats:
            type_stats[module_type] = {
                'count': 0,
                'params': 0
            }
        
        type_stats[module_type]['count'] += 1
        
        # 只统计该模块的直接参数（不包括子模块）
        for param_name, param in module.named_parameters(recurse=False):
            type_stats[module_type]['params'] += param.numel()
    
    # 排序并打印
    sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['params'], reverse=True)
    
    print(f"\n{'Layer Type':<35} {'Count':>8} {'Parameters':>15} {'Avg per Layer':>15}")
    print("-" * 100)
    
    for layer_type, stats in sorted_types:
        if stats['params'] > 0:  # 只显示有参数的层
            count = stats['count']
            params = stats['params']
            avg_params = params / count if count > 0 else 0
            
            print(f"{layer_type:<35} {count:>8} {params:>13,} {avg_params:>13,.0f}")
    
    print("\n" + "=" * 100)


def get_model_memory_usage(model, input_shape, dtype=torch.float32):
    """
    估算模型的显存使用
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状 (batch, num_ports, channels, seq_len)
        dtype: 数据类型
    """
    print("\n" + "=" * 100)
    print("MEMORY USAGE ESTIMATION")
    print("=" * 100)
    
    # 参数内存
    param_memory = 0
    for param in model.parameters():
        # 复数参数占用2倍内存（实部+虚部）
        if param.dtype in [torch.complex64, torch.complex128]:
            param_memory += param.numel() * param.element_size()
        else:
            param_memory += param.numel() * param.element_size()
    
    # 梯度内存（训练时）
    grad_memory = param_memory  # 梯度与参数大小相同
    
    # 输入数据内存（粗略估计）
    batch_size = input_shape[0]
    input_memory = batch_size * input_shape[1] * input_shape[2] * input_shape[3] * 8  # complex64 = 8 bytes
    
    # 中间激活值内存（粗略估计，实际会更多）
    # 假设编码器每层保存skip connection
    estimated_activations = input_memory * 10  # 经验值
    
    # 总内存
    total_train_memory = param_memory + grad_memory + input_memory + estimated_activations
    total_inference_memory = param_memory + input_memory + estimated_activations
    
    print(f"\n{'Component':<30} {'Training':>20} {'Inference':>20}")
    print("-" * 100)
    print(f"{'Parameters':<30} {format_bytes(param_memory):>20} {format_bytes(param_memory):>20}")
    print(f"{'Gradients':<30} {format_bytes(grad_memory):>20} {'-':>20}")
    print(f"{'Input Data':<30} {format_bytes(input_memory):>20} {format_bytes(input_memory):>20}")
    print(f"{'Activations (estimated)':<30} {format_bytes(estimated_activations):>20} {format_bytes(estimated_activations):>20}")
    print("-" * 100)
    print(f"{'TOTAL (estimated)':<30} {format_bytes(total_train_memory):>20} {format_bytes(total_inference_memory):>20}")
    
    print("\n" + "=" * 100)


def format_bytes(bytes_value):
    """格式化字节数"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("COMPLEX U-NET PARAMETER ANALYSIS")
    print("=" * 100)
    
    # 创建默认模型
    print("\n【Creating Default Model】")
    model = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
        depth=3,
        attention_flag=True,
        activation='modrelu',
        circular=True
    )
    
    # 1. 详细参数统计
    stats = count_parameters(model, verbose=True)
    
    # 2. 按层类型分析
    analyze_by_layer_type(model)
    
    # 3. 显存使用估算
    input_shape = (8, 4, 2, 12)  # (batch, num_ports, channels, seq_len)
    get_model_memory_usage(model, input_shape)
    
    # 4. 比较不同配置
    configs = [
        {
            'name': 'Tiny (depth=2, base=16)',
            'params': {
                'input_channels': 2,
                'output_channels': 1,
                'base_channels': 16,
                'depth': 2,
                'attention_flag': False,
                'circular': True
            }
        },
        {
            'name': 'Small (depth=3, base=16)',
            'params': {
                'input_channels': 2,
                'output_channels': 1,
                'base_channels': 16,
                'depth': 3,
                'attention_flag': False,
                'circular': True
            }
        },
        {
            'name': 'Default (depth=3, base=32)',
            'params': {
                'input_channels': 2,
                'output_channels': 1,
                'base_channels': 32,
                'depth': 3,
                'attention_flag': True,
                'circular': True
            }
        },
        {
            'name': 'Large (depth=4, base=32)',
            'params': {
                'input_channels': 2,
                'output_channels': 1,
                'base_channels': 32,
                'depth': 4,
                'attention_flag': True,
                'circular': True
            }
        },
        {
            'name': 'XLarge (depth=4, base=64)',
            'params': {
                'input_channels': 2,
                'output_channels': 1,
                'base_channels': 64,
                'depth': 4,
                'attention_flag': True,
                'circular': True
            }
        }
    ]
    
    compare_model_configs(configs)
    
    # 5. 总结
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nDefault Configuration:")
    print(f"  - Total Parameters:     {stats['total_params']:,}")
    print(f"  - Trainable Parameters: {stats['trainable_params']:,}")
    print(f"  - Input Shape:          (batch=8, ports=4, channels=2, seq_len=12)")
    print(f"  - Output Shape:         (batch=8, ports=4, channels=1, seq_len=12)")
    print(f"  - Network Depth:        3")
    print(f"  - Base Channels:        32")
    print(f"  - Attention:            Enabled")
    print(f"  - Circular Conv:        Enabled")
    print("\n" + "=" * 100)
