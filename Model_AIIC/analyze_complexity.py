"""
模型参数量与计算复杂度分析

功能:
1. 统计不同超参数组合下的模型参数量
2. 计算推理时的 FLOPs 和 MACs
3. 估算内存使用
4. 生成对比报告

用法:
    python Model_AIIC/analyze_complexity.py
    python Model_AIIC/analyze_complexity.py --stages "2,3,4" --share_weights "True,False"
"""

import os
import sys
import torch
import numpy as np
import argparse
from pathlib import Path
import json
from collections import OrderedDict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC.channel_separator import ResidualRefinementSeparator


def count_parameters(model):
    """
    统计模型参数量
    
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
        param_dict: 各层参数详情
    """
    total_params = 0
    trainable_params = 0
    param_dict = OrderedDict()
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        
        param_dict[name] = {
            'shape': list(param.shape),
            'numel': num_params,
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        }
    
    return total_params, trainable_params, param_dict


def count_complex_linear_flops(in_features, out_features, batch_size=1):
    """
    计算复数线性层的 FLOPs
    
    复数矩阵乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    每个复数乘法需要: 4次实数乘法 + 2次实数加法
    
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        batch_size: 批大小
        
    Returns:
        flops: 浮点运算次数
        macs: 乘加运算次数
        real_muls: 实数乘法次数
        real_adds: 实数加法次数
    """
    # 每个输出元素需要 in_features 次复数乘加
    # 复数乘法: 4 real_mul + 2 real_add
    # 复数加法: 2 real_add
    
    num_outputs = batch_size * out_features
    
    # 乘法
    real_muls = num_outputs * in_features * 4  # 每个复数MAC的乘法
    
    # 加法
    # 1. 复数乘法内部的加法: num_outputs * in_features * 2
    # 2. 累加 in_features 个复数结果: num_outputs * (in_features - 1) * 2
    real_adds = num_outputs * in_features * 2 + num_outputs * (in_features - 1) * 2
    
    # FLOPs = 乘法 + 加法
    flops = real_muls + real_adds
    
    # MACs (Multiply-Accumulate)
    macs = num_outputs * in_features * 2  # 复数MAC = 2个实数MAC
    
    return flops, macs, real_muls, real_adds


def analyze_model_complexity(model, seq_len=12, num_ports=4, batch_size=1):
    """
    分析模型计算复杂度
    
    Returns:
        complexity: 复杂度统计字典
    """
    complexity = {
        'input_shape': (batch_size, seq_len),
        'output_shape': (batch_size, num_ports, seq_len),
        'batch_size': batch_size,
        'total_flops': 0,
        'total_macs': 0,
        'total_real_muls': 0,
        'total_real_adds': 0,
        'layer_details': []
    }
    
    num_stages = model.num_stages
    share_weights = model.share_weights_across_stages
    
    # 分析每个阶段和每个端口的 MLP
    for stage_idx in range(num_stages):
        for port_idx in range(num_ports):
            # 获取对应的 MLP
            if share_weights:
                # 共享权重：每个 port 有一个 MLP，所有 stage 共享
                mlp = model.port_mlps[port_idx]
            else:
                # 不共享：每个 port 每个 stage 独立
                mlp = model.port_mlps[port_idx][stage_idx]
            
            # 分析 MLP 层
            # mlp 包含 mlp_real 和 mlp_imag，结构相同
            # 每个都是: Linear(seq_len*2, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, seq_len)
            
            stage_flops = 0
            stage_macs = 0
            stage_muls = 0
            stage_adds = 0
            
            # Layer 1: Linear(seq_len*2, hidden_dim)
            in_features = seq_len * 2
            out_features = 64  # hidden_dim 固定为 64
            # 实部和虚部各一个
            for part in ['real', 'imag']:
                # Linear: out = in @ weight^T + bias
                # FLOPs = batch * out_features * (2 * in_features)
                # (每个输出: in_features 乘法 + in_features-1 加法 + 1 偏置加法 ≈ 2*in_features)
                muls = batch_size * out_features * in_features
                adds = batch_size * out_features * in_features  # 包括累加和偏置
                stage_muls += muls
                stage_adds += adds
                stage_flops += muls + adds
                stage_macs += batch_size * out_features * in_features
            
            # ReLU: 只是比较，计算量可忽略（或计数为 batch * out_features 次比较）
            
            # Layer 2: Linear(hidden_dim, hidden_dim)
            in_features = 64
            out_features = 64
            for part in ['real', 'imag']:
                muls = batch_size * out_features * in_features
                adds = batch_size * out_features * in_features
                stage_muls += muls
                stage_adds += adds
                stage_flops += muls + adds
                stage_macs += batch_size * out_features * in_features
            
            # Layer 3: Linear(hidden_dim, seq_len)
            in_features = 64
            out_features = seq_len
            for part in ['real', 'imag']:
                muls = batch_size * out_features * in_features
                adds = batch_size * out_features * in_features
                stage_muls += muls
                stage_adds += adds
                stage_flops += muls + adds
                stage_macs += batch_size * out_features * in_features
            
            # 记录这个 MLP 的复杂度
            mlp_name = f"port{port_idx}_stage{stage_idx}"
            if share_weights and stage_idx > 0:
                mlp_name += " (shared weights)"
            
            complexity['layer_details'].append({
                'stage': stage_idx,
                'port': port_idx,
                'layer': mlp_name,
                'type': 'ComplexMLP',
                'flops': stage_flops,
                'macs': stage_macs,
                'real_muls': stage_muls,
                'real_adds': stage_adds
            })
            
            # 计算量：无论是否共享权重，每个 stage 都要计算
            # 权重共享只影响参数量，不影响计算量
            complexity['total_flops'] += stage_flops
            complexity['total_macs'] += stage_macs
            complexity['total_real_muls'] += stage_muls
            complexity['total_real_adds'] += stage_adds
        
        # Residual connection: 每个 stage 结束后的残差加法
        # h_out = h_in - sum(h_ports) + h_in
        # 每个 port 的输出相加: (num_ports-1) * batch * seq_len * 2 (real+imag)
        # 然后与输入相减再相加: 2 * batch * seq_len * 2
        residual_adds = batch_size * num_ports * seq_len * 2  # 简化估计
        complexity['total_real_adds'] += residual_adds
        complexity['total_flops'] += residual_adds
    
    return complexity


def estimate_memory(model, seq_len=12, num_ports=4, batch_size=1, dtype_size=8):
    """
    估算推理时的内存使用
    
    Args:
        dtype_size: 每个复数元素的字节数 (complex64 = 8 bytes)
        
    Returns:
        memory_dict: 内存使用详情 (bytes)
    """
    # 模型参数内存
    param_memory = sum(p.numel() * dtype_size for p in model.parameters())
    
    # 输入/输出内存
    input_memory = batch_size * seq_len * dtype_size
    output_memory = batch_size * num_ports * seq_len * dtype_size
    
    # 中间激活值内存（粗略估计）
    # 每个阶段需要存储中间结果
    num_stages = model.num_stages
    hidden_dim = 64  # 固定的 hidden_dim
    
    # 假设每个阶段的最大中间张量
    activation_per_stage = batch_size * num_ports * max(seq_len, hidden_dim) * dtype_size
    total_activation = activation_per_stage * num_stages
    
    memory_dict = {
        'parameters': param_memory,
        'input': input_memory,
        'output': output_memory,
        'activations': total_activation,
        'total': param_memory + input_memory + output_memory + total_activation
    }
    
    return memory_dict


def format_number(num):
    """格式化数字（K, M, G）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.0f}"


def format_bytes(num_bytes):
    """格式化字节数（KB, MB, GB）"""
    if num_bytes >= 1024**3:
        return f"{num_bytes/1024**3:.2f} GB"
    elif num_bytes >= 1024**2:
        return f"{num_bytes/1024**2:.2f} MB"
    elif num_bytes >= 1024:
        return f"{num_bytes/1024:.2f} KB"
    else:
        return f"{num_bytes:.0f} B"


def main():
    parser = argparse.ArgumentParser(description='分析模型参数量和计算复杂度')
    
    parser.add_argument('--stages', type=str, default='2,3,4',
                       help='要分析的阶段数（逗号分隔）')
    parser.add_argument('--share_weights', type=str, default='True,False',
                       help='是否共享权重（逗号分隔）')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='推理批大小（用于计算 FLOPs）')
    parser.add_argument('--output', type=str, default='./model_complexity_analysis',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 解析参数
    stages_list = [int(x) for x in args.stages.split(',')]
    share_weights_list = [x.lower() == 'true' for x in args.share_weights.split(',')]
    
    print("="*80)
    print("模型复杂度分析")
    print("="*80)
    print(f"阶段数: {stages_list}")
    print(f"共享权重: {share_weights_list}")
    print(f"推理批大小: {args.batch_size}")
    print("="*80)
    
    # 固定参数
    seq_len = 12
    num_ports = 4
    hidden_dim = 64
    
    # 分析所有组合
    results = []
    
    for num_stages in stages_list:
        for share_weights in share_weights_list:
            print(f"\n{'='*80}")
            print(f"配置: stages={num_stages}, share_weights={share_weights}")
            print(f"{'='*80}")
            
            # 创建模型
            model = ResidualRefinementSeparator(
                seq_len=seq_len,
                num_ports=num_ports,
                hidden_dim=hidden_dim,
                num_stages=num_stages,
                share_weights_across_stages=share_weights,
                normalize_energy=True
            )
            model.eval()
            
            # 统计参数
            total_params, trainable_params, param_dict = count_parameters(model)
            
            print(f"\n📊 参数统计:")
            print(f"  总参数量: {format_number(total_params)} ({total_params:,})")
            print(f"  可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
            print(f"  参数内存: {format_bytes(total_params * 8)}")  # complex64 = 8 bytes
            
            # 分析计算复杂度
            complexity = analyze_model_complexity(model, seq_len, num_ports, args.batch_size)
            
            print(f"\n⚡ 计算复杂度 (batch_size={args.batch_size}):")
            print(f"  总 FLOPs: {format_number(complexity['total_flops'])} ({complexity['total_flops']:,})")
            print(f"  总 MACs: {format_number(complexity['total_macs'])} ({complexity['total_macs']:,})")
            print(f"  实数乘法: {format_number(complexity['total_real_muls'])} ({complexity['total_real_muls']:,})")
            print(f"  实数加法: {format_number(complexity['total_real_adds'])} ({complexity['total_real_adds']:,})")
            
            # 估算内存
            memory = estimate_memory(model, seq_len, num_ports, args.batch_size)
            
            print(f"\n💾 内存估算 (batch_size={args.batch_size}):")
            print(f"  参数内存: {format_bytes(memory['parameters'])}")
            print(f"  输入内存: {format_bytes(memory['input'])}")
            print(f"  输出内存: {format_bytes(memory['output'])}")
            print(f"  中间激活: {format_bytes(memory['activations'])}")
            print(f"  总内存: {format_bytes(memory['total'])}")
            
            # 保存结果
            result = {
                'config': {
                    'num_stages': num_stages,
                    'share_weights': share_weights,
                    'seq_len': seq_len,
                    'num_ports': num_ports,
                    'hidden_dim': hidden_dim
                },
                'parameters': {
                    'total': total_params,
                    'trainable': trainable_params,
                    'memory_bytes': total_params * 8,
                    'details': param_dict
                },
                'complexity': complexity,
                'memory': memory
            }
            
            results.append(result)
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 格式
    json_path = output_dir / 'complexity_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 保存 JSON: {json_path}")
    
    # 生成对比表格（Markdown）
    md_path = output_dir / 'complexity_comparison.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 模型复杂度对比\n\n")
        f.write(f"**分析日期**: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**批大小**: {args.batch_size}\n\n")
        f.write("---\n\n")
        
        f.write("## 参数量对比\n\n")
        f.write("| 阶段数 | 共享权重 | 总参数量 | 参数内存 |\n")
        f.write("|--------|----------|----------|----------|\n")
        for r in results:
            cfg = r['config']
            params = r['parameters']
            f.write(f"| {cfg['num_stages']} | {cfg['share_weights']} | "
                   f"{format_number(params['total'])} | "
                   f"{format_bytes(params['memory_bytes'])} |\n")
        f.write("\n")
        
        f.write("## 计算复杂度对比\n\n")
        f.write("| 阶段数 | 共享权重 | FLOPs | MACs | 实数乘法 | 实数加法 |\n")
        f.write("|--------|----------|-------|------|----------|----------|\n")
        for r in results:
            cfg = r['config']
            comp = r['complexity']
            f.write(f"| {cfg['num_stages']} | {cfg['share_weights']} | "
                   f"{format_number(comp['total_flops'])} | "
                   f"{format_number(comp['total_macs'])} | "
                   f"{format_number(comp['total_real_muls'])} | "
                   f"{format_number(comp['total_real_adds'])} |\n")
        f.write("\n")
        
        f.write("## 内存使用对比\n\n")
        f.write("| 阶段数 | 共享权重 | 参数 | 输入 | 输出 | 激活 | 总计 |\n")
        f.write("|--------|----------|------|------|------|------|------|\n")
        for r in results:
            cfg = r['config']
            mem = r['memory']
            f.write(f"| {cfg['num_stages']} | {cfg['share_weights']} | "
                   f"{format_bytes(mem['parameters'])} | "
                   f"{format_bytes(mem['input'])} | "
                   f"{format_bytes(mem['output'])} | "
                   f"{format_bytes(mem['activations'])} | "
                   f"{format_bytes(mem['total'])} |\n")
        f.write("\n")
        
        f.write("---\n\n")
        f.write("## 详细说明\n\n")
        f.write("### FLOPs (Floating Point Operations)\n")
        f.write("- 浮点运算总数（乘法 + 加法）\n")
        f.write("- 复数运算已拆分为实数运算\n")
        f.write("- 包含前向传播的所有计算\n\n")
        
        f.write("### MACs (Multiply-Accumulate Operations)\n")
        f.write("- 乘加操作数\n")
        f.write("- 1 MAC = 1 乘法 + 1 加法\n")
        f.write("- FLOPs ≈ 2 × MACs（对于矩阵乘法）\n\n")
        
        f.write("### 复数运算\n")
        f.write("- 复数乘法: (a+bi)(c+di) = (ac-bd) + (ad+bc)i\n")
        f.write("  - 需要: 4次实数乘法 + 2次实数加法\n")
        f.write("- 复数加法: (a+bi) + (c+di) = (a+c) + (b+d)i\n")
        f.write("  - 需要: 2次实数加法\n\n")
        
        f.write("### 内存估算\n")
        f.write("- 参数内存: 模型权重占用的内存\n")
        f.write("- 输入/输出内存: 单个batch的输入输出\n")
        f.write("- 激活内存: 推理时的中间结果（粗略估计）\n")
        f.write("- complex64: 每个复数占用 8 字节\n\n")
    
    print(f"✓ 保存 Markdown: {md_path}")
    
    # 打印摘要
    print(f"\n{'='*80}")
    print("分析摘要")
    print(f"{'='*80}")
    print(f"配置数量: {len(results)}")
    
    # 找出最小和最大参数量
    min_params = min(results, key=lambda x: x['parameters']['total'])
    max_params = max(results, key=lambda x: x['parameters']['total'])
    
    print(f"\n最少参数: stages={min_params['config']['num_stages']}, "
          f"share={min_params['config']['share_weights']}, "
          f"{format_number(min_params['parameters']['total'])}")
    print(f"最多参数: stages={max_params['config']['num_stages']}, "
          f"share={max_params['config']['share_weights']}, "
          f"{format_number(max_params['parameters']['total'])}")
    
    # 找出最小和最大计算量
    min_flops = min(results, key=lambda x: x['complexity']['total_flops'])
    max_flops = max(results, key=lambda x: x['complexity']['total_flops'])
    
    print(f"\n最少 FLOPs: stages={min_flops['config']['num_stages']}, "
          f"share={min_flops['config']['share_weights']}, "
          f"{format_number(min_flops['complexity']['total_flops'])}")
    print(f"最多 FLOPs: stages={max_flops['config']['num_stages']}, "
          f"share={max_flops['config']['share_weights']}, "
          f"{format_number(max_flops['complexity']['total_flops'])}")
    
    print(f"\n结果保存到: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
