"""
测试循环卷积 vs 普通卷积的差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('.')

from complexUnet import ComplexConv1d, ComplexResidualUNet

def test_padding_behavior():
    """测试padding行为"""
    print("="*80)
    print("测试循环卷积 vs 普通卷积的padding行为")
    print("="*80)
    
    # 创建简单的输入
    batch_size = 1
    in_channels = 1
    seq_len = 8
    
    # 创建一个简单的复数序列
    x = torch.zeros(batch_size, in_channels, seq_len, dtype=torch.complex64)
    x[0, 0, :] = torch.arange(seq_len, dtype=torch.float32) + 1j * torch.arange(seq_len, dtype=torch.float32)
    
    print(f"\n输入序列 (长度={seq_len}):")
    print(f"  Real: {x[0, 0, :].real.tolist()}")
    print(f"  Imag: {x[0, 0, :].imag.tolist()}")
    
    # 测试循环卷积
    print("\n" + "-"*80)
    print("【循环卷积】circular=True (padding使用循环模式)")
    print("-"*80)
    conv_circular = ComplexConv1d(in_channels, 1, kernel_size=3, padding=1, circular=True)
    
    # 手动查看padding效果
    pad_left = 1
    pad_right = 1
    x_real_padded = F.pad(x.real, (pad_left, pad_right), mode='circular')
    x_imag_padded = F.pad(x.imag, (pad_left, pad_right), mode='circular')
    
    print(f"\nPadding后的序列 (长度={x_real_padded.shape[-1]}):")
    print(f"  Real: {x_real_padded[0, 0, :].tolist()}")
    print(f"  说明: 左边填充了最右边的值({x.real[0,0,-1].item():.0f}), 右边填充了最左边的值({x.real[0,0,0].item():.0f})")
    
    output_circular = conv_circular(x)
    print(f"\n输出形状: {output_circular.shape}")
    print(f"输出长度: {output_circular.shape[-1]} (与输入相同)")
    
    # 测试普通卷积
    print("\n" + "-"*80)
    print("【普通卷积】circular=False (padding填充0)")
    print("-"*80)
    conv_normal = ComplexConv1d(in_channels, 1, kernel_size=3, padding=1, circular=False)
    
    # 普通padding效果
    x_real_padded_zero = F.pad(x.real, (pad_left, pad_right), mode='constant', value=0)
    x_imag_padded_zero = F.pad(x.imag, (pad_left, pad_right), mode='constant', value=0)
    
    print(f"\nPadding后的序列 (长度={x_real_padded_zero.shape[-1]}):")
    print(f"  Real: {x_real_padded_zero[0, 0, :].tolist()}")
    print(f"  说明: 左边填充了0, 右边填充了0")
    
    output_normal = conv_normal(x)
    print(f"\n输出形状: {output_normal.shape}")
    print(f"输出长度: {output_normal.shape[-1]} (与输入相同)")
    
    # 比较差异
    print("\n" + "="*80)
    print("【关键差异】")
    print("="*80)
    print(f"循环卷积: padding使用序列的首尾元素（模拟周期性）")
    print(f"普通卷积: padding填充0（边界效应）")
    print(f"\n对于非周期信号，普通卷积更合适！")


def test_model_behavior():
    """测试整个模型的行为"""
    print("\n" + "="*80)
    print("测试ComplexResidualUNet的卷积模式")
    print("="*80)
    
    # 创建模型（循环卷积）
    print("\n创建模型 (circular=True - 循环卷积)...")
    model_circular = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=8,
        depth=2,
        circular=True
    )
    
    # 检查参数
    total_params = sum(p.numel() for p in model_circular.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  卷积模式: 循环卷积 (circular padding)")
    
    # 创建模型（普通卷积）
    print("\n创建模型 (circular=False - 普通卷积)...")
    model_normal = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=8,
        depth=2,
        circular=False
    )
    
    # 检查参数
    total_params = sum(p.numel() for p in model_normal.parameters())
    print(f"  总参数量: {total_params:,}")
    print(f"  卷积模式: 普通卷积 (zero padding)")
    
    # 测试前向传播
    batch_size = 2
    num_ports = 4
    seq_len = 12
    
    x = torch.randn(batch_size, num_ports, 2, seq_len, dtype=torch.complex64)
    
    print(f"\n输入形状: {x.shape}")
    
    with torch.no_grad():
        output_circular = model_circular(x)
        output_normal = model_normal(x)
    
    print(f"\n循环卷积输出形状: {output_circular.shape}")
    print(f"普通卷积输出形状: {output_normal.shape}")
    
    print("\n" + "="*80)
    print("✓ 两种模式都能正常工作")
    print("✓ 现在默认使用普通卷积 (circular=False)")
    print("✓ Padding填充0而不是循环填充")
    print("="*80)


if __name__ == "__main__":
    test_padding_behavior()
    test_model_behavior()
