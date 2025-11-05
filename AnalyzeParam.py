"""
ComplexResidualUNet Parameter Analysis Tool
Automatically adapts to different configuration parameters
"""
# -*- coding: utf-8 -*-

import sys
import io
import torch
import torch.nn as nn
from complexUnet import ComplexResidualUNet

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def count_complex_conv_params(in_ch, out_ch, kernel_size, bias=True):
    """Calculate parameters for complex convolution"""
    weight_params = 2 * (in_ch * out_ch * kernel_size)
    bias_params = 2 * out_ch if bias else 0
    return weight_params + bias_params


def count_complex_bn_params(num_features):
    """Calculate parameters for complex batch normalization"""
    return 4 * num_features


def count_modrelu_params(num_features):
    """Calculate parameters for modReLU"""
    return num_features


def count_residual_block_params(in_ch, out_ch, use_attention=False, reduction=16):
    """Calculate parameters for one ComplexResidualBlock"""
    params = 0
    
    # conv1: in_ch -> out_ch
    params += count_complex_conv_params(in_ch, out_ch, kernel_size=3, bias=False)
    # bn1
    params += count_complex_bn_params(out_ch)
    # modReLU1
    params += count_modrelu_params(out_ch)
    # conv2: out_ch -> out_ch
    params += count_complex_conv_params(out_ch, out_ch, kernel_size=3, bias=False)
    # bn2
    params += count_complex_bn_params(out_ch)
    
    # attention (optional)
    if use_attention:
        reduced_ch = max(1, out_ch // reduction)
        params += count_complex_conv_params(out_ch, reduced_ch, kernel_size=1, bias=False)
        params += count_complex_conv_params(reduced_ch, out_ch, kernel_size=1, bias=False)
    
    # shortcut (if channels differ)
    if in_ch != out_ch:
        params += count_complex_conv_params(in_ch, out_ch, kernel_size=1, bias=False)
    
    # modReLU2
    params += count_modrelu_params(out_ch)
    
    return params


def print_residual_block_details(in_ch, out_ch, use_attention=False, reduction=16):
    """Print detailed parameter calculation for one ComplexResidualBlock"""
    
    print(f"  │   ")
    print(f"  │   【详细参数计算】")
    
    # conv1
    conv1_params = count_complex_conv_params(in_ch, out_ch, 3, bias=False)
    print(f"  │   ├─ conv1: ComplexConv1d(in={in_ch}, out={out_ch}, kernel=3)")
    print(f"  │   │   = 2(复数) × in_ch × out_ch × kernel")
    print(f"  │   │   = 2 × {in_ch} × {out_ch} × 3 = {conv1_params}")
    
    # bn1
    bn1_params = count_complex_bn_params(out_ch)
    print(f"  │   ├─ bn1: ComplexBatchNorm1d({out_ch})")
    print(f"  │   │   = 4(实部γ+β,虚部γ+β) × features")
    print(f"  │   │   = 4 × {out_ch} = {bn1_params}")
    
    # modReLU1
    modrelu1_params = count_modrelu_params(out_ch)
    print(f"  │   ├─ modReLU1: ComplexModReLU({out_ch})")
    print(f"  │   │   = {modrelu1_params} (每通道1个bias)")
    
    # conv2
    conv2_params = count_complex_conv_params(out_ch, out_ch, 3, bias=False)
    print(f"  │   ├─ conv2: ComplexConv1d(in={out_ch}, out={out_ch}, kernel=3)")
    print(f"  │   │   = 2 × {out_ch} × {out_ch} × 3 = {conv2_params}")
    
    # bn2
    bn2_params = count_complex_bn_params(out_ch)
    print(f"  │   ├─ bn2: ComplexBatchNorm1d({out_ch})")
    print(f"  │   │   = 4 × {out_ch} = {bn2_params}")
    
    total = conv1_params + bn1_params + modrelu1_params + conv2_params + bn2_params
    
    # attention
    if use_attention:
        reduced_ch = max(1, out_ch // reduction)
        att1_params = count_complex_conv_params(out_ch, reduced_ch, 1, bias=False)
        att2_params = count_complex_conv_params(reduced_ch, out_ch, 1, bias=False)
        att_total = att1_params + att2_params
        print(f"  │   ├─ attention: ComplexAttention(ch={out_ch}, reduction={reduction})")
        print(f"  │   │   reduced_ch = max(1, {out_ch}//{reduction}) = {reduced_ch}")
        print(f"  │   │   fc1: 2×{out_ch}×{reduced_ch}×1 = {att1_params}")
        print(f"  │   │   fc2: 2×{reduced_ch}×{out_ch}×1 = {att2_params}")
        print(f"  │   │   小计 = {att_total}")
        total += att_total
    
    # shortcut
    if in_ch != out_ch:
        shortcut_params = count_complex_conv_params(in_ch, out_ch, 1, bias=False)
        print(f"  │   ├─ shortcut: ComplexConv1d(in={in_ch}, out={out_ch}, kernel=1)")
        print(f"  │   │   = 2 × {in_ch} × {out_ch} × 1 = {shortcut_params}")
        total += shortcut_params
    
    # modReLU2
    modrelu2_params = count_modrelu_params(out_ch)
    print(f"  │   └─ modReLU2: ComplexModReLU({out_ch})")
    print(f"  │       = {modrelu2_params}")
    total += modrelu2_params
    
    print(f"  │   ")
    print(f"  │   总计 = {conv1_params} + {bn1_params} + {modrelu1_params} + {conv2_params} + {bn2_params}", end="")
    if use_attention:
        print(f" + {att_total}", end="")
    if in_ch != out_ch:
        print(f" + {shortcut_params}", end="")
    print(f" + {modrelu2_params} = {total}")
    
    return total



def analyze_model_params(input_channels=2, output_channels=1, base_channels=32, 
                        depth=3, attention_flag=False, activation='modrelu'):
    """Analyze model parameters in detail"""
    
    print("=" * 120)
    print(" " * 35 + "ComplexResidualUNet 参数量详细分析")
    print(f" 配置: input_channels={input_channels}, output_channels={output_channels}, base_channels={base_channels}, depth={depth}, attention={attention_flag}")
    print("=" * 120)
    
    total_params = 0
    
    # Calculate channel progression
    encoder_channels = []
    for i in range(depth):
        ch = min(base_channels * (2 ** i), 256)
        encoder_channels.append(ch)
    
    bottleneck_in = encoder_channels[-1]
    bottleneck_out = min(bottleneck_in * 2, 256)
    
    # ==================== Encoder ====================
    print("\n" + "━" * 120)
    print("【编码器 Encoder】")
    print("━" * 120)
    
    for i in range(depth):
        in_ch = input_channels if i == 0 else encoder_channels[i-1]
        out_ch = encoder_channels[i]
        
        print(f"\n  ├─ enc_blocks[{i}]: ComplexResidualBlock")
        print(f"  │   维度:  (B, C={in_ch:>3}, L) → (B, C={out_ch:>3}, L)")
        
        # 使用详细打印函数
        block_params = print_residual_block_details(in_ch, out_ch, attention_flag)
        print(f"  │   参数量: {block_params:,}")
        total_params += block_params
        
        if i < depth:
            print(f"  │")
            print(f"  ├─ down_samples[{i}]: ComplexConv1d(stride=2)")
            print(f"  │   维度:  (B, C={out_ch:>3}, L) → (B, C={out_ch:>3}, L//2)")
            down_params = count_complex_conv_params(out_ch, out_ch, 2, bias=True)
            print(f"  │   计算:  2×{out_ch}×{out_ch}×2(权重) + 2×{out_ch}(偏置) = {down_params:,}")
            print(f"  │   参数量: {down_params:,}")
            total_params += down_params
    
    # ==================== Bottleneck ====================
    print("\n" + "━" * 120)
    print("【瓶颈层 Bottleneck】")
    print("━" * 120)
    
    print(f"\n  bottleneck: ComplexResidualBlock")
    print(f"    维度:  (B, C={bottleneck_in:>3}, L) → (B, C={bottleneck_out:>3}, L)")
    
    bottleneck_params = count_residual_block_params(bottleneck_in, bottleneck_out, attention_flag)
    print(f"    参数量: {bottleneck_params:,}")
    total_params += bottleneck_params
    
    # ==================== Decoder ====================
    print("\n" + "━" * 120)
    print("【解码器 Decoder】")
    print("━" * 120)
    
    for i in range(depth):
        idx = depth - 1 - i
        
        # Up-sample
        if i == 0:
            up_in_ch = bottleneck_out
        else:
            up_in_ch = encoder_channels[idx + 1]
        up_out_ch = encoder_channels[idx]
        
        print(f"\n  ├─ up_samples[{i}]: ComplexConvTranspose1d(stride=2)")
        print(f"  │   维度:  (B, C={up_in_ch:>3}, L) → (B, C={up_out_ch:>3}, L*2)")
        up_params = 2 * (up_in_ch * up_out_ch * 2)
        print(f"  │   计算:  2(复数) × {up_in_ch} × {up_out_ch} × 2(kernel) = {up_params:,}")
        print(f"  │   参数量: {up_params:,}")
        total_params += up_params
        
        print(f"  │")
        
        # Decoder block
        skip_ch = encoder_channels[idx]
        dec_in_ch = up_out_ch + skip_ch
        dec_out_ch = up_out_ch
        
        print(f"  ├─ dec_blocks[{i}]: ComplexResidualBlock + Skip Connection")
        print(f"  │   上采样: (B, C={up_out_ch:>3}, L)")
        print(f"  │   跳跃连接: (B, C={skip_ch:>3}, L) ← 来自 enc_blocks[{idx}]")
        print(f"  │   拼接后: (B, C={dec_in_ch:>3}, L) → 输出: (B, C={dec_out_ch:>3}, L)")
        
        dec_params = print_residual_block_details(dec_in_ch, dec_out_ch, attention_flag)
        print(f"  │   参数量: {dec_params:,}")
        total_params += dec_params
    
    # ==================== Output Layer ====================
    print("\n" + "━" * 120)
    print("【输出层 Output】")
    print("━" * 120)
    
    print(f"\n  final_conv: ComplexConv1d(kernel_size=1)")
    print(f"    维度:  (B, C={encoder_channels[0]:>3}, L) → (B, C={output_channels:>3}, L)")
    
    final_params = count_complex_conv_params(encoder_channels[0], output_channels, 1, bias=True)
    weight_params = 2 * encoder_channels[0] * output_channels * 1
    bias_params = 2 * output_channels
    print(f"    计算:  2×{encoder_channels[0]}×{output_channels}×1(权重) + 2×{output_channels}(偏置) = {weight_params} + {bias_params} = {final_params}")
    print(f"    参数量: {final_params}")
    total_params += final_params
    
    # ==================== Summary ====================
    print("\n" + "=" * 120)
    print("【参数统计汇总】")
    print("=" * 120)
    
    print(f"\n  计算得到的总参数量: {total_params:,}")
    
    # Verification
    print(f"\n【模型验证】")
    model = ComplexResidualUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        depth=depth,
        attention_flag=attention_flag,
        activation=activation
    )
    
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  实际模型参数量:     {actual_params:,}")
    
    print("\n" + "=" * 120)
    if total_params == actual_params:
        print(f"✓✓✓ 验证通过！计算参数量 = 实际参数量 = {total_params:,}")
    else:
        diff = actual_params - total_params
        print(f"✗✗✗ 验证失败！差异: {diff} 个参数")
        print(f"    计算值: {total_params:,}")
        print(f"    实际值: {actual_params:,}")
    print("=" * 120)
    
    print("\n【维度符号说明】")
    print("  B = Batch (批次大小)")
    print("  C = Channels (通道数)")
    print("  L = Length (序列长度)")
    print("=" * 120)
    
    return total_params, actual_params


if __name__ == "__main__":
    results = []
    
    # Test configuration 1
    print("\n" + "█" * 120)
    print(" " * 50 + "配置 1")
    print("█" * 120)
    calc1, actual1 = analyze_model_params(input_channels=2, output_channels=1, base_channels=8, 
                        depth=2, attention_flag=True, activation='modrelu')
    results.append(("配置1: depth=2, base=8, attention=True", calc1, actual1))
    
    # # Test configuration 2
    # print("\n\n" + "█" * 120)
    # print(" " * 50 + "配置 2")
    # print("█" * 120)
    # calc2, actual2 = analyze_model_params(input_channels=2, output_channels=1, base_channels=32, 
    #                     depth=3, attention_flag=True, activation='modrelu')
    # results.append(("配置2: depth=3, base=32, attention=True", calc2, actual2))
    
    # # Test configuration 3
    # print("\n\n" + "█" * 120)
    # print(" " * 50 + "配置 3")
    # print("█" * 120)
    # calc3, actual3 = analyze_model_params(input_channels=2, output_channels=1, base_channels=16, 
    #                     depth=3, attention_flag=False, activation='modrelu')
    # results.append(("配置3: depth=3, base=16, attention=False", calc3, actual3))
    
    # Final summary
    print("\n\n" + "█" * 120)
    print("【最终验证汇总】")
    print("█" * 120)
    print()
    print(f"{'配置':<60} {'计算参数量':>15} {'实际参数量':>15}   {'验证结果':>10}")
    print("─" * 120)
    
    all_pass = True
    for config_name, calc, actual in results:
        status = "✓ 通过" if calc == actual else "✗ 失败"
        if calc != actual:
            all_pass = False
        print(f"{config_name:<55} {calc:>15,} {actual:>15,}   {status:>10}")
    
    print("─" * 120)
    print()
    
    if all_pass:
        print("✓✓✓ 所有配置验证通过！")
    else:
        print("⚠️  部分配置验证失败，请检查计算逻辑。")
    
    print()
    print("█" * 120)



