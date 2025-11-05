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


def print_residual_block_details(in_ch, out_ch, use_attention=False, reduction=16, block_name="", model=None):
    """Print detailed parameter calculation for one ComplexResidualBlock"""
    
    print(f"  │   ")
    print(f"  │   【详细参数计算】")
    
    # conv1
    conv1_params = count_complex_conv_params(in_ch, out_ch, 3, bias=False)
    print(f"  │   ├─ conv1: ComplexConv1d(in={in_ch}, out={out_ch}, kernel=3)")
    print(f"  │   │   = 2(复数) × in_ch × out_ch × kernel")
    print(f"  │   │   = 2(复数) × {in_ch}(输入通道) × {out_ch}(输出通道) × 3(卷积核大小)")
    print(f"  │   │   = {conv1_params}")
    
    # 从模型中获取实际参数
    if model is not None and block_name:
        try:
            conv1_weight = getattr(model, block_name).conv1.conv_real.weight
            conv1_actual = conv1_weight.numel() * 2  # 实部+虚部
            print(f"  │   │   【实际模型】{block_name}.conv1: weight_re={conv1_weight.numel()}, weight_im={conv1_weight.numel()}, 总计={conv1_actual}")
            if conv1_actual != conv1_params:
                print(f"  │   │   ⚠️  差异: 计算={conv1_params}, 实际={conv1_actual}")
        except:
            pass
    
    # bn1
    bn1_params = count_complex_bn_params(out_ch)
    print(f"  │   ├─ bn1: ComplexBatchNorm1d({out_ch})")
    print(f"  │   │   = 4(实部γ+β,虚部γ+β) × features")
    print(f"  │   │   = 4(两组参数) × {out_ch}(通道数)")
    print(f"  │   │   = {bn1_params}")
    
    if model is not None and block_name:
        try:
            bn1_weight_re = getattr(model, block_name).bn1.weight_re
            bn1_bias_re = getattr(model, block_name).bn1.bias_re
            bn1_weight_im = getattr(model, block_name).bn1.weight_im
            bn1_bias_im = getattr(model, block_name).bn1.bias_im
            bn1_actual = bn1_weight_re.numel() + bn1_bias_re.numel() + bn1_weight_im.numel() + bn1_bias_im.numel()
            print(f"  │   │   【实际模型】{block_name}.bn1: γ_re={bn1_weight_re.numel()}, β_re={bn1_bias_re.numel()}, γ_im={bn1_weight_im.numel()}, β_im={bn1_bias_im.numel()}, 总计={bn1_actual}")
            if bn1_actual != bn1_params:
                print(f"  │   │   ⚠️  差异: 计算={bn1_params}, 实际={bn1_actual}")
        except:
            pass
    
    # modReLU1
    modrelu1_params = count_modrelu_params(out_ch)
    print(f"  │   ├─ modReLU1: ComplexModReLU({out_ch})")
    print(f"  │   │   = {modrelu1_params}(通道数) (每通道1个bias参数)")
    
    if model is not None and block_name:
        try:
            modrelu1_bias = getattr(model, block_name).activation1.bias
            modrelu1_actual = modrelu1_bias.numel()
            print(f"  │   │   【实际模型】{block_name}.activation1: bias={modrelu1_actual}")
            if modrelu1_actual != modrelu1_params:
                print(f"  │   │   ⚠️  差异: 计算={modrelu1_params}, 实际={modrelu1_actual}")
        except:
            pass
    
    # conv2
    conv2_params = count_complex_conv_params(out_ch, out_ch, 3, bias=False)
    print(f"  │   ├─ conv2: ComplexConv1d(in={out_ch}, out={out_ch}, kernel=3)")
    print(f"  │   │   = 2(复数) × in_ch × out_ch × kernel")
    print(f"  │   │   = 2(复数) × {out_ch}(输入通道) × {out_ch}(输出通道) × 3(卷积核大小)")
    print(f"  │   │   = {conv2_params}")
    
    if model is not None and block_name:
        try:
            conv2_weight = getattr(model, block_name).conv2.conv_real.weight
            conv2_actual = conv2_weight.numel() * 2
            print(f"  │   │   【实际模型】{block_name}.conv2: weight_re={conv2_weight.numel()}, weight_im={conv2_weight.numel()}, 总计={conv2_actual}")
            if conv2_actual != conv2_params:
                print(f"  │   │   ⚠️  差异: 计算={conv2_params}, 实际={conv2_actual}")
        except:
            pass
    
    # bn2
    bn2_params = count_complex_bn_params(out_ch)
    print(f"  │   ├─ bn2: ComplexBatchNorm1d({out_ch})")
    print(f"  │   │   = 4(实部γ+β,虚部γ+β) × {out_ch}(通道数)")
    print(f"  │   │   = {bn2_params}")
    
    if model is not None and block_name:
        try:
            bn2_weight_re = getattr(model, block_name).bn2.weight_re
            bn2_bias_re = getattr(model, block_name).bn2.bias_re
            bn2_weight_im = getattr(model, block_name).bn2.weight_im
            bn2_bias_im = getattr(model, block_name).bn2.bias_im
            bn2_actual = bn2_weight_re.numel() + bn2_bias_re.numel() + bn2_weight_im.numel() + bn2_bias_im.numel()
            print(f"  │   │   【实际模型】{block_name}.bn2: γ_re={bn2_weight_re.numel()}, β_re={bn2_bias_re.numel()}, γ_im={bn2_weight_im.numel()}, β_im={bn2_bias_im.numel()}, 总计={bn2_actual}")
            if bn2_actual != bn2_params:
                print(f"  │   │   ⚠️  差异: 计算={bn2_params}, 实际={bn2_actual}")
        except:
            pass
    
    total = conv1_params + bn1_params + modrelu1_params + conv2_params + bn2_params
    
    # attention
    if use_attention:
        reduced_ch = max(1, out_ch // reduction)
        att1_params = count_complex_conv_params(out_ch, reduced_ch, 1, bias=False)
        att2_params = count_complex_conv_params(reduced_ch, out_ch, 1, bias=False)
        att_total = att1_params + att2_params
        print(f"  │   ├─ attention: ComplexAttention(ch={out_ch}, reduction={reduction})")
        print(f"  │   │   reduced_ch = max(1, {out_ch}//{reduction}) = {reduced_ch}")
        print(f"  │   │   fc1: 2(复数) × {out_ch}(输入) × {reduced_ch}(降维后) × 1(1x1卷积) = {att1_params}")
        print(f"  │   │   fc2: 2(复数) × {reduced_ch}(降维后) × {out_ch}(恢复) × 1(1x1卷积) = {att2_params}")
        print(f"  │   │   小计 = {att1_params} + {att2_params} = {att_total}")
        
        if model is not None and block_name:
            try:
                att_module = getattr(model, block_name).attention
                att_fc0_weight = att_module.fc[0].conv_real.weight
                att_fc2_weight = att_module.fc[2].conv_real.weight
                att_actual = (att_fc0_weight.numel() + att_fc2_weight.numel()) * 2
                print(f"  │   │   【实际模型】{block_name}.attention: fc[0]={att_fc0_weight.numel()*2}, fc[2]={att_fc2_weight.numel()*2}, 总计={att_actual}")
                if att_actual != att_total:
                    print(f"  │   │   ⚠️  差异: 计算={att_total}, 实际={att_actual}")
            except:
                pass
        
        total += att_total
    
    # shortcut
    if in_ch != out_ch:
        shortcut_params = count_complex_conv_params(in_ch, out_ch, 1, bias=False)
        print(f"  │   ├─ shortcut: ComplexConv1d(in={in_ch}, out={out_ch}, kernel=1)")
        print(f"  │   │   = 2(复数) × {in_ch}(输入通道) × {out_ch}(输出通道) × 1(1x1投影)")
        print(f"  │   │   = {shortcut_params}")
        
        if model is not None and block_name:
            try:
                shortcut_weight = getattr(model, block_name).shortcut.conv_real.weight
                shortcut_actual = shortcut_weight.numel() * 2
                print(f"  │   │   【实际模型】{block_name}.shortcut: weight_re={shortcut_weight.numel()}, weight_im={shortcut_weight.numel()}, 总计={shortcut_actual}")
                if shortcut_actual != shortcut_params:
                    print(f"  │   │   ⚠️  差异: 计算={shortcut_params}, 实际={shortcut_actual}")
            except:
                pass
        
        total += shortcut_params
    
    # modReLU2
    modrelu2_params = count_modrelu_params(out_ch)
    print(f"  │   └─ modReLU2: ComplexModReLU({out_ch})")
    print(f"  │       = {modrelu2_params}(通道数) (每通道1个bias参数)")
    
    if model is not None and block_name:
        try:
            modrelu2_bias = getattr(model, block_name).activation2.bias
            modrelu2_actual = modrelu2_bias.numel()
            print(f"  │       【实际模型】{block_name}.activation2: bias={modrelu2_actual}")
            if modrelu2_actual != modrelu2_params:
                print(f"  │       ⚠️  差异: 计算={modrelu2_params}, 实际={modrelu2_actual}")
        except:
            pass
    
    total += modrelu2_params
    
    print(f"  │   ")
    print(f"  │   总计 = {conv1_params}(conv1) + {bn1_params}(bn1) + {modrelu1_params}(modReLU1) + {conv2_params}(conv2) + {bn2_params}(bn2)", end="")
    if use_attention:
        print(f" + {att_total}(attention)", end="")
    if in_ch != out_ch:
        print(f" + {shortcut_params}(shortcut)", end="")
    print(f" + {modrelu2_params}(modReLU2)")
    print(f"  │        = {total}")
    
    return total



def analyze_model_params(input_channels=2, output_channels=1, base_channels=32, 
                        depth=3, attention_flag=False, activation='modrelu'):
    """Analyze model parameters in detail - 直接基于模型结构分析"""
    
    print("=" * 120)
    print(" " * 35 + "ComplexResidualUNet 参数量详细分析")
    print(f" 配置: input_channels={input_channels}, output_channels={output_channels}, base_channels={base_channels}, depth={depth}, attention={attention_flag}")
    print("=" * 120)
    
    # 创建模型 - 所有分析都基于这个实际模型
    model = ComplexResidualUNet(
        input_channels=input_channels,
        output_channels=output_channels,
        base_channels=base_channels,
        depth=depth,
        attention_flag=attention_flag,
        activation=activation
    )
    
    print(f"\n【模型结构概览】")
    print(f"  编码器层数: {len(model.enc_blocks)}")
    print(f"  下采样层数: {len(model.down_samples)}")
    print(f"  解码器层数: {len(model.dec_blocks)}")
    print(f"  上采样层数: {len(model.up_samples)}")
    print(f"  瓶颈层: {'有' if hasattr(model, 'bottleneck') else '无'}")
    print(f"  输出层: {'有' if hasattr(model, 'final_conv') else '无'}")
    
    total_params = 0
    calculated_params = 0
    
    # ==================== 编码器 - 直接遍历模型的 enc_blocks ====================
    print("\n" + "━" * 120)
    print("【编码器 Encoder】")
    print("━" * 120)
    
    for i, enc_block in enumerate(model.enc_blocks):
        # 从实际模型获取通道信息
        in_ch = enc_block.conv1.conv_real.weight.shape[1]  # 输入通道
        out_ch = enc_block.conv1.conv_real.weight.shape[0]  # 输出通道
        
        print(f"\n  ├─ enc_blocks[{i}]: ComplexResidualBlock")
        print(f"  │   维度:  (B, C={in_ch:>3}, L) → (B, C={out_ch:>3}, L)")
        
        # 计算并验证
        has_attention = hasattr(enc_block, 'attention') and enc_block.attention is not None
        block_params = print_residual_block_details(in_ch, out_ch, has_attention, 
                                                    block_name=f"enc_blocks.{i}", model=model)
        
        # 实际参数
        actual_block_params = sum(p.numel() for p in enc_block.parameters())
        print(f"  │   【验证】计算={block_params:,}, 实际={actual_block_params:,}, ", end="")
        if block_params == actual_block_params:
            print("✓ 匹配")
        else:
            print(f"✗ 差异={actual_block_params - block_params}")
        
        calculated_params += block_params
        total_params += actual_block_params
        
        # 下采样层 - 检查是否存在对应的下采样
        if i < len(model.down_samples):
            down_sample = model.down_samples[i]
            down_in_ch = down_sample.conv_real.weight.shape[1]
            down_out_ch = down_sample.conv_real.weight.shape[0]
            
            print(f"  │")
            print(f"  ├─ down_samples[{i}]: ComplexConv1d(stride=2)")
            print(f"  │   维度:  (B, C={down_out_ch:>3}, L) → (B, C={down_out_ch:>3}, L//2)")
            
            # 计算参数
            has_bias = down_sample.conv_real.bias is not None
            down_params_calc = count_complex_conv_params(down_in_ch, down_out_ch, 2, bias=has_bias)
            
            weight_params = 2 * down_in_ch * down_out_ch * 2
            bias_params = 2 * down_out_ch if has_bias else 0
            print(f"  │   = 2(复数) × in_ch × out_ch × kernel + 2(复数) × out_ch(偏置)")
            print(f"  │   = 2(复数) × {down_in_ch}(输入通道) × {down_out_ch}(输出通道) × 2(卷积核) + 2(复数) × {down_out_ch}(偏置)")
            print(f"  │   = {weight_params}(权重) + {bias_params}(偏置) = {down_params_calc}")
            
            # 实际参数
            down_params_actual = sum(p.numel() for p in down_sample.parameters())
            print(f"  │   【验证】计算={down_params_calc:,}, 实际={down_params_actual:,}, ", end="")
            if down_params_calc == down_params_actual:
                print("✓ 匹配")
            else:
                print(f"✗ 差异={down_params_actual - down_params_calc}")
            
            calculated_params += down_params_calc
            total_params += down_params_actual
    
    # ==================== 瓶颈层 - 直接从模型获取 ====================
    print("\n" + "━" * 120)
    print("【瓶颈层 Bottleneck】")
    print("━" * 120)
    
    if hasattr(model, 'bottleneck'):
        bottleneck = model.bottleneck
        in_ch = bottleneck.conv1.conv_real.weight.shape[1]
        out_ch = bottleneck.conv1.conv_real.weight.shape[0]
        
        print(f"\n  bottleneck: ComplexResidualBlock")
        print(f"    维度:  (B, C={in_ch:>3}, L) → (B, C={out_ch:>3}, L)")
        
        # 计算参数
        has_attention = hasattr(bottleneck, 'attention') and bottleneck.attention is not None
        bottleneck_params_calc = count_residual_block_params(in_ch, out_ch, has_attention)
        
        # 实际参数
        bottleneck_params_actual = sum(p.numel() for p in bottleneck.parameters())
        
        print(f"    【验证】计算={bottleneck_params_calc:,}, 实际={bottleneck_params_actual:,}, ", end="")
        if bottleneck_params_calc == bottleneck_params_actual:
            print("✓ 匹配")
        else:
            print(f"✗ 差异={bottleneck_params_actual - bottleneck_params_calc}")
        
        calculated_params += bottleneck_params_calc
        total_params += bottleneck_params_actual
    
    # ==================== 解码器 - 直接遍历模型的 dec_blocks ====================
    print("\n" + "━" * 120)
    print("【解码器 Decoder】")
    print("━" * 120)
    
    for i, dec_block in enumerate(model.dec_blocks):
        # 上采样层
        if i < len(model.up_samples):
            up_sample = model.up_samples[i]
            up_in_ch = up_sample.conv_real.weight.shape[1]
            up_out_ch = up_sample.conv_real.weight.shape[0]
            
            print(f"\n  ├─ up_samples[{i}]: ComplexConvTranspose1d(stride=2)")
            print(f"  │   维度:  (B, C={up_in_ch:>3}, L) → (B, C={up_out_ch:>3}, L*2)")
            
            # 计算参数
            up_params_calc = 2 * (up_in_ch * up_out_ch * 2)
            print(f"  │   = 2(复数) × in_ch × out_ch × kernel")
            print(f"  │   = 2(复数) × {up_in_ch}(输入通道) × {up_out_ch}(输出通道) × 2(转置卷积核)")
            print(f"  │   = {up_params_calc}")
            
            # 实际参数
            up_params_actual = sum(p.numel() for p in up_sample.parameters())
            print(f"  │   【验证】计算={up_params_calc:,}, 实际={up_params_actual:,}, ", end="")
            if up_params_calc == up_params_actual:
                print("✓ 匹配")
            else:
                print(f"✗ 差异={up_params_actual - up_params_calc}")
            
            calculated_params += up_params_calc
            total_params += up_params_actual
            
            print(f"  │")
        
        # 解码器块
        dec_in_ch = dec_block.conv1.conv_real.weight.shape[1]
        dec_out_ch = dec_block.conv1.conv_real.weight.shape[0]
        
        print(f"  ├─ dec_blocks[{i}]: ComplexResidualBlock + Skip Connection")
        print(f"  │   拼接后输入: (B, C={dec_in_ch:>3}, L) → 输出: (B, C={dec_out_ch:>3}, L)")
        
        # 计算并验证
        has_attention = hasattr(dec_block, 'attention') and dec_block.attention is not None
        dec_params_calc = print_residual_block_details(dec_in_ch, dec_out_ch, has_attention,
                                                      block_name=f"dec_blocks.{i}", model=model)
        
        # 实际参数
        dec_params_actual = sum(p.numel() for p in dec_block.parameters())
        print(f"  │   【验证】计算={dec_params_calc:,}, 实际={dec_params_actual:,}, ", end="")
        if dec_params_calc == dec_params_actual:
            print("✓ 匹配")
        else:
            print(f"✗ 差异={dec_params_actual - dec_params_calc}")
        
        calculated_params += dec_params_calc
        total_params += dec_params_actual
    
    # ==================== 输出层 - 直接从模型获取 ====================
    print("\n" + "━" * 120)
    print("【输出层 Output】")
    print("━" * 120)
    
    if hasattr(model, 'final_conv'):
        final_conv = model.final_conv
        final_in_ch = final_conv.conv_real.weight.shape[1]
        final_out_ch = final_conv.conv_real.weight.shape[0]
        
        print(f"\n  final_conv: ComplexConv1d(kernel_size=1)")
        print(f"    维度:  (B, C={final_in_ch:>3}, L) → (B, C={final_out_ch:>3}, L)")
        
        # 计算参数
        has_bias = final_conv.conv_real.bias is not None
        final_params_calc = count_complex_conv_params(final_in_ch, final_out_ch, 1, bias=has_bias)
        
        weight_params = 2 * final_in_ch * final_out_ch * 1
        bias_params = 2 * final_out_ch if has_bias else 0
        print(f"    = 2(复数) × in_ch × out_ch × kernel + 2(复数) × out_ch(偏置)")
        print(f"    = 2(复数) × {final_in_ch}(输入通道) × {final_out_ch}(输出通道) × 1(1x1卷积) + 2(复数) × {final_out_ch}(偏置)")
        print(f"    = {weight_params}(权重) + {bias_params}(偏置) = {final_params_calc}")
        
        # 实际参数
        final_params_actual = sum(p.numel() for p in final_conv.parameters())
        print(f"    【验证】计算={final_params_calc:,}, 实际={final_params_actual:,}, ", end="")
        if final_params_calc == final_params_actual:
            print("✓ 匹配")
        else:
            print(f"✗ 差异={final_params_actual - final_params_calc}")
        
        calculated_params += final_params_calc
        total_params += final_params_actual
    
    # ==================== 总结 ====================
    print("\n" + "=" * 120)
    print("【参数统计汇总】")
    print("=" * 120)
    
    print(f"\n  【基于模型结构统计】")
    print(f"  实际模型总参数量:   {total_params:,}")
    print(f"\n  【基于公式计算】")
    print(f"  计算得到的总参数量: {calculated_params:,}")
    
    print("\n" + "=" * 120)
    if calculated_params == total_params:
        print(f"✓✓✓ 验证通过！计算参数量 = 实际参数量 = {total_params:,}")
    else:
        diff = total_params - calculated_params
        print(f"✗✗✗ 验证失败！差异: {diff} 个参数")
        print(f"    计算值: {calculated_params:,}")
        print(f"    实际值: {total_params:,}")
    print("=" * 120)
    
    print("\n【维度符号说明】")
    print("  B = Batch (批次大小)")
    print("  C = Channels (通道数)")
    print("  L = Length (序列长度)")
    print("=" * 120)
    
    return calculated_params, total_params


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



