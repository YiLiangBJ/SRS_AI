"""
Test Circular Convolution in ComplexConv1d
验证循环卷积的正确性
"""

import torch
import torch.nn.functional as F
from complexUnet import ComplexConv1d

def test_circular_padding():
    """测试循环填充是否正确"""
    print("=" * 80)
    print("Test 1: Circular Padding")
    print("=" * 80)
    
    # 创建一个简单的序列 [0, 1, 2, ..., 11]
    x = torch.arange(12, dtype=torch.float32).view(1, 1, 12)
    print(f"\nOriginal sequence: {x[0, 0].tolist()}")
    
    # 测试 kernel_size=3 的循环填充
    kernel_size = 3
    pad_total = kernel_size - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    x_padded = F.pad(x, (pad_left, pad_right), mode='circular')
    print(f"\nCircular padded (k=3): {x_padded[0, 0].tolist()}")
    print(f"Expected: [11.0, 0.0, 1.0, 2.0, ..., 11.0, 0.0]")
    
    # 验证边界
    assert x_padded[0, 0, 0] == 11.0, "Left padding should wrap to last element"
    assert x_padded[0, 0, -1] == 0.0, "Right padding should wrap to first element"
    print("✓ Circular padding correct!")
    
    return True

def test_circular_conv_length():
    """测试循环卷积是否保持长度不变"""
    print("\n" + "=" * 80)
    print("Test 2: Sequence Length Preservation")
    print("=" * 80)
    
    batch_size = 2
    in_channels = 3
    out_channels = 16
    seq_len = 12
    
    # 创建测试输入
    x = torch.randn(batch_size, in_channels, seq_len, dtype=torch.complex64)
    print(f"\nInput shape: {x.shape}")
    
    # 测试循环卷积
    conv_circular = ComplexConv1d(in_channels, out_channels, kernel_size=3, circular=True)
    y_circular = conv_circular(x)
    print(f"Output shape (circular=True): {y_circular.shape}")
    
    # 测试普通卷积
    conv_normal = ComplexConv1d(in_channels, out_channels, kernel_size=3, padding=1, circular=False)
    y_normal = conv_normal(x)
    print(f"Output shape (circular=False, padding=1): {y_normal.shape}")
    
    # 验证长度
    assert y_circular.shape[-1] == seq_len, f"Circular conv should preserve length: {y_circular.shape[-1]} != {seq_len}"
    assert y_normal.shape[-1] == seq_len, f"Normal conv with padding should preserve length: {y_normal.shape[-1]} != {seq_len}"
    
    print("✓ Both convolutions preserve sequence length!")
    
    return True

def test_boundary_values():
    """测试边界值的卷积是否正确使用循环"""
    print("\n" + "=" * 80)
    print("Test 3: Boundary Convolution with Circular Padding")
    print("=" * 80)
    
    # 创建一个可识别的序列
    seq_len = 12
    x = torch.arange(seq_len, dtype=torch.float32).view(1, 1, seq_len)
    
    # 转换为复数
    x_complex = torch.complex(x, torch.zeros_like(x))
    
    print(f"\nInput sequence (real part): {x[0, 0].tolist()}")
    
    # 创建一个简单的卷积核 [1, 1, 1] (平均)
    conv = ComplexConv1d(1, 1, kernel_size=3, circular=True)
    
    # 手动设置权重为 [1, 1, 1] / 3
    with torch.no_grad():
        conv.conv_real.weight.fill_(1/3)
        conv.conv_imag.weight.fill_(1/3)
        if conv.conv_real.bias is not None:
            conv.conv_real.bias.fill_(0)
            conv.conv_imag.bias.fill_(0)
    
    # 进行卷积
    y = conv(x_complex)
    y_real = y.real[0, 0]
    
    print(f"\nOutput (real part): {y_real.tolist()}")
    
    # 检查第一个位置：应该是 (11 + 0 + 1) / 3
    expected_first = (11 + 0 + 1) / 3
    actual_first = y_real[0].item()
    print(f"\nFirst position:")
    print(f"  Expected (11+0+1)/3 = {expected_first:.4f}")
    print(f"  Actual = {actual_first:.4f}")
    
    # 检查最后一个位置：应该是 (10 + 11 + 0) / 3
    expected_last = (10 + 11 + 0) / 3
    actual_last = y_real[-1].item()
    print(f"\nLast position:")
    print(f"  Expected (10+11+0)/3 = {expected_last:.4f}")
    print(f"  Actual = {actual_last:.4f}")
    
    # 验证（允许小误差）
    assert abs(actual_first - expected_first) < 0.01, "First position incorrect!"
    assert abs(actual_last - expected_last) < 0.01, "Last position incorrect!"
    
    print("\n✓ Boundary convolution using circular padding correctly!")
    
    return True

def test_full_unet():
    """测试完整的 U-Net 与循环卷积"""
    print("\n" + "=" * 80)
    print("Test 4: Full U-Net with Circular Convolution")
    print("=" * 80)
    
    from complexUnet import ComplexResidualUNet, create_input_with_positional_encoding
    
    batch_size = 4
    num_ports = 4
    seq_len = 12
    N = 64
    
    # 创建模型（使用循环卷积）
    model_circular = ComplexResidualUNet(
        input_channels=2,
        output_channels=1,
        base_channels=32,
        depth=3,
        attention_flag=True,
        activation='modrelu',
        circular=True
    )
    
    # 创建输入
    channel_estimates = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64)
    pos_values = torch.tensor([0, 2, 6, 8], dtype=torch.long)
    input_tensor = create_input_with_positional_encoding(channel_estimates, pos_values, N, 'cpu')
    
    print(f"\nInput shape: {input_tensor.shape}")
    
    # 前向传播
    output = model_circular(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {num_ports}, 1, {seq_len})")
    
    # 验证输出形状
    assert output.shape == (batch_size, num_ports, 1, seq_len), "Output shape incorrect!"
    
    print("\n✓ Full U-Net with circular convolution works correctly!")
    
    # 打印参数量
    total_params = sum(p.numel() for p in model_circular.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CIRCULAR CONVOLUTION TEST SUITE")
    print("=" * 80)
    
    try:
        # 运行所有测试
        test_circular_padding()
        test_circular_conv_length()
        test_boundary_values()
        test_full_unet()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED! ✓")
        print("=" * 80)
        print("\nCircular convolution implementation verified:")
        print("  ✓ Circular padding works correctly")
        print("  ✓ Sequence length is preserved")
        print("  ✓ Boundary values use wraparound correctly")
        print("  ✓ Full U-Net integration successful")
        print("\nExample: [0,1,2,...,11]")
        print("  - First conv uses: [11, 0, 1]")
        print("  - Last conv uses:  [10, 11, 0]")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
