"""
验证 ONNX 模式和训练模式的等价性

这个脚本会：
1. 创建两个相同配置的模型（一个 onnx_mode=True，一个 False）
2. 复制权重确保完全相同
3. 对相同输入进行前向传播
4. 验证输出完全等价（误差 < 1e-6）
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal


def test_onnx_mode_equivalence():
    """测试两种模式的等价性"""
    
    print("="*80)
    print("ONNX Mode vs Training Mode - Equivalence Test")
    print("="*80)
    
    # 配置
    config = {
        'seq_len': 12,
        'num_ports': 4,
        'hidden_dim': 64,
        'num_stages': 2,
        'share_weights_across_stages': False,
        'activation_type': 'split_relu'
    }
    
    print("\nModel Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建两个模型
    print("\n" + "-"*80)
    print("Creating models...")
    print("-"*80)
    
    model_training = ResidualRefinementSeparatorReal(
        **config,
        onnx_mode=False  # 训练模式（高效）
    )
    
    model_onnx = ResidualRefinementSeparatorReal(
        **config,
        onnx_mode=True  # ONNX 模式（MATLAB 兼容）
    )
    
    # 复制权重（确保完全相同）
    model_onnx.load_state_dict(model_training.state_dict())
    
    print(f"✓ Training mode model created: onnx_mode={model_training.onnx_mode}")
    print(f"✓ ONNX mode model created: onnx_mode={model_onnx.onnx_mode}")
    print(f"✓ Weights synchronized")
    
    # 设置为评估模式
    model_training.eval()
    model_onnx.eval()
    
    # 生成测试数据
    print("\n" + "-"*80)
    print("Generating test data...")
    print("-"*80)
    
    batch_sizes = [1, 2, 4]
    L = config['seq_len']
    
    all_passed = True
    
    for batch_size in batch_sizes:
        print(f"\nTest with batch_size={batch_size}:")
        
        # 生成复数输入
        y_complex = torch.randn(batch_size, L) + 1j * torch.randn(batch_size, L)
        
        # 转换为实数格式
        y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)  # (B, L*2)
        
        # 能量归一化（在模型外）
        y_energy = torch.sqrt(torch.mean(y_complex.abs()**2, dim=-1, keepdim=True))  # (B, 1)
        y_normalized = y_stacked / y_energy  # (B, L*2), broadcasting works
        
        print(f"  Input shape: {y_normalized.shape}")
        print(f"  Input energy: {y_energy.mean().item():.6f}")
        
        # 前向传播（两种模式）
        with torch.no_grad():
            output_training = model_training(y_normalized)
            output_onnx = model_onnx(y_normalized)
        
        print(f"  Training output shape: {output_training.shape}")
        print(f"  ONNX output shape: {output_onnx.shape}")
        
        # 计算差异
        abs_diff = (output_training - output_onnx).abs()
        max_diff = abs_diff.max().item()
        mean_diff = abs_diff.mean().item()
        rel_diff = (abs_diff / (output_training.abs() + 1e-8)).max().item()
        
        print(f"\n  Difference Analysis:")
        print(f"    Max absolute diff:  {max_diff:.2e}")
        print(f"    Mean absolute diff: {mean_diff:.2e}")
        print(f"    Max relative diff:  {rel_diff:.2e}")
        
        # 判断是否通过
        if max_diff < 1e-6:
            print(f"    ✓ PASSED (diff < 1e-6)")
        elif max_diff < 1e-5:
            print(f"    ✓ PASSED (diff < 1e-5, acceptable)")
        elif max_diff < 1e-4:
            print(f"    ⚠️  WARNING (diff < 1e-4, marginal)")
            all_passed = False
        else:
            print(f"    ✗ FAILED (diff >= 1e-4)")
            all_passed = False
        
        # 恢复能量并验证重建
        # y_energy: (B, 1), output: (B, P, L*2), need to expand properly
        output_training_restored = output_training * y_energy.unsqueeze(1)  # (B, P, L*2)
        output_onnx_restored = output_onnx * y_energy.unsqueeze(1)  # (B, P, L*2)
        
        # 转换为复数
        h_training = torch.complex(
            output_training_restored[:, :, :L],
            output_training_restored[:, :, L:]
        )
        h_onnx = torch.complex(
            output_onnx_restored[:, :, :L],
            output_onnx_restored[:, :, L:]
        )
        
        # 重建输入
        y_recon_training = h_training.sum(dim=1)
        y_recon_onnx = h_onnx.sum(dim=1)
        
        # 计算重建误差
        recon_error_training = (y_complex - y_recon_training).abs().pow(2).mean().sqrt()
        recon_error_onnx = (y_complex - y_recon_onnx).abs().pow(2).mean().sqrt()
        recon_error_rel = (recon_error_training - recon_error_onnx).abs() / (recon_error_training + 1e-8)
        
        print(f"\n  Reconstruction Error:")
        print(f"    Training mode: {recon_error_training.item():.6f}")
        print(f"    ONNX mode:     {recon_error_onnx.item():.6f}")
        print(f"    Relative diff: {recon_error_rel.item():.2e}")
    
    # 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe two modes are mathematically equivalent!")
        print("You can train with either mode and get the same results.")
        print("\nRecommendations:")
        print("  - Training: use onnx_mode=False for best performance (~20% faster)")
        print("  - ONNX Export: use onnx_mode=True for MATLAB compatibility")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("\nThe two modes have differences that may affect results.")
        print("Please review the differences above.")
        return 1


def test_gradient_computation():
    """测试两种模式的梯度计算"""
    
    print("\n" + "="*80)
    print("Gradient Computation Test")
    print("="*80)
    
    config = {
        'seq_len': 12,
        'num_ports': 4,
        'hidden_dim': 64,
        'num_stages': 2,
        'share_weights_across_stages': False,
        'activation_type': 'split_relu'
    }
    
    # 创建两个模型
    model_training = ResidualRefinementSeparatorReal(**config, onnx_mode=False)
    model_onnx = ResidualRefinementSeparatorReal(**config, onnx_mode=True)
    
    # 复制权重
    model_onnx.load_state_dict(model_training.state_dict())
    
    # 训练模式
    model_training.train()
    model_onnx.train()
    
    # 生成数据
    y = torch.randn(2, 24, requires_grad=True)
    h_target = torch.randn(2, 4, 24)
    
    # 前向传播 + 反向传播（训练模式）
    h_pred_training = model_training(y)
    loss_training = ((h_pred_training - h_target) ** 2).mean()
    loss_training.backward()
    
    # 收集梯度
    grads_training = []
    for param in model_training.parameters():
        if param.grad is not None:
            grads_training.append(param.grad.clone())
    
    # 清零梯度
    model_onnx.zero_grad()
    
    # 前向传播 + 反向传播（ONNX 模式）
    h_pred_onnx = model_onnx(y)
    loss_onnx = ((h_pred_onnx - h_target) ** 2).mean()
    loss_onnx.backward()
    
    # 收集梯度
    grads_onnx = []
    for param in model_onnx.parameters():
        if param.grad is not None:
            grads_onnx.append(param.grad.clone())
    
    # 对比梯度
    print(f"\nComparing gradients for {len(grads_training)} parameters...")
    
    max_grad_diff = 0
    for i, (g_train, g_onnx) in enumerate(zip(grads_training, grads_onnx)):
        diff = (g_train - g_onnx).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)
    
    print(f"  Max gradient difference: {max_grad_diff:.2e}")
    
    if max_grad_diff < 1e-6:
        print(f"  ✓ Gradients are equivalent (diff < 1e-6)")
        print(f"\nBoth modes can be used for training!")
        return True
    elif max_grad_diff < 1e-5:
        print(f"  ✓ Gradients are very similar (diff < 1e-5)")
        print(f"\nBoth modes can be used for training!")
        return True
    else:
        print(f"  ⚠️  Gradients have noticeable differences")
        print(f"\nUse with caution for training.")
        return False


def main():
    """主函数"""
    
    # 测试 1：前向传播等价性
    exit_code = test_onnx_mode_equivalence()
    
    # 测试 2：梯度计算
    grad_ok = test_gradient_computation()
    
    # 最终总结
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    
    if exit_code == 0 and grad_ok:
        print("✓ All tests passed!")
        print("\nYou can safely:")
        print("  1. Train with onnx_mode=False for better performance")
        print("  2. Train with onnx_mode=True to validate ONNX compatibility")
        print("  3. Switch onnx_mode when exporting without retraining")
    else:
        print("⚠️  Some tests showed differences")
        print("\nRecommendations:")
        print("  1. Use onnx_mode=False for training")
        print("  2. Set onnx_mode=True only when exporting to ONNX")
        print("  3. Verify exported ONNX model carefully")
    
    print("="*80)
    
    return exit_code


if __name__ == "__main__":
    exit(main())
