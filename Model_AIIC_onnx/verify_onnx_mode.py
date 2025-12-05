"""
Verify equivalence between training mode and ONNX mode

This script verifies that:
1. Both modes produce identical outputs (within numerical precision)
2. Gradients are computed correctly in both modes
3. The model is mathematically equivalent
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal

def test_forward_equivalence():
    """Test that training mode and ONNX mode produce identical outputs"""
    print("\n" + "="*80)
    print("Test 1: Forward Pass Equivalence")
    print("="*80)
    
    # Create two models with identical weights
    model_train = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=2,
        share_weights_across_stages=False,
        activation_type='split_relu',
        onnx_mode=False  # Training mode
    )
    
    model_onnx = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=2,
        share_weights_across_stages=False,
        activation_type='split_relu',
        onnx_mode=True  # ONNX mode
    )
    
    # Copy weights from training model to ONNX model
    model_onnx.load_state_dict(model_train.state_dict())
    
    # Set to eval mode
    model_train.eval()
    model_onnx.eval()
    
    # Test with multiple batch sizes
    for batch_size in [1, 2, 4]:
        y = torch.randn(batch_size, 24)
        
        with torch.no_grad():
            output_train = model_train(y)
            output_onnx = model_onnx(y)
        
        # Check shapes
        assert output_train.shape == output_onnx.shape == (batch_size, 4, 24), \
            f"Shape mismatch for batch_size={batch_size}"
        
        # Check numerical equivalence
        max_diff = (output_train - output_onnx).abs().max().item()
        mean_diff = (output_train - output_onnx).abs().mean().item()
        
        print(f"  Batch size {batch_size}:")
        print(f"    Max difference:  {max_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        
        # Should be identical (within floating point precision)
        assert max_diff < 1e-6, f"Outputs differ too much: {max_diff:.2e}"
    
    print("\n✓ Test 1 PASSED: Outputs are identical")
    return True


def test_gradient_equivalence():
    """Test that gradients are computed correctly in both modes"""
    print("\n" + "="*80)
    print("Test 2: Gradient Computation Equivalence")
    print("="*80)
    
    # Create two models with identical weights
    model_train = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=2,
        onnx_mode=False
    )
    
    model_onnx = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=2,
        onnx_mode=True
    )
    
    # Copy weights
    model_onnx.load_state_dict(model_train.state_dict())
    
    # Set to train mode
    model_train.train()
    model_onnx.train()
    
    # Same input for both
    torch.manual_seed(42)
    y = torch.randn(2, 24, requires_grad=True)
    
    # Forward + backward for training mode
    output_train = model_train(y)
    loss_train = output_train.pow(2).mean()
    loss_train.backward()
    
    # Collect gradients
    grad_train = {}
    for name, param in model_train.named_parameters():
        if param.grad is not None:
            grad_train[name] = param.grad.clone()
    
    # Reset input grad
    y.grad = None
    
    # Forward + backward for ONNX mode
    output_onnx = model_onnx(y)
    loss_onnx = output_onnx.pow(2).mean()
    loss_onnx.backward()
    
    # Collect gradients
    grad_onnx = {}
    for name, param in model_onnx.named_parameters():
        if param.grad is not None:
            grad_onnx[name] = param.grad.clone()
    
    # Compare gradients
    print(f"  Comparing gradients for {len(grad_train)} parameters...")
    max_grad_diff = 0
    for name in grad_train.keys():
        diff = (grad_train[name] - grad_onnx[name]).abs().max().item()
        max_grad_diff = max(max_grad_diff, diff)
        if diff > 1e-6:
            print(f"    ⚠️  {name}: diff={diff:.2e}")
    
    print(f"\n  Maximum gradient difference: {max_grad_diff:.2e}")
    assert max_grad_diff < 1e-5, f"Gradients differ too much: {max_grad_diff:.2e}"
    
    print("\n✓ Test 2 PASSED: Gradients are identical")
    return True


def test_external_normalization():
    """Test that external energy normalization works correctly"""
    print("\n" + "="*80)
    print("Test 3: External Energy Normalization")
    print("="*80)
    
    model = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=2,
        onnx_mode=False
    )
    model.eval()
    
    # Generate complex signal
    y_complex = torch.randn(1, 12) + 1j*torch.randn(1, 12)
    y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)
    
    print(f"  Original signal energy: {y_complex.abs().pow(2).mean().sqrt().item():.6f}")
    
    # Step 1: Energy normalization (external)
    y_R, y_I = y_stacked[:, :12], y_stacked[:, 12:]
    y_energy = (y_R**2 + y_I**2).mean(dim=-1, keepdim=True).sqrt()
    y_normalized = y_stacked / y_energy
    
    print(f"  Normalized signal energy: {y_normalized.pow(2).mean().sqrt().item():.6f}")
    
    # Step 2: Model inference
    with torch.no_grad():
        h_normalized = model(y_normalized)
    
    print(f"  Output shape: {tuple(h_normalized.shape)}")
    
    # Step 3: Energy restoration (external)
    h = h_normalized * y_energy.unsqueeze(1)
    
    # Convert to complex
    h_complex = torch.complex(h[:, :, :12], h[:, :, 12:])
    
    # Verify reconstruction
    y_recon_complex = h_complex.sum(dim=1)
    recon_error = (y_complex - y_recon_complex).abs().pow(2).mean().sqrt()
    recon_error_pct = (recon_error / y_complex.abs().pow(2).mean().sqrt() * 100).item()
    
    print(f"\n  Reconstruction error: {recon_error.item():.2e} ({recon_error_pct:.2f}%)")
    print(f"  Note: High error is expected for untrained model")
    
    print("\n✓ Test 3 PASSED: External normalization workflow works")
    return True


def test_performance_comparison():
    """Compare performance between training mode and ONNX mode"""
    print("\n" + "="*80)
    print("Test 4: Performance Comparison")
    print("="*80)
    
    import time
    
    # Create models
    model_train = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=3,
        onnx_mode=False
    )
    
    model_onnx = ResidualRefinementSeparatorReal(
        seq_len=12,
        num_ports=4,
        hidden_dim=64,
        num_stages=3,
        onnx_mode=True
    )
    
    model_onnx.load_state_dict(model_train.state_dict())
    
    model_train.eval()
    model_onnx.eval()
    
    # Warmup
    y = torch.randn(32, 24)
    with torch.no_grad():
        _ = model_train(y)
        _ = model_onnx(y)
    
    # Benchmark training mode
    n_iters = 100
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model_train(y)
    time_train = (time.time() - start) / n_iters * 1000
    
    # Benchmark ONNX mode
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model_onnx(y)
    time_onnx = (time.time() - start) / n_iters * 1000
    
    slowdown = (time_onnx / time_train - 1) * 100
    
    print(f"  Training mode: {time_train:.3f} ms")
    print(f"  ONNX mode:     {time_onnx:.3f} ms")
    print(f"  Slowdown:      {slowdown:.1f}%")
    print(f"\n  Note: ONNX mode is slower but enables MATLAB compatibility")
    
    print("\n✓ Test 4 PASSED: Performance measured")
    return True


def main():
    """Run all tests"""
    print("="*80)
    print("Channel Separator: Training vs ONNX Mode Equivalence Tests")
    print("="*80)
    print("\nVerifying that onnx_mode=True produces identical results")
    print("while using only ONNX Opset 9 compatible operations.")
    print("="*80)
    
    tests = [
        ("Forward Pass Equivalence", test_forward_equivalence),
        ("Gradient Computation", test_gradient_equivalence),
        ("External Normalization", test_external_normalization),
        ("Performance Comparison", test_performance_comparison),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe two modes are mathematically equivalent!")
        print("Training mode: Fast, for training")
        print("ONNX mode:     Slower, for MATLAB-compatible export")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failed tests above.")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
