"""
Test different base_channels configurations
"""

import torch
from complexUnet import ComplexResidualUNet

def test_base_channels(base_ch):
    """Test a specific base_channels configuration"""
    try:
        print(f"\nTesting base_channels={base_ch}...")
        
        # Create model
        model = ComplexResidualUNet(
            input_channels=2,
            output_channels=1,
            base_channels=base_ch,
            depth=3,
            attention_flag=True,
            circular=True
        )
        
        # Create test input
        x = torch.randn(2, 4, 2, 12, dtype=torch.complex64)
        
        # Forward pass
        y = model(x)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ Success!")
        print(f"    Input:  {x.shape}")
        print(f"    Output: {y.shape}")
        print(f"    Params: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("Testing Different base_channels Configurations")
    print("=" * 80)
    
    # Test various base_channels values
    test_configs = [4, 8, 16, 32, 64]
    
    results = {}
    for base_ch in test_configs:
        results[base_ch] = test_base_channels(base_ch)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for base_ch, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  base_channels={base_ch:3d}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("=" * 80)
