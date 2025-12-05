"""
ONNX Export Diagnostic Tool

This script:
1. Exports your model to ONNX
2. Lists all operators used
3. Checks for potential compatibility issues
4. Tests ONNX Runtime inference
5. Provides recommendations

Usage:
    python Model_AIIC_onnx/diagnose_onnx.py \
        --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
except ImportError:
    from channel_separator import ResidualRefinementSeparatorReal


# Problematic operators for different backends
PROBLEMATIC_OPS = {
    'MATLAB': [
        'Slice', 'Gather', 'Unsqueeze', 'Squeeze', 'Expand',
        'ReduceSum', 'ReduceMean', 'Where', 'ConstantOfShape'
    ],
    'OpenVINO_old': [
        'Loop', 'If', 'NonZero', 'TopK'
    ],
    'TensorRT_old': [
        'NonZero', 'Resize', 'ScatterElements'
    ],
    'generic': [
        'Loop', 'If', 'Scan'  # Control flow
    ]
}


def diagnose_model(checkpoint_path=None, opset_version=11):
    """
    Diagnose ONNX export compatibility
    
    Args:
        checkpoint_path: Path to trained model (optional, will use random init if None)
        opset_version: ONNX opset version to test
    """
    print("=" * 80)
    print("ONNX Export Diagnostic Tool")
    print("=" * 80)
    print()
    
    # Create model
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        hparams = checkpoint['hyperparameters']
        
        model = ResidualRefinementSeparatorReal(
            seq_len=config['seq_len'],
            num_ports=hparams['num_ports'],
            hidden_dim=config['hidden_dim'],
            num_stages=config['num_stages'],
            share_weights_across_stages=config['share_weights'],
            normalize_energy=config['normalize_energy'],
            activation_type=config.get('activation_type', 'split_relu')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        seq_len = config['seq_len']
        num_ports = hparams['num_ports']
    else:
        print("Using default model configuration (no checkpoint)")
        seq_len = 12
        num_ports = 4
        model = ResidualRefinementSeparatorReal(
            seq_len=seq_len,
            num_ports=num_ports,
            hidden_dim=64,
            num_stages=2,
            share_weights_across_stages=False,
            normalize_energy=True,
            activation_type='split_relu'
        )
    
    model.eval()
    print(f"  Sequence length: {seq_len}")
    print(f"  Num ports:       {num_ports}")
    print()
    
    # Create dummy input
    dummy_input = torch.randn(1, seq_len * 2)
    
    # Test forward pass
    print("Testing PyTorch forward pass...")
    try:
        with torch.no_grad():
            output_torch = model(dummy_input)
        print(f"  ✓ Forward pass successful")
        print(f"  Input shape:  {tuple(dummy_input.shape)}")
        print(f"  Output shape: {tuple(output_torch.shape)}")
        print()
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return
    
    # Export to ONNX
    onnx_path = 'diagnostic_model.onnx'
    print(f"Exporting to ONNX (Opset {opset_version})...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['y_stacked'],
            output_names=['h_stacked'],
            dynamic_axes=None,
            verbose=False
        )
        print(f"  ✓ Export successful: {onnx_path}")
        print()
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
        print()
        print("Common causes:")
        print("  1. Unsupported operations in model")
        print("  2. Dynamic shapes")
        print("  3. Custom autograd functions")
        return
    
    # Analyze ONNX model
    print("Analyzing ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        
        # Check model
        try:
            onnx.checker.check_model(onnx_model)
            print("  ✓ ONNX model is valid")
        except Exception as e:
            print(f"  ⚠️  ONNX model validation warning: {e}")
        
        # List operators
        ops = {}
        for node in onnx_model.graph.node:
            op_type = node.op_type
            ops[op_type] = ops.get(op_type, 0) + 1
        
        print(f"\n  Operators used ({len(ops)} types, {sum(ops.values())} total):")
        for op, count in sorted(ops.items()):
            print(f"    {op:20s} : {count:3d} occurrence(s)")
        print()
        
        # Check for problematic operators
        print("Compatibility Analysis:")
        print("-" * 80)
        
        issues_found = False
        
        for backend, problematic in PROBLEMATIC_OPS.items():
            backend_issues = [op for op in ops.keys() if op in problematic]
            if backend_issues:
                issues_found = True
                print(f"\n  ⚠️  {backend} compatibility issues:")
                for op in backend_issues:
                    print(f"      - {op} ({ops[op]} occurrence(s))")
        
        if not issues_found:
            print("\n  ✓ No known compatibility issues detected!")
        
        print()
        
    except ImportError:
        print("  ⚠️  ONNX package not found, skipping analysis")
        print("     Install: pip install onnx")
        print()
    except Exception as e:
        print(f"  ✗ Analysis failed: {e}")
        print()
    
    # Test ONNX Runtime
    print("Testing ONNX Runtime inference...")
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(onnx_path)
        
        # Test inference
        x_numpy = dummy_input.numpy()
        outputs_onnx = sess.run(None, {'y_stacked': x_numpy})
        output_onnx = outputs_onnx[0]
        
        # Compare with PyTorch
        max_diff = np.max(np.abs(output_torch.numpy() - output_onnx))
        mean_diff = np.mean(np.abs(output_torch.numpy() - output_onnx))
        
        print(f"  ✓ ONNX Runtime inference successful")
        print(f"  Numerical accuracy:")
        print(f"    Max difference:  {max_diff:.2e}")
        print(f"    Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-5:
            print(f"    ✓ Excellent accuracy!")
        elif max_diff < 1e-3:
            print(f"    ✓ Good accuracy")
        else:
            print(f"    ⚠️  Moderate accuracy - check for precision issues")
        print()
        
    except ImportError:
        print("  ⚠️  ONNX Runtime not found, skipping test")
        print("     Install: pip install onnxruntime")
        print()
    except Exception as e:
        print(f"  ✗ ONNX Runtime test failed: {e}")
        print()
    
    # Provide recommendations
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)
    print()
    
    if issues_found:
        print("⚠️  Compatibility issues detected. Recommendations:")
        print()
        
        if any(op in ops for op in ['Slice', 'Gather']):
            print("1. Dynamic Slicing detected:")
            print("   Replace:")
            print("     y_R = y_stacked[:, :L]")
            print("     y_I = y_stacked[:, L:]")
            print("   With:")
            print("     y_R, y_I = torch.chunk(y_stacked, 2, dim=-1)")
            print()
        
        if 'Expand' in ops:
            print("2. Expand operation detected:")
            print("   Replace:")
            print("     features = y.unsqueeze(1).expand(-1, P, -1)")
            print("   With:")
            print("     features = y.unsqueeze(1).repeat(1, P, 1)")
            print("   Or add:")
            print("     features = y.unsqueeze(1).expand(-1, P, -1).contiguous()")
            print()
        
        if any(op in ops for op in ['Unsqueeze', 'Squeeze']):
            print("3. Squeeze/Unsqueeze operations:")
            print("   ✓ OK for ONNX Runtime and OpenVINO")
            print("   ✗ Not supported in MATLAB")
            print("   No action needed for OpenVINO deployment")
            print()
    else:
        print("✓ No major issues detected!")
        print()
        print("Your model should work with:")
        print("  - ONNX Runtime")
        print("  - OpenVINO")
        print("  - TensorRT (most versions)")
        print()
    
    print("Next steps:")
    print(f"  1. Review operators list above")
    print(f"  2. If needed, modify model based on recommendations")
    print(f"  3. Test OpenVINO conversion:")
    print(f"     mo --input_model {onnx_path} --output_dir openvino_model")
    print()
    
    # Cleanup
    if Path(onnx_path).exists():
        print(f"ONNX model saved: {onnx_path}")
        print("You can inspect it with: netron.app")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose ONNX export compatibility'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained model checkpoint (optional)')
    parser.add_argument('--opset', type=int, default=11,
                       help='ONNX opset version (default: 11)')
    
    args = parser.parse_args()
    
    diagnose_model(
        checkpoint_path=args.checkpoint,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()
