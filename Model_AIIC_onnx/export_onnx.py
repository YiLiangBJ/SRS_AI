"""
Export trained model to ONNX format for MATLAB deployment

This script exports the real-valued channel separator model to ONNX format,
which can be imported into MATLAB for inference.

Usage:
    python Model_AIIC_onnx/export_onnx.py \
        --checkpoint ./Model_AIIC_onnx/out6ports/stages=3_share=False/model.pth \
        --output model.onnx
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
except ImportError:
    from channel_separator import ResidualRefinementSeparatorReal


def export_to_onnx(
    checkpoint_path,
    output_path,
    opset_version=14,
    verbose=True
):
    """
    Export trained model to ONNX format
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.pth)
        output_path: Output ONNX file path
        opset_version: ONNX opset version (14+ recommended for MATLAB R2021a+)
        verbose: Print detailed information
    
    Returns:
        output_path: Path to exported ONNX file
    """
    if verbose:
        print("="*80)
        print("Exporting Model to ONNX Format")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output:     {output_path}")
        print(f"Opset:      {opset_version}")
        print()
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    hyperparams = checkpoint.get('hyperparameters', {})
    
    if verbose:
        print("Model Configuration:")
        print(f"  Sequence length: {config['seq_len']}")
        print(f"  Hidden dim:      {config['hidden_dim']}")
        print(f"  Num stages:      {config['num_stages']}")
        print(f"  Share weights:   {config['share_weights']}")
    
    # Get parameters
    seq_len = config['seq_len']
    num_ports = hyperparams.get('num_ports', 4)
    pos_values = hyperparams.get('pos_values', None)
    
    if pos_values is not None:
        num_ports = len(pos_values)
        if verbose:
            print(f"  Num ports:       {num_ports}")
            print(f"  Port positions:  {pos_values}")
    
    # Get activation type
    activation_type = config.get('activation_type', 'split_relu')
    if verbose:
        print(f"  Activation:      {activation_type}")
    
    # Get onnx_mode (default to False if not in config, but will set to True for export)
    onnx_mode_saved = config.get('onnx_mode', False)
    if verbose:
        print(f"  ONNX mode (saved): {onnx_mode_saved}")
    
    # Create model
    model = ResidualRefinementSeparatorReal(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=config['hidden_dim'],
        num_stages=config['num_stages'],
        share_weights_across_stages=config['share_weights'],
        activation_type=activation_type,
        onnx_mode=False  # Create with onnx_mode=False first (for weight loading compatibility)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # ⭐ Switch to ONNX mode for export (ensures MATLAB compatibility)
    model.onnx_mode = True
    if verbose:
        print(f"  ⭐ ONNX mode set to: True (for MATLAB compatible export)")
    
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters:      {num_params:,}")
        print()
    
    # Create dummy input (real-valued stacked format)
    # Input shape: (batch_size, seq_len * 2) = (batch_size, L*2)
    # where [:, :L] is real part, [:, L:] is imaginary part
    dummy_input = torch.randn(1, seq_len * 2)
    
    if verbose:
        print("Input/Output Format:")
        print(f"  Input:  (batch, {seq_len * 2}) = [real({seq_len}); imag({seq_len})]")
        print(f"  Output: (batch, {num_ports}, {seq_len * 2}) = [[h0_real; h0_imag], ...]")
        print()
    
    # Export to ONNX
    if verbose:
        print("Exporting to ONNX...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['y_real_imag'],
            output_names=['h_real_imag'],
            dynamic_axes={
                'y_real_imag': {0: 'batch_size'},
                'h_real_imag': {0: 'batch_size'}
            },
            verbose=False
        )
        
        if verbose:
            print(f"✓ ONNX export successful!")
            print(f"  Saved to: {output_path}")
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  File size: {file_size:.2f} MB")
            print()
            
            print("MATLAB Usage:")
            print("─" * 80)
            print("% Load ONNX model")
            print(f"net = importONNXNetwork('{Path(output_path).name}', 'OutputLayerType', 'regression');")
            print()
            print("% Prepare input (convert complex to [real; imag])")
            print(f"y = randn(1, {seq_len}) + 1i*randn(1, {seq_len});  % Complex signal")
            print("y_stacked = [real(y), imag(y)];  % Convert to real stacked")
            print()
            print("% ⚠️  IMPORTANT: Energy normalization (done externally)")
            print("y_energy = sqrt(mean(abs(y).^2));")
            print("y_normalized = y_stacked / y_energy;")
            print()
            print("% Predict")
            print("h_normalized = predict(net, y_normalized);")
            print()
            print("% ⚠️  IMPORTANT: Restore energy")
            print("h_stacked = h_normalized * y_energy;")
            print()
            print("% Convert back to complex")
            print(f"L = {seq_len}; P = {num_ports};")
            print("h_real = h_real_imag(:, :, 1:L);")
            print("h_imag = h_real_imag(:, :, L+1:end);")
            print("h = complex(h_real, h_imag);  % (1, P, L)")
            print("─" * 80)
            print()
        
        return output_path
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Export trained model to ONNX format"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output ONNX file path (default: same dir as checkpoint)')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version (default: 14)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        # Use same directory as checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output = os.path.join(checkpoint_dir, 'model.onnx')
    
    # Export
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
