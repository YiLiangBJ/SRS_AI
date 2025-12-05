"""
Export ONNX model compatible with MATLAB

MATLAB ONNX Importer Limitations:
- Only supports Opset 9 (not 14)
- Doesn't support dynamic operations (Slice, Gather, Unsqueeze, etc.)
- Limited operator support

This script creates a simplified model that:
- Uses Opset 9
- Removes dynamic operations
- Disables energy normalization (done in MATLAB instead)
- Uses fixed batch size (no dynamic axes)

Usage:
    python Model_AIIC_onnx/export_onnx_matlab.py \
        --checkpoint ./Model_AIIC_onnx/test/stages=2_share=False_loss=nmse_act=split_relu/model.pth \
        --output model_matlab.onnx \
        --opset 9
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Model_AIIC_onnx.channel_separator import ResidualRefinementSeparatorReal
except ImportError:
    from channel_separator import ResidualRefinementSeparatorReal


class MATLABCompatibleModel(nn.Module):
    """
    Ultra-simplified model for MATLAB compatibility
    
    MATLAB limitations require:
    - No dynamic operations (expand, unsqueeze, etc.)
    - No torch.cat across batch dimension
    - Simple linear paths only
    
    This model processes each port independently without any complex tensor ops.
    """
    def __init__(self, base_model):
        super().__init__()
        self.seq_len = base_model.seq_len
        self.num_ports = base_model.num_ports
        self.num_stages = base_model.num_stages
        self.share_weights = base_model.share_weights_across_stages
        
        # Extract just the MLPs from base model
        self.port_mlps = base_model.port_mlps
        
    def forward(self, y_real):
        """
        Ultra-simple forward pass - each port independently
        
        Args:
            y_real: (1, L*2) - [real; imag] format
        
        Returns:
            Concatenated outputs (1, P*L*2)
            User must reshape in MATLAB to (1, P, L*2)
        """
        outputs = []
        
        # Process each port independently
        for port_idx in range(self.num_ports):
            x = y_real  # Start with input
            
            # Run through stages for this port
            for stage_idx in range(self.num_stages):
                if self.share_weights:
                    mlp = self.port_mlps[port_idx]
                else:
                    mlp = self.port_mlps[port_idx][stage_idx]
                
                x = mlp(x)  # (1, L*2)
            
            outputs.append(x)
        
        # Concatenate all port outputs into single vector
        # MATLAB will need to reshape (1, P*L*2) -> (1, P, L*2)
        result = torch.cat(outputs, dim=-1)  # (1, P*L*2)
        
        return result


def export_to_onnx_matlab(
    checkpoint_path,
    output_path,
    opset_version=9,
    verbose=True
):
    """
    Export trained model to MATLAB-compatible ONNX format
    
    Args:
        checkpoint_path: Path to trained model checkpoint (.pth)
        output_path: Output ONNX file path
        opset_version: ONNX opset version (9 for MATLAB compatibility)
        verbose: Print detailed information
    
    Returns:
        output_path: Path to exported ONNX file
    """
    if verbose:
        print("="*80)
        print("Exporting Model to MATLAB-Compatible ONNX")
        print("="*80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output:     {output_path}")
        print(f"Opset:      {opset_version} (MATLAB compatible)")
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
        print(f"  Normalize:       False (disabled for MATLAB)")
    
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
    
    # Create base model
    base_model = ResidualRefinementSeparatorReal(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=config['hidden_dim'],
        num_stages=config['num_stages'],
        share_weights_across_stages=config['share_weights'],
        normalize_energy=False,  # Will be done in MATLAB
        activation_type=activation_type
    )
    
    # Load weights
    base_model.load_state_dict(checkpoint['model_state_dict'])
    base_model.eval()
    
    # Wrap for MATLAB compatibility
    model = MATLABCompatibleModel(base_model)
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Parameters:      {num_params:,}")
        print()
    
    # Create dummy input (fixed batch size = 1)
    dummy_input = torch.randn(1, seq_len * 2)
    
        if verbose:
            print("Input/Output Format:")
            print(f"  Input:  (1, {seq_len * 2}) = [real({seq_len}); imag({seq_len})]")
            print(f"  Output: (1, {num_ports * seq_len * 2}) = concatenated port outputs")
            print(f"          → Reshape in MATLAB to (1, {num_ports}, {seq_len * 2})")
            print()
            print("⚠️  IMPORTANT Notes:")
            print("    1. Energy normalization is DISABLED (do in MATLAB)")
            print("    2. Output is FLATTENED - must reshape in MATLAB")
            print("    3. Residual coupling is REMOVED for simplicity")
            print()
            print("  Reshaping in MATLAB:")
            print(f"    h_flat = predict(net, y);  % (1, {num_ports * seq_len * 2})")
            print(f"    h = reshape(h_flat, [1, {num_ports}, {seq_len * 2}]);")
            print()    # Export to ONNX
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
            dynamic_axes=None,  # Fixed batch size for MATLAB
            verbose=False
        )
        
        if verbose:
            print(f"✓ ONNX model exported!")
            print(f"  Saved to: {output_path}")
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"  File size: {file_size:.2f} MB")
            print()
        
        # Validate ONNX model
        if verbose:
            print("Validating ONNX model...")
        
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            if verbose:
                print("✓ ONNX model validated successfully!")
                print()
        except ImportError:
            if verbose:
                print("⚠️  ONNX package not found, skipping validation")
                print()
        except Exception as e:
            if verbose:
                print(f"⚠️  Validation warning: {e}")
                print()
        
        # Test inference with ONNX Runtime
        if verbose:
            print("Testing ONNX inference...")
        
        try:
            import onnxruntime as ort
            
            ort_session = ort.InferenceSession(output_path)
            test_input = torch.randn(1, seq_len * 2).numpy()
            
            ort_inputs = {ort_session.get_inputs()[0].name: test_input}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            if verbose:
                print(f"✓ Inference test passed!")
                print(f"  Input shape:  {test_input.shape}")
                print(f"  Output shape: {ort_outputs[0].shape}")
                print()
        except ImportError:
            if verbose:
                print("⚠️  ONNX Runtime not found, skipping inference test")
                print()
        except Exception as e:
            if verbose:
                print(f"⚠️  Inference test warning: {e}")
                print()
        
        if verbose:
            print("="*80)
            print("MATLAB Usage Example:")
            print("="*80)
            print(f"""
% Load ONNX model
net = importONNXNetwork('{output_path}', 'OutputLayerType', 'regression');

% Prepare input data
y = randn(1, {seq_len}) + 1i*randn(1, {seq_len});  % Complex signal
y_stacked = [real(y), imag(y)];  % Convert to [real; imag] format

% ⚠️ IMPORTANT: Normalize energy (since model has it disabled)
y_energy = sqrt(mean(abs(y).^2));
y_normalized = y_stacked / y_energy;

% Inference - output is flattened
h_flat = predict(net, y_normalized);  % (1, {num_ports * seq_len * 2})

% ⚠️ IMPORTANT: Reshape output
h_stacked = reshape(h_flat, [1, {num_ports}, {seq_len * 2}]);

% ⚠️ IMPORTANT: Restore energy
h_stacked = h_stacked * y_energy;

% Convert back to complex
L = {seq_len};
h_real = h_stacked(:, :, 1:L);
h_imag = h_stacked(:, :, L+1:end);
h = complex(h_real, h_imag);  % (1, {num_ports}, {seq_len})

% Display results
disp('Separated channels:');
disp(size(h));
            """)
            print("="*80)
        
        return output_path
        
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Export MATLAB-compatible ONNX model (Opset 9)"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='model_matlab.onnx',
                       help='Output ONNX file path (default: model_matlab.onnx)')
    parser.add_argument('--opset', type=int, default=9,
                       help='ONNX opset version (default: 9 for MATLAB)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export
    try:
        export_to_onnx_matlab(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            opset_version=args.opset,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
