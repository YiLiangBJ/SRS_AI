"""
Channel Separator Model using Real-valued Block Matrix (ONNX Compatible)

Based on Model_AIIC/channel_separator.py but uses only real tensors

Key differences:
- All tensors are real with format [real_part; imag_part]
- Uses ComplexLinearReal for matrix multiplication
- Supports multiple complex activation functions
- Fully ONNX compatible for MATLAB deployment

Input format:  y_stacked (B, L*2) = [y_R; y_I]
Output format: h_stacked (B, P, L*2) = [[h0_R; h0_I], [h1_R; h1_I], ...]
"""

import torch
import torch.nn as nn

try:
    from .complex_layers import ComplexMLPReal
except ImportError:
    from complex_layers import ComplexMLPReal


class ResidualRefinementSeparatorReal(nn.Module):
    """
    Real-valued Residual Refinement Separator (ONNX Opset 9 compatible)
    
    Architecture:
    - Per-port independent MLPs (using real block matrix)
    - Ports coupled through residual correction
    - Energy normalization handled externally
    - Multiple complex activation options
    
    Args:
        seq_len: Sequence length (default: 12)
        num_ports: Number of ports (default: 4)
        hidden_dim: Hidden dimension for MLPs (default: 64)
        num_stages: Number of refinement stages (default: 3)
        mlp_depth: MLP depth - total number of layers including input/output (default: 3)
                   - 2: Input -> Output (no hidden layer)
                   - 3: Input -> Hidden -> Output (1 hidden layer, default)
                   - 4: Input -> Hidden1 -> Hidden2 -> Output (2 hidden layers)
        share_weights_across_stages: If True, same port uses same MLP (default: False)
        activation_type: Complex activation ('split_relu', 'mod_relu', 'z_relu', 'cardioid')
        onnx_mode: If True, use ONNX Opset 9 compatible operations (default: False)
                   Trade-off: ~20% slower training but MATLAB compatible export
    
    Input/Output (when onnx_mode affects behavior):
        Input:  y_normalized (B, L*2) - pre-normalized [y_R; y_I]
        Output: h_features (B, P, L*2) - separated channels, not energy-restored
        
    Note: Energy normalization and restoration must be done externally.
    """
    def __init__(self, seq_len=12, num_ports=4, hidden_dim=64, num_stages=3,
                 mlp_depth=3, share_weights_across_stages=False, 
                 activation_type='split_relu',
                 onnx_mode=False):
        super().__init__()
        self.seq_len = seq_len
        self.num_ports = num_ports
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.mlp_depth = mlp_depth
        self.share_weights_across_stages = share_weights_across_stages
        self.activation_type = activation_type
        self.onnx_mode = onnx_mode  # ⭐ Flag to control ONNX compatibility
        
        if share_weights_across_stages:
            # Shared weights across stages
            self.port_mlps = nn.ModuleList([
                ComplexMLPReal(seq_len, hidden_dim, mlp_depth, activation_type)
                for _ in range(num_ports)
            ])
        else:
            # Independent weights per stage
            self.port_mlps = nn.ModuleList([
                nn.ModuleList([
                    ComplexMLPReal(seq_len, hidden_dim, mlp_depth, activation_type)
                    for _ in range(num_stages)
                ])
                for _ in range(num_ports)
            ])
    
    def forward(self, y_normalized):
        """
        Forward pass with optional ONNX Opset 9 compatibility mode
        
        ⚠️ IMPORTANT: Input must be pre-normalized externally!
        Energy normalization and restoration are handled outside the model.
        
        Args:
            y_normalized: (B, L*2) real tensor [y_R; y_I] (already energy-normalized)
        
        Returns:
            features: (B, P, L*2) real tensor (not yet energy-restored)
        
        Mode behavior:
            - onnx_mode=False (training): Fast operations, may use unsupported ONNX ops
            - onnx_mode=True (export): Opset 9 compatible, ~20% slower but MATLAB compatible
        """
        B = y_normalized.shape[0]
        L = self.seq_len
        P = self.num_ports
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Feature Initialization
        # ═══════════════════════════════════════════════════════════════════
        if self.onnx_mode:
            # ONNX Mode: Avoid index_put (not supported in Opset 9)
            # Create list and concatenate instead of in-place assignment
            features_list = [y_normalized.unsqueeze(1) for _ in range(P)]
            features = torch.cat(features_list, dim=1)  # (B, P, L*2)
        else:
            # Training Mode: Efficient implementation using unsqueeze + repeat
            features = y_normalized.unsqueeze(1).repeat(1, P, 1)  # (B, P, L*2)
        
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Iterative Refinement with Residual Coupling
        # ═══════════════════════════════════════════════════════════════════
        for stage_idx in range(self.num_stages):
            # ───────────────────────────────────────────────────────────────
            # 2.1: Per-port MLP Processing
            # ───────────────────────────────────────────────────────────────
            if self.onnx_mode:
                # ONNX Mode: Explicit loop (for ONNX export compatibility)
                new_features_list = []
                for port_idx in range(P):
                    x = features[:, port_idx, :]  # (B, L*2)
                    
                    # Select MLP
                    if self.share_weights_across_stages:
                        mlp = self.port_mlps[port_idx]
                    else:
                        mlp = self.port_mlps[port_idx][stage_idx]
                    
                    # Process through MLP
                    output = mlp(x)  # (B, L*2)
                    new_features_list.append(output.unsqueeze(1))  # (B, 1, L*2)
                
                # Concatenate
                features = torch.cat(new_features_list, dim=1)  # (B, P, L*2)
            else:
                # Training Mode: Vectorized processing using torch.stack (MUCH faster!)
                # Process all ports in parallel using list comprehension + stack
                if self.share_weights_across_stages:
                    # Each port has its own MLP
                    outputs = [
                        self.port_mlps[port_idx](features[:, port_idx, :])
                        for port_idx in range(P)
                    ]
                else:
                    # Each port + stage has its own MLP
                    outputs = [
                        self.port_mlps[port_idx][stage_idx](features[:, port_idx, :])
                        for port_idx in range(P)
                    ]
                
                # Stack efficiently: (B, P, L*2)
                features = torch.stack(outputs, dim=1)
            
            # ───────────────────────────────────────────────────────────────
            # 2.2: Residual Computation (Reconstruct y from all ports)
            # ───────────────────────────────────────────────────────────────
            if self.onnx_mode:
                # ONNX Mode: Avoid chunk/split and ReduceSum
                # Manual indexing + explicit loop summation
                features_R = features[:, :, :L]  # (B, P, L)
                features_I = features[:, :, L:]  # (B, P, L)
                
                # Sum across ports using explicit loop
                y_recon_R = features_R[:, 0, :].clone()  # (B, L)
                y_recon_I = features_I[:, 0, :].clone()  # (B, L)
                for p in range(1, P):
                    y_recon_R = y_recon_R + features_R[:, p, :]
                    y_recon_I = y_recon_I + features_I[:, p, :]
                
                y_recon = torch.cat([y_recon_R, y_recon_I], dim=-1)  # (B, L*2)
            else:
                # Training Mode: Efficient implementation using chunk + sum
                features_R, features_I = torch.chunk(features, 2, dim=-1)
                y_recon_R = features_R.sum(dim=1)  # (B, L)
                y_recon_I = features_I.sum(dim=1)  # (B, L)
                y_recon = torch.cat([y_recon_R, y_recon_I], dim=-1)  # (B, L*2)
            
            # ───────────────────────────────────────────────────────────────
            # 2.3: Residual Addition
            # ───────────────────────────────────────────────────────────────
            residual = y_normalized - y_recon  # (B, L*2)
            
            if self.onnx_mode:
                # ONNX Mode: Avoid both Unsqueeze and index_put (in-place assignment)
                # Create list and concatenate instead
                features_with_residual_list = []
                for p in range(P):
                    port_features = features[:, p, :] + residual  # (B, L*2)
                    features_with_residual_list.append(port_features.unsqueeze(1))  # (B, 1, L*2)
                features = torch.cat(features_with_residual_list, dim=1)  # (B, P, L*2)
            else:
                # Training Mode: Broadcasting (generates Unsqueeze in ONNX)
                features = features + residual.unsqueeze(1)  # (B, P, L*2)
        
        # ═══════════════════════════════════════════════════════════════════
        # Return: No energy restoration - must be done externally
        # ═══════════════════════════════════════════════════════════════════
        return features
    
    def get_unshifted_channels(self, separated_stacked, pos_values):
        """
        Post-processing: unshift channels
        
        Args:
            separated_stacked: (B, P, L*2) separated signals [h_R; h_I]
            pos_values: List of port positions
        
        Returns:
            channels_stacked: (B, P, L*2) unshifted channels
        """
        L = self.seq_len
        channels = []
        
        for p_idx, pos in enumerate(pos_values):
            h_R = separated_stacked[:, p_idx, :L]
            h_I = separated_stacked[:, p_idx, L:]
            
            # Unshift both components
            h_R_unshifted = torch.roll(h_R, shifts=-pos, dims=-1)
            h_I_unshifted = torch.roll(h_I, shifts=-pos, dims=-1)
            
            channels.append(torch.cat([h_R_unshifted, h_I_unshifted], dim=-1))
        
        return torch.stack(channels, dim=1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("="*80)
    print("Residual Refinement Separator (Real-valued, ONNX Compatible)")
    print("="*80)
    
    # Test parameters
    batch_size = 128
    seq_len = 12
    num_ports = 4
    pos_values = [0, 3, 6, 9]
    hidden_dim = 64
    num_stages = 3
    
    print(f"\nConfiguration:")
    print(f"  Batch size:  {batch_size}")
    print(f"  Seq length:  {seq_len}")
    print(f"  Num ports:   {num_ports}")
    print(f"  Positions:   {pos_values}")
    print(f"  Hidden dim:  {hidden_dim}")
    print(f"  Num stages:  {num_stages}")
    
    # Generate complex test data
    print(f"\n{'─'*80}")
    print("Generating Test Data (Complex domain)")
    print(f"{'─'*80}")
    
    h_complex = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64)
    
    # Create mixed signal
    y_complex = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    targets_complex = []
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_complex[:, i], shifts=pos, dims=-1)
        y_complex += shifted
        targets_complex.append(shifted)
    targets_complex = torch.stack(targets_complex, dim=1)  # (B, P, L)
    
    print(f"  h_complex shape:       {h_complex.shape}")
    print(f"  y_complex shape:       {y_complex.shape}")
    print(f"  targets_complex shape: {targets_complex.shape}")
    
    # Convert to real stacked format
    print(f"\n{'─'*80}")
    print("Converting to Real Stacked Format")
    print(f"{'─'*80}")
    
    # Input: [y_R; y_I]
    y_stacked = torch.cat([y_complex.real, y_complex.imag], dim=-1)  # (B, L*2)
    
    # Targets: [h_R; h_I]
    targets_stacked = torch.cat([
        targets_complex.real,
        targets_complex.imag
    ], dim=-1)  # (B, P, L*2)
    
    print(f"  y_stacked shape:       {y_stacked.shape}")
    print(f"  targets_stacked shape: {targets_stacked.shape}")
    
    # Test different configurations
    print(f"\n{'='*80}")
    print("Testing Different Configurations")
    print(f"{'='*80}")
    
    configs = [
        {'share_weights': False, 'activation': 'split_relu'},
        {'share_weights': True, 'activation': 'split_relu'},
        {'share_weights': False, 'activation': 'mod_relu'},
        {'share_weights': False, 'activation': 'cardioid'},
    ]
    
    for config in configs:
        print(f"\n{'─'*80}")
        share = config['share_weights']
        act = config['activation']
        print(f"Config: share_weights={share}, activation={act}")
        print(f"{'─'*80}")
        
        model = ResidualRefinementSeparatorReal(
            seq_len=seq_len,
            num_ports=num_ports,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            share_weights_across_stages=share,
            activation_type=act
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        # Forward pass
        separated_stacked = model(y_stacked)
        print(f"  Output shape: {separated_stacked.shape}")
        
        # Calculate NMSE (in real domain)
        mse = (separated_stacked - targets_stacked).pow(2).mean()
        target_power = targets_stacked.pow(2).mean()
        nmse = mse / (target_power + 1e-8)
        nmse_db = 10 * torch.log10(nmse + 1e-10)
        
        print(f"  Initial NMSE: {nmse.item():.4f} ({nmse_db.item():.2f} dB)")
    
    # Test ONNX export
    print(f"\n{'='*80}")
    print("Testing ONNX Export")
    print(f"{'='*80}")
    
    model = ResidualRefinementSeparatorReal(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=hidden_dim,
        num_stages=num_stages,
        share_weights_across_stages=False,
        activation_type='split_relu'
    )
    model.eval()
    
    dummy_input = torch.randn(1, seq_len * 2)
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            'test_separator_real.onnx',
            export_params=True,
            opset_version=14,
            input_names=['y_stacked'],
            output_names=['h_stacked'],
            dynamic_axes={
                'y_stacked': {0: 'batch_size'},
                'h_stacked': {0: 'batch_size'}
            }
        )
        print("  ✓ ONNX export successful: test_separator_real.onnx")
        print(f"    Input:  y_stacked (batch, {seq_len * 2})")
        print(f"    Output: h_stacked (batch, {num_ports}, {seq_len * 2})")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
    
    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print(f"{'='*80}")
