"""
Channel Separator Model for SRS Multi-Port Channel Estimation

Problem:
    y = sum_{p in P} circshift(h_p, p) + noise
    
    Where:
    - y: received signal (time domain, 12 points)
    - h_p: channel of port p (time domain)
    - circshift(h_p, p): circular shift by p positions
    - P: set of active ports (e.g., {0, 2, 6, 8})

Goal:
    Separate the mixed signal y into individual shifted channel components.
    
Output:
    circshift(h_0, 0), circshift(h_2, 2), circshift(h_6, 6), circshift(h_8, 8)
    
    Then post-process: h_p = circshift(output_p, -p)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualRefinementSeparator(nn.Module):
    """
    Per-port independent channel separator with residual coupling
    
    Architecture:
    - Each port has its own MLP (port间参数独立)
    - Same port across stages can share weights (可选)
    - Ports are coupled only through residual correction
    
    Args:
        seq_len: Sequence length (default: 12)
        num_ports: Number of ports (default: 4)
        hidden_dim: Hidden dimension for MLPs (default: 64)
        num_stages: Number of refinement stages (default: 3)
        share_weights_across_stages: If True, same port uses same MLP across stages (default: False)
    """
    def __init__(self, seq_len=12, num_ports=4, hidden_dim=64, num_stages=3, 
                 share_weights_across_stages=False, normalize_energy=True):
        super().__init__()
        self.seq_len = seq_len
        self.num_ports = num_ports
        self.num_stages = num_stages
        self.share_weights_across_stages = share_weights_across_stages
        self.normalize_energy = normalize_energy
        
        if share_weights_across_stages:
            # 模式A: 同port不同stage共享参数
            # 只需要为每个port创建一个MLP
            self.port_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(seq_len * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, seq_len * 2)
                )
                for _ in range(num_ports)  # 每个port一个MLP
            ])
        else:
            # 模式B: 每个port每个stage独立参数
            # port_mlps[port_idx][stage_idx] = MLP for that port at that stage
            self.port_mlps = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(seq_len * 2, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, seq_len * 2)
                    )
                    for _ in range(num_stages)  # 每个stage
                ])
                for _ in range(num_ports)  # 每个port
            ])
    
    def forward(self, y):
        """
        Forward pass with per-port independent processing and residual coupling
        
        Args:
            y: (B, L) complex tensor - received mixed signal
            
        Returns:
            features: (B, P, L) complex tensor - separated port signals
        """
        B, L = y.shape
        
        # Energy normalization (optional)
        if self.normalize_energy:
            y_energy = y.abs().pow(2).mean(dim=-1, keepdim=True).sqrt()  # (B, 1)
            y_normalized = y / (y_energy + 1e-8)
        else:
            y_normalized = y
            y_energy = torch.ones(B, 1, device=y.device, dtype=y.real.dtype)
        
        # Initialize: all ports start with normalized y
        features_real = y_normalized.real.unsqueeze(1).repeat(1, self.num_ports, 1)  # (B, P, L)
        features_imag = y_normalized.imag.unsqueeze(1).repeat(1, self.num_ports, 1)  # (B, P, L)
        
        # Iterative refinement through stages
        for stage_idx in range(self.num_stages):
            # Temporary storage for this stage's outputs
            new_features_real = []
            new_features_imag = []
            
            # Each port processes independently through its own MLP
            for port_idx in range(self.num_ports):
                # Input: this port's current features (real + imag)
                x = torch.cat([
                    features_real[:, port_idx],  # (B, L)
                    features_imag[:, port_idx]   # (B, L)
                ], dim=-1)  # (B, L*2)
                
                # Forward through this port's MLP
                if self.share_weights_across_stages:
                    # 共享模式：同port所有stage用同一个MLP
                    mlp = self.port_mlps[port_idx]
                else:
                    # 独立模式：每个port每个stage独立MLP
                    mlp = self.port_mlps[port_idx][stage_idx]
                
                output = mlp(x)  # (B, L*2)
                
                # Split into real and imaginary parts
                out_real = output[:, :L]
                out_imag = output[:, L:]
                
                new_features_real.append(out_real)
                new_features_imag.append(out_imag)
            
            # Stack all ports: (B, P, L)
            features_real = torch.stack(new_features_real, dim=1)
            features_imag = torch.stack(new_features_imag, dim=1)
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # Residual correction: couple ports (allows denoising)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # Reconstruct y from all ports
            y_recon_real = features_real.sum(dim=1)  # (B, L)
            y_recon_imag = features_imag.sum(dim=1)  # (B, L)
            
            # Compute residual (use normalized y)
            residual_real = y_normalized.real - y_recon_real    # (B, L)
            residual_imag = y_normalized.imag - y_recon_imag    # (B, L)
            
            # Add residual directly to all ports (no division)
            # This allows model to denoise: sum(outputs) may not equal y
            features_real = features_real + residual_real.unsqueeze(1)
            features_imag = features_imag + residual_imag.unsqueeze(1)
        
        # Combine real and imaginary parts
        features = torch.complex(features_real, features_imag)
        
        # Restore energy
        if self.normalize_energy:
            features = features * y_energy.unsqueeze(-1)
        
        return features
    
    def get_unshifted_channels(self, separated, pos_values):
        """Post-processing to get unshifted channels"""
        channels = []
        for p_idx, pos in enumerate(pos_values):
            h_p = torch.roll(separated[:, p_idx], shifts=-pos, dims=-1)
            channels.append(h_p)
        return torch.stack(channels, dim=1)


if __name__ == "__main__":
    """Quick test"""
    
    # Test parameters
    batch_size = 128
    seq_len = 12
    num_ports = 4
    pos_values = [0, 2, 6, 8]
    
    print("="*80)
    print("Residual Refinement Channel Separator Test")
    print("="*80)
    
    # Generate synthetic data with SNR control
    snr_db = 20.0  # Can be scalar or list [snr0, snr1, snr2, snr3]
    
    # Generate base channels
    h_base = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64)
    
    # Generate noise with unit power
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()  # Unit power noise
    
    # Adjust signal power based on SNR
    if isinstance(snr_db, (list, tuple)):
        # Different SNR for each port
        h_true = torch.zeros_like(h_base)
        for i in range(num_ports):
            signal_power = 10 ** (snr_db[i] / 10)
            h_true[:, i] = h_base[:, i] * signal_power.sqrt()
    else:
        # Same SNR for all ports
        signal_power = 10 ** (snr_db / 10)
        h_true = h_base * signal_power.sqrt()
    
    # Create mixed signal with shifted channels + noise
    y_clean = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    targets = []
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_true[:, i], shifts=pos, dims=-1)
        y_clean += shifted
        targets.append(shifted)
    targets = torch.stack(targets, dim=1)  # (B, P, L)
    
    # Add noise
    y = y_clean + noise
    
    # Test both modes
    num_stages = 3
    hidden_dim = 64
    
    print(f"\n{'='*80}")
    print("Testing Both Modes")
    print(f"{'='*80}")
    
    # Calculate expected parameters
    params_per_mlp = (seq_len * 2 * hidden_dim + hidden_dim) + \
                     (hidden_dim * hidden_dim + hidden_dim) + \
                     (hidden_dim * seq_len * 2 + seq_len * 2)
    
    print(f"\nPer-MLP parameters: {params_per_mlp:,}")
    print(f"  Input layer:  {seq_len * 2} × {hidden_dim} + {hidden_dim} = {seq_len * 2 * hidden_dim + hidden_dim:,}")
    print(f"  Hidden layer: {hidden_dim} × {hidden_dim} + {hidden_dim} = {hidden_dim * hidden_dim + hidden_dim:,}")
    print(f"  Output layer: {hidden_dim} × {seq_len * 2} + {seq_len * 2} = {hidden_dim * seq_len * 2 + seq_len * 2:,}")
    
    for share_weights in [False, True]:
        print(f"\n{'─'*80}")
        mode_name = "Shared" if share_weights else "Independent"
        print(f"Mode: {mode_name} (share_weights_across_stages={share_weights})")
        print(f"{'─'*80}")
        
        model = ResidualRefinementSeparator(
            seq_len=seq_len, 
            num_ports=num_ports, 
            hidden_dim=hidden_dim,
            num_stages=num_stages,
            share_weights_across_stages=share_weights
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        if share_weights:
            expected = params_per_mlp * num_ports
            print(f"  Expected: {params_per_mlp:,} × {num_ports} ports = {expected:,}")
        else:
            expected = params_per_mlp * num_ports * num_stages
            print(f"  Expected: {params_per_mlp:,} × {num_ports} ports × {num_stages} stages = {expected:,}")
        
        print(f"  Actual:   {num_params:,}")
        print(f"  Match:    {'✓' if num_params == expected else '✗'}")
        
        # Test forward pass
        separated = model(y)
        
        # Calculate NMSE (逐点比较 separated vs targets)
        mse = (separated - targets).abs().pow(2).mean()
        signal_power = targets.abs().pow(2).mean()
        nmse = mse / (signal_power + 1e-8)
        nmse_db = 10 * torch.log10(nmse + 1e-10)
        
        # Also check reconstruction error (optional)
        y_recon = separated.sum(dim=1)
        recon_mse = (y - y_recon).abs().pow(2).mean()
        
        print(f"  Output shape: {separated.shape}")
        print(f"  NMSE:         {nmse_db:.2f} dB")
        print(f"  Recon MSE:    {recon_mse:.6f}")
    
    print(f"\n{'='*80}")
    print("✓ Both modes tested successfully!")
    print(f"{'='*80}")
