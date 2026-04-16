"""
Separator1: Dual-Path Real MLP Channel Separator

This model uses two independent real-valued MLPs to process real and imaginary
parts separately. It's the simpler architecture with slightly more parameters
but cleaner implementation.

Key features:
- Two independent MLPs (one for real, one for imaginary)
- Optional hidden LayerNorm before activation
- Optional hidden ReLU activation
- PyTorch native complex tensor support
- Per-port residual refinement
- Optional per-sample RMS normalization inside the model
"""

import torch
import torch.nn as nn
from .base_model import BaseSeparatorModel


class Separator1(BaseSeparatorModel):
    """
    Dual-Path Real MLP Channel Separator
    
    Architecture:
    - Each port has separate MLPs for real and imaginary parts
    - Ports coupled through residual correction
    - Uses PyTorch complex tensors internally
    
    Parameter Count: ~120k (num_ports=4, stages=3, hidden_dim=64, mlp_depth=3, share_weights=False)
    
    Args:
        seq_len: Sequence length (default: 12)
        num_ports: Number of ports (default: 4)
        hidden_dim: Hidden dimension for MLPs (default: 64)
        num_stages: Number of refinement stages (default: 3)
        mlp_depth: MLP depth - total layers including input/output (default: 3)
                   - 2: Input -> Output (no hidden layer)
                   - 3: Input -> Hidden -> Output (1 hidden layer, default)
                   - 4: Input -> Hidden1 -> Hidden2 -> Output (2 hidden layers)
        share_weights_across_stages: If True, same port uses same MLP across stages (default: False)
        use_hidden_layer_norm: Apply LayerNorm after each hidden linear layer (default: True)
        use_hidden_relu: Apply ReLU after each hidden normalization step (default: True)
    
    Input/Output:
        Input:  y (B, L) complex tensor or (B, 2L) real stacked mixed signal
        Output: h (B, P, L) complex tensor - separated channels
    """
    
    def __init__(self, seq_len=12, num_ports=4, hidden_dim=64, num_stages=3,
                 mlp_depth=3, share_weights_across_stages=False,
                 use_hidden_layer_norm=True, use_hidden_relu=True,
                 normalize_energy=True):
        super().__init__(seq_len, num_ports, normalize_energy=normalize_energy)
        
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.mlp_depth = mlp_depth
        self.share_weights_across_stages = share_weights_across_stages
        self.use_hidden_layer_norm = use_hidden_layer_norm
        self.use_hidden_relu = use_hidden_relu
        
        if share_weights_across_stages:
            # Mode A: Same port shares weights across stages
            self.port_mlps = nn.ModuleList([
                self._create_dual_path_mlp(seq_len, hidden_dim, mlp_depth)
                for _ in range(num_ports)
            ])
        else:
            # Mode B: Each port-stage combination has independent weights
            self.port_mlps = nn.ModuleList([
                nn.ModuleList([
                    self._create_dual_path_mlp(seq_len, hidden_dim, mlp_depth)
                    for _ in range(num_stages)
                ])
                for _ in range(num_ports)
            ])
    
    def _create_dual_path_mlp(self, seq_len, hidden_dim, mlp_depth):
        """
        Create Dual-Path MLP for complex signal processing
        
        Processes real and imaginary parts with separate MLPs.
        
        Args:
            seq_len: Sequence length
            hidden_dim: Hidden dimension
            mlp_depth: Total number of layers (input + hidden + output)
        
        Returns:
            DualPathMLP module
        """
        class DualPathMLP(nn.Module):
            def __init__(self, seq_len, hidden_dim, mlp_depth, use_hidden_layer_norm, use_hidden_relu):
                super().__init__()
                
                if mlp_depth < 2:
                    raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")
                
                num_hidden = mlp_depth - 2
                
                self.use_hidden_layer_norm = use_hidden_layer_norm
                self.use_hidden_relu = use_hidden_relu
                self.mlp_real = self._build_branch(seq_len, hidden_dim, num_hidden)
                self.mlp_imag = self._build_branch(seq_len, hidden_dim, num_hidden)

            def _build_branch(self, seq_len, hidden_dim, num_hidden):
                layers = []
                input_dim = seq_len * 2
                hidden_layers = num_hidden + 1

                for hidden_idx in range(hidden_layers):
                    in_features = input_dim if hidden_idx == 0 else hidden_dim
                    layers.append(nn.Linear(in_features, hidden_dim))
                    if self.use_hidden_layer_norm:
                        layers.append(nn.LayerNorm(hidden_dim))
                    if self.use_hidden_relu:
                        layers.append(nn.ReLU())

                layers.append(nn.Linear(hidden_dim, seq_len))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                # x: (B, L*2) real-stacked in block format [real_part, imag_part]
                # Process directly without converting to complex tensors
                out_real = self.mlp_real(x)  # (B, L)
                out_imag = self.mlp_imag(x)  # (B, L)
                # Return in the same block-stacked format [real_part, imag_part]
                return torch.cat([out_real, out_imag], dim=-1)  # (B, L*2)
        
        return DualPathMLP(
            seq_len,
            hidden_dim,
            mlp_depth,
            self.use_hidden_layer_norm,
            self.use_hidden_relu,
        )
    
    def forward(self, y):
        """
        Forward pass with per-port processing and residual refinement
        
        Args:
            y: (B, L*2) real stacked [y_R; y_I] or (B, L) complex
        
        Returns:
            h: (B, P, L*2) real stacked, or (B, P, L) complex if input is complex
        """
        return_complex = torch.is_complex(y)
        y, input_rms = self.normalize_input_energy(y)
        if return_complex:
            y = torch.cat([y.real, y.imag], dim=-1)
        elif not torch.jit.is_tracing() and y.shape[-1] != self.seq_len * 2:
            raise ValueError(
                f"Expected real-stacked input with last dim {self.seq_len * 2}, got {tuple(y.shape)}"
            )
        
        # Initialize: all ports start with input y
        features = y.unsqueeze(1).repeat(1, self.num_ports, 1)  # (B, P, L*2)
        
        # Iterative refinement through stages
        for stage_idx in range(self.num_stages):
            new_features = []
            
            # Process each port independently
            for port_idx in range(self.num_ports):
                x = features[:, port_idx]  # (B, L*2)
                
                # Select MLP based on sharing mode
                if self.share_weights_across_stages:
                    mlp = self.port_mlps[port_idx]
                else:
                    mlp = self.port_mlps[port_idx][stage_idx]
                
                output = mlp(x)
                new_features.append(output)
            
            features = torch.stack(new_features, dim=1)  # (B, P, L*2)
            
            # Residual correction
            y_recon = features.sum(dim=1)  # (B, L*2)
            residual = y - y_recon  # (B, L*2)
            features = features + residual.unsqueeze(1)  # Broadcast residual

        if return_complex:
            features = torch.complex(features[..., :self.seq_len], features[..., self.seq_len:])

        return self.restore_output_energy(features, input_rms)
    
    @classmethod
    def from_config(cls, config):
        """
        Create model from configuration dictionary
        
        Args:
            config: Configuration dict with keys:
                   - seq_len
                   - num_ports
                   - hidden_dim (optional, default: 64)
                   - num_stages (optional, default: 3)
                   - mlp_depth (optional, default: 3)
                   - share_weights_across_stages (optional, default: False)
                   - use_hidden_layer_norm (optional, default: True)
                   - use_hidden_relu (optional, default: True)
        
        Returns:
            model: Separator1 instance
        """
        return cls(
            seq_len=config['seq_len'],
            num_ports=config['num_ports'],
            hidden_dim=config.get('hidden_dim', 64),
            num_stages=config.get('num_stages', 3),
            mlp_depth=config.get('mlp_depth', 3),
            share_weights_across_stages=config.get('share_weights_across_stages', False),
            use_hidden_layer_norm=config.get('use_hidden_layer_norm', True),
            use_hidden_relu=config.get('use_hidden_relu', True),
            normalize_energy=config.get('normalize_energy', True),
        )
