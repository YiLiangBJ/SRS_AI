"""
Separator2: ComplexLinear Channel Separator (ONNX Compatible)

This model uses ComplexLinearReal layers with shared weight matrices.
It's more parameter-efficient and supports ONNX export for MATLAB deployment.

Key features:
- ComplexLinear layers (weight_real, weight_imag)
- Multiple complex activation functions
- ONNX Opset 9 compatible
- ~20% less parameters than Separator1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseSeparatorModel


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Complex Layers (embedded in this file for Separator2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ComplexLinearReal(nn.Module):
    """
    Complex linear layer using real-valued block matrix
    
    Implements: y = Wx + b where W = W_R + jW_I
    
    Block matrix form:
    [y_R]   [W_R  -W_I] [x_R]   [b_R]
    [y_I] = [W_I   W_R] [x_I] + [b_I]
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Xavier initialization for complex weights"""
        std = math.sqrt(1.0 / (2.0 * self.in_features))
        nn.init.normal_(self.weight_real, mean=0.0, std=std)
        nn.init.normal_(self.weight_imag, mean=0.0, std=std)
    
    def forward(self, x_stacked):
        """
        Args:
            x_stacked: (B, in_features*2) = [x_R; x_I]
        Returns:
            y_stacked: (B, out_features*2) = [y_R; y_I]
        """
        x_R = x_stacked[:, :self.in_features]
        x_I = x_stacked[:, self.in_features:]
        
        y_R = F.linear(x_R, self.weight_real) - F.linear(x_I, self.weight_imag)
        y_I = F.linear(x_R, self.weight_imag) + F.linear(x_I, self.weight_real)
        
        if self.bias_real is not None:
            y_R = y_R + self.bias_real
            y_I = y_I + self.bias_imag
        
        return torch.cat([y_R, y_I], dim=-1)


# Complex activation functions
def complex_relu(x_stacked, in_features):
    """Simple ReLU on entire tensor (FASTEST)"""
    return F.relu(x_stacked)


def complex_split_relu(x_stacked, in_features):
    """ReLU separately on real and imaginary"""
    x_R = F.relu(x_stacked[:, :in_features])
    x_I = F.relu(x_stacked[:, in_features:])
    return torch.cat([x_R, x_I], dim=-1)


def complex_mod_relu(x_stacked, in_features, bias=0.5):
    """Modulus ReLU (preserves phase, slow)"""
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    magnitude = torch.sqrt(x_R.pow(2) + x_I.pow(2) + 1e-8)
    magnitude_activated = F.relu(magnitude + bias)
    scale = magnitude_activated / (magnitude + 1e-8)
    
    return torch.cat([x_R * scale, x_I * scale], dim=-1)


def complex_z_relu(x_stacked, in_features):
    """zReLU (first quadrant only, very slow)"""
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    theta = torch.atan2(x_I, x_R)
    gate = ((theta >= 0) & (theta <= math.pi / 2)).float()
    
    return torch.cat([F.relu(x_R) * gate, F.relu(x_I) * gate], dim=-1)


def complex_cardioid(x_stacked, in_features):
    """Cardioid activation (very slow)"""
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    theta = torch.atan2(x_I, x_R)
    scale = 0.5 * (1 + torch.cos(theta))
    
    return torch.cat([x_R * scale, x_I * scale], dim=-1)


ACTIVATION_FUNCTIONS = {
    'relu': complex_relu,
    'split_relu': complex_split_relu,
    'mod_relu': complex_mod_relu,
    'z_relu': complex_z_relu,
    'cardioid': complex_cardioid
}


class ComplexMLPReal(nn.Module):
    """
    MLP for complex-valued signals using real block matrices
    
    Input/Output: (B, seq_len*2) = [x_R; x_I]
    """
    def __init__(self, seq_len, hidden_dim, mlp_depth=3, activation_type='split_relu'):
        super().__init__()
        
        if mlp_depth < 2:
            raise ValueError(f"mlp_depth must be >= 2 (got {mlp_depth})")
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.mlp_depth = mlp_depth
        self.activation_type = activation_type
        
        if activation_type not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation: {activation_type}")
        
        self.activation_fn = ACTIVATION_FUNCTIONS[activation_type]
        
        # Build layers
        layers = []
        num_hidden = mlp_depth - 2
        
        # Input layer
        layers.append(ComplexLinearReal(seq_len, hidden_dim))
        
        # Hidden layers
        for _ in range(num_hidden):
            layers.append(ComplexLinearReal(hidden_dim, hidden_dim))
        
        # Output layer
        layers.append(ComplexLinearReal(hidden_dim, seq_len))
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, seq_len*2) = [x_R; x_I]
        Returns:
            y: (B, seq_len*2) = [y_R; y_I]
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation except on last layer
            if i < len(self.layers) - 1:
                if self.activation_type == 'relu':
                    x = self.activation_fn(x, None)
                else:
                    x = self.activation_fn(x, layer.out_features)
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Separator2 Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Separator2(BaseSeparatorModel):
    """
    ComplexLinear Channel Separator (ONNX Compatible)
    
    Architecture:
    - ComplexLinearReal layers (shared weight matrices)
    - Per-port residual refinement
    - Multiple activation functions
    - ONNX exportable
    
    Parameter Count: ~107k (num_ports=4, stages=3, hidden_dim=64, mlp_depth=3, share_weights=False)
    
    Args:
        seq_len: Sequence length (default: 12)
        num_ports: Number of ports (default: 4)
        hidden_dim: Hidden dimension (default: 64)
        num_stages: Number of refinement stages (default: 3)
        mlp_depth: MLP depth (default: 3)
        share_weights_across_stages: Share weights across stages (default: False)
        activation_type: Activation function (default: 'relu')
                        Options: 'relu', 'split_relu', 'mod_relu', 'z_relu', 'cardioid'
        onnx_mode: ONNX Opset 9 compatibility mode (default: False)
    
    Input/Output:
        Input:  y (B, L*2) real stacked [y_R; y_I]
        Output: h (B, P, L*2) real stacked [[h0_R; h0_I], ...]
    """
    
    def __init__(self, seq_len=12, num_ports=4, hidden_dim=64, num_stages=3,
                 mlp_depth=3, share_weights_across_stages=False,
                 activation_type='relu', onnx_mode=False):
        super().__init__(seq_len, num_ports)
        
        self.hidden_dim = hidden_dim
        self.num_stages = num_stages
        self.mlp_depth = mlp_depth
        self.share_weights_across_stages = share_weights_across_stages
        self.activation_type = activation_type
        self.onnx_mode = onnx_mode
        
        if share_weights_across_stages:
            self.port_mlps = nn.ModuleList([
                ComplexMLPReal(seq_len, hidden_dim, mlp_depth, activation_type)
                for _ in range(num_ports)
            ])
        else:
            self.port_mlps = nn.ModuleList([
                nn.ModuleList([
                    ComplexMLPReal(seq_len, hidden_dim, mlp_depth, activation_type)
                    for _ in range(num_stages)
                ])
                for _ in range(num_ports)
            ])
    
    def forward(self, y):
        """
        Forward pass
        
        Args:
            y: (B, L*2) real stacked [y_R; y_I]
        
        Returns:
            h: (B, P, L*2) separated channels
        """
        B = y.shape[0]
        P = self.num_ports
        L = self.seq_len
        
        # Initialize features
        if self.onnx_mode:
            features_list = [y.unsqueeze(1) for _ in range(P)]
            features = torch.cat(features_list, dim=1)
        else:
            features = y.unsqueeze(1).repeat(1, P, 1)
        
        # Iterative refinement
        for stage_idx in range(self.num_stages):
            # Process each port
            if self.onnx_mode:
                new_features_list = []
                for port_idx in range(P):
                    x = features[:, port_idx, :]
                    
                    if self.share_weights_across_stages:
                        mlp = self.port_mlps[port_idx]
                    else:
                        mlp = self.port_mlps[port_idx][stage_idx]
                    
                    output = mlp(x)
                    new_features_list.append(output.unsqueeze(1))
                
                features = torch.cat(new_features_list, dim=1)
            else:
                if self.share_weights_across_stages:
                    outputs = [self.port_mlps[p](features[:, p, :]) for p in range(P)]
                else:
                    outputs = [self.port_mlps[p][stage_idx](features[:, p, :]) for p in range(P)]
                
                features = torch.stack(outputs, dim=1)
            
            # Residual correction
            if self.onnx_mode:
                features_R = features[:, :, :L]
                features_I = features[:, :, L:]
                
                y_recon_R = features_R[:, 0, :].clone()
                y_recon_I = features_I[:, 0, :].clone()
                for p in range(1, P):
                    y_recon_R = y_recon_R + features_R[:, p, :]
                    y_recon_I = y_recon_I + features_I[:, p, :]
                
                y_recon = torch.cat([y_recon_R, y_recon_I], dim=-1)
            else:
                features_R, features_I = torch.chunk(features, 2, dim=-1)
                y_recon = torch.cat([features_R.sum(dim=1), features_I.sum(dim=1)], dim=-1)
            
            residual = y - y_recon
            
            if self.onnx_mode:
                features_list = []
                for p in range(P):
                    features_list.append((features[:, p, :] + residual).unsqueeze(1))
                features = torch.cat(features_list, dim=1)
            else:
                features = features + residual.unsqueeze(1)
        
        return features
    
    @classmethod
    def from_config(cls, config):
        """Create model from configuration dictionary"""
        return cls(
            seq_len=config['seq_len'],
            num_ports=config['num_ports'],
            hidden_dim=config.get('hidden_dim', 64),
            num_stages=config.get('num_stages', 3),
            mlp_depth=config.get('mlp_depth', 3),
            share_weights_across_stages=config.get('share_weights_across_stages', False),
            activation_type=config.get('activation_type', 'relu'),
            onnx_mode=config.get('onnx_mode', False)
        )
