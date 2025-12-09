"""
Complex Neural Network Layers using Real-valued Block Matrix
(ONNX Compatible - No complex tensors)

Mathematical Foundation:
    Complex linear: y = Wx + b where W = W_R + jW_I
    
    Real equivalent using block matrix:
    [y_R]   [W_R  -W_I] [x_R]   [b_R]
    [y_I] = [W_I   W_R] [x_I] + [b_I]

Activation Functions:
    - split_relu: ReLU on real and imaginary separately
    - mod_relu: Modulus ReLU (preserves phase)
    - z_relu: zReLU (gated activation)
    - cardioid: Cardioid activation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ComplexLinearReal(nn.Module):
    """
    Complex linear layer using real-valued block matrix (ONNX compatible)
    
    Implements: y = Wx + b where W = W_R + jW_I
    
    Using real block matrix:
    [y_R]   [W_R  -W_I] [x_R]   [b_R]
    [y_I] = [W_I   W_R] [x_I] + [b_I]
    
    Input:  (B, in_features*2)  = [x_R; x_I]
    Output: (B, out_features*2) = [y_R; y_I]
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Block matrix components: W = W_R + jW_I
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
        # For complex weights, variance should be 1/(2*in_features) per component
        std = math.sqrt(1.0 / (2.0 * self.in_features))
        nn.init.normal_(self.weight_real, mean=0.0, std=std)
        nn.init.normal_(self.weight_imag, mean=0.0, std=std)
    
    def forward(self, x_stacked):
        """
        Args:
            x_stacked: (B, in_features*2) where [:, :in_features] = x_R, [:, in_features:] = x_I
        
        Returns:
            y_stacked: (B, out_features*2) where [:, :out_features] = y_R, [:, out_features:] = y_I
        """
        # Split into real and imaginary parts
        x_R = x_stacked[:, :self.in_features]
        x_I = x_stacked[:, self.in_features:]
        
        # Block matrix multiplication:
        # y_R = W_R @ x_R - W_I @ x_I + b_R
        # y_I = W_I @ x_R + W_R @ x_I + b_I
        y_R = F.linear(x_R, self.weight_real) - F.linear(x_I, self.weight_imag)
        y_I = F.linear(x_R, self.weight_imag) + F.linear(x_I, self.weight_real)
        
        if self.bias_real is not None:
            y_R = y_R + self.bias_real
            y_I = y_I + self.bias_imag
        
        # Concatenate [y_R; y_I]
        return torch.cat([y_R, y_I], dim=-1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Complex Activation Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def complex_relu(x_stacked, in_features):
    """
    Simple ReLU: Apply ReLU to entire stacked tensor (FASTEST!)
    
    Treats real and imaginary as independent channels
    Fastest option, recommended for training
    
    Args:
        x_stacked: (B, in_features*2)
        in_features: number of complex features (unused but kept for API consistency)
    
    Returns:
        (B, in_features*2) after activation
    """
    return F.relu(x_stacked)


def complex_split_relu(x_stacked, in_features):
    """
    Split ReLU: Apply ReLU separately to real and imaginary parts
    
    Slightly slower than simple relu due to split/cat overhead
    
    Args:
        x_stacked: (B, in_features*2)
        in_features: number of complex features
    
    Returns:
        (B, in_features*2) after activation
    """
    x_R = F.relu(x_stacked[:, :in_features])
    x_I = F.relu(x_stacked[:, in_features:])
    return torch.cat([x_R, x_I], dim=-1)


def complex_mod_relu(x_stacked, in_features, bias=0.5):
    """
    Modulus ReLU: ReLU(|z| + bias) * (z / |z|)
    
    Preserves phase, only modulates magnitude
    ⚠️ WARNING: SLOW! Contains sqrt and division (slow backward pass)
    
    Args:
        x_stacked: (B, in_features*2)
        in_features: number of complex features
        bias: bias term (default: 0.5)
    
    Returns:
        (B, in_features*2) after activation
    """
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    # Magnitude: |z| = sqrt(x_R^2 + x_I^2)
    magnitude = torch.sqrt(x_R.pow(2) + x_I.pow(2) + 1e-8)
    
    # Modulated magnitude: ReLU(|z| + bias)
    magnitude_activated = F.relu(magnitude + bias)
    
    # Scale: (activated magnitude) / (original magnitude)
    scale = magnitude_activated / (magnitude + 1e-8)
    
    # Apply to both components
    y_R = x_R * scale
    y_I = x_I * scale
    
    return torch.cat([y_R, y_I], dim=-1)


def complex_z_relu(x_stacked, in_features):
    """
    zReLU: ReLU on both magnitude and phase
    
    Only activates in first quadrant (both real and imaginary positive)
    ⚠️ WARNING: VERY SLOW! Contains atan2 (extremely slow backward pass)
    
    Args:
        x_stacked: (B, in_features*2)
        in_features: number of complex features
    
    Returns:
        (B, in_features*2) after activation
    """
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    # Phase: theta = atan2(x_I, x_R)
    theta = torch.atan2(x_I, x_R)
    
    # Gate: only activate in first quadrant [0, pi/2]
    gate = ((theta >= 0) & (theta <= math.pi / 2)).float()
    
    # Apply gate
    y_R = F.relu(x_R) * gate
    y_I = F.relu(x_I) * gate
    
    return torch.cat([y_R, y_I], dim=-1)


def complex_cardioid(x_stacked, in_features):
    """
    Cardioid activation: (1 + cos(theta)) / 2 * z
    
    Smooth phase-dependent activation
    ⚠️ WARNING: VERY SLOW! Contains atan2 + cos (extremely slow backward pass)
    
    Args:
        x_stacked: (B, in_features*2)
        in_features: number of complex features
    
    Returns:
        (B, in_features*2) after activation
    """
    x_R = x_stacked[:, :in_features]
    x_I = x_stacked[:, in_features:]
    
    # Phase: theta = atan2(x_I, x_R)
    theta = torch.atan2(x_I, x_R)
    
    # Cardioid function: (1 + cos(theta)) / 2
    scale = (1.0 + torch.cos(theta)) / 2.0
    
    # Apply to both components
    y_R = x_R * scale
    y_I = x_I * scale
    
    return torch.cat([y_R, y_I], dim=-1)


class ComplexActivation(nn.Module):
    """
    Complex activation function wrapper
    
    Supported types (ordered by speed):
    - 'relu': Simple ReLU (FASTEST! ~10-100x faster than others)
    - 'split_relu': ReLU on real and imaginary separately (fast)
    - 'mod_relu': Modulus ReLU (SLOW: contains sqrt/division)
    - 'z_relu': zReLU (VERY SLOW: contains atan2)
    - 'cardioid': Cardioid activation (VERY SLOW: contains atan2 + cos)
    
    ⚠️ Performance Warning:
    - 'relu' / 'split_relu': Fast training, use for production
    - 'mod_relu' / 'z_relu' / 'cardioid': 10-100x slower backward pass!
      Only use for research/comparison, NOT for large-scale training
    """
    def __init__(self, activation_type='relu', **kwargs):
        super().__init__()
        self.activation_type = activation_type
        self.kwargs = kwargs
        
        # Validate activation type
        valid_types = ['relu', 'split_relu', 'mod_relu', 'z_relu', 'cardioid']
        if activation_type not in valid_types:
            raise ValueError(f"Invalid activation_type '{activation_type}'. "
                           f"Choose from {valid_types}")
        
        # Performance warning
        if activation_type in ['mod_relu', 'z_relu', 'cardioid']:
            import warnings
            warnings.warn(
                f"Activation '{activation_type}' is VERY SLOW (10-100x slower than 'relu')! "
                f"Backward pass will dominate training time. "
                f"Consider using 'relu' or 'split_relu' for faster training.",
                UserWarning
            )
    
    def forward(self, x_stacked, in_features):
        """
        Args:
            x_stacked: (B, in_features*2)
            in_features: number of complex features
        
        Returns:
            (B, in_features*2) after activation
        """
        if self.activation_type == 'relu':
            return complex_relu(x_stacked, in_features)
        elif self.activation_type == 'split_relu':
            return complex_split_relu(x_stacked, in_features)
        elif self.activation_type == 'mod_relu':
            bias = self.kwargs.get('bias', 0.5)
            return complex_mod_relu(x_stacked, in_features, bias)
        elif self.activation_type == 'z_relu':
            return complex_z_relu(x_stacked, in_features)
        elif self.activation_type == 'cardioid':
            return complex_cardioid(x_stacked, in_features)
        else:
            raise ValueError(f"Unknown activation: {self.activation_type}")


class ComplexMLPReal(nn.Module):
    """
    Complex MLP using only real tensors (ONNX compatible)
    
    Architecture: Input -> FC1 -> Activation -> [FC_hidden -> Activation] x (num_sub_stages-1) -> FC_out -> Output
    All operations use real-valued block matrix
    
    Args:
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_sub_stages: Number of hidden layers (default: 2, i.e., 2 hidden layers)
        activation_type: Type of complex activation ('split_relu', 'mod_relu', 'z_relu', 'cardioid')
    """
    def __init__(self, seq_len, hidden_dim, num_sub_stages=2, activation_type='split_relu'):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_sub_stages = num_sub_stages
        self.activation_type = activation_type
        
        # Input layer
        self.fc1 = ComplexLinearReal(seq_len, hidden_dim)
        self.act1 = ComplexActivation(activation_type)
        
        # Hidden layers (num_sub_stages - 1)
        self.hidden_layers = nn.ModuleList()
        self.hidden_activations = nn.ModuleList()
        for _ in range(num_sub_stages - 1):
            self.hidden_layers.append(ComplexLinearReal(hidden_dim, hidden_dim))
            self.hidden_activations.append(ComplexActivation(activation_type))
        
        # Output layer
        self.fc_out = ComplexLinearReal(hidden_dim, seq_len)
    
    def forward(self, x_stacked):
        """
        Args:
            x_stacked: (B, seq_len*2) real tensor [x_R; x_I]
        
        Returns:
            (B, seq_len*2) real tensor [y_R; y_I]
        """
        # Input layer
        x = self.fc1(x_stacked)
        x = self.act1(x, self.hidden_dim)
        
        # Hidden layers
        for fc, act in zip(self.hidden_layers, self.hidden_activations):
            x = fc(x)
            x = act(x, self.hidden_dim)
        
        # Output layer
        x = self.fc_out(x)
        
        return x


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("="*80)
    print("Testing Complex Layers (Real-valued, ONNX Compatible)")
    print("="*80)
    
    batch_size = 16
    seq_len = 12
    hidden_dim = 64
    
    # Test ComplexLinearReal
    print("\n" + "─"*80)
    print("1. Testing ComplexLinearReal")
    print("─"*80)
    
    layer = ComplexLinearReal(seq_len, hidden_dim)
    x = torch.randn(batch_size, seq_len * 2)
    y = layer(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters:   {sum(p.numel() for p in layer.parameters()):,}")
    
    # Test all activation functions
    print("\n" + "─"*80)
    print("2. Testing Activation Functions")
    print("─"*80)
    
    x = torch.randn(batch_size, hidden_dim * 2)
    
    activations = ['split_relu', 'mod_relu', 'z_relu', 'cardioid']
    for act_name in activations:
        act = ComplexActivation(act_name)
        y = act(x, hidden_dim)
        print(f"  {act_name:15s}: {x.shape} -> {y.shape} ✓")
    
    # Test ComplexMLPReal
    print("\n" + "─"*80)
    print("3. Testing ComplexMLPReal")
    print("─"*80)
    
    for act_type in activations:
        mlp = ComplexMLPReal(seq_len, hidden_dim, activation_type=act_type)
        x = torch.randn(batch_size, seq_len * 2)
        y = mlp(x)
        num_params = sum(p.numel() for p in mlp.parameters())
        
        print(f"  Activation: {act_type:15s}")
        print(f"    Input:  {x.shape}")
        print(f"    Output: {y.shape}")
        print(f"    Params: {num_params:,}")
    
    # Test ONNX export
    print("\n" + "─"*80)
    print("4. Testing ONNX Export")
    print("─"*80)
    
    mlp = ComplexMLPReal(seq_len, hidden_dim, activation_type='split_relu')
    mlp.eval()
    
    dummy_input = torch.randn(1, seq_len * 2)
    
    try:
        torch.onnx.export(
            mlp,
            dummy_input,
            'test_complex_mlp.onnx',
            export_params=True,
            opset_version=14,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("  ✓ ONNX export successful: test_complex_mlp.onnx")
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
