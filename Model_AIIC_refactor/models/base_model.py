"""
Base model class for all channel separator models.

All models must inherit from BaseSeparatorModel and implement:
- forward(y) -> h
- from_config(config) -> model
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseSeparatorModel(nn.Module, ABC):
    """
    Base class for all channel separator models
    
    Enforces unified interface:
    - Input:  y (B, L) or (B, L*2) - mixed signal
    - Output: h (B, P, L) or (B, P, L*2) - separated channels
    
    All subclasses must implement forward() and from_config()
    """
    
    def __init__(self, seq_len: int, num_ports: int, normalize_energy: bool = True):
        """
        Args:
            seq_len: Sequence length
            num_ports: Number of ports
        """
        super().__init__()
        self.seq_len = seq_len
        self.num_ports = num_ports
        self.normalize_energy = normalize_energy

    def normalize_input_energy(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each sample to unit RMS power while keeping a restoration scale.

        For real-stacked inputs [real; imag], RMS is computed in the complex domain:
        mean(real^2 + imag^2) over the sequence length, matching complex |y|^2.
        """
        scale_dtype = y.real.dtype if torch.is_complex(y) else y.dtype
        if not self.normalize_energy:
            scale = torch.ones(y.shape[0], 1, device=y.device, dtype=scale_dtype)
            return y, scale

        if torch.is_complex(y):
            rms = y.abs().pow(2).mean(dim=-1, keepdim=True).sqrt()
        else:
            y_real = y[..., :self.seq_len]
            y_imag = y[..., self.seq_len:]
            rms = (y_real.pow(2) + y_imag.pow(2)).mean(dim=-1, keepdim=True).sqrt()

        normalized = y / (rms + 1e-8)
        return normalized, rms.to(scale_dtype)

    def restore_output_energy(self, separated: torch.Tensor, input_rms: torch.Tensor) -> torch.Tensor:
        """Restore the original input RMS to separated outputs."""
        if not self.normalize_energy:
            return separated
        return separated * input_rms.unsqueeze(1)
    
    @abstractmethod
    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - must be implemented by subclass
        
        Args:
            y: Mixed signal tensor
               - (B, L) complex or
               - (B, L*2) real stacked [real; imag]
        
        Returns:
            h: Separated channels
               - (B, P, L) complex or
               - (B, P, L*2) real stacked
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Create model instance from configuration dictionary
        
        Args:
            config: Configuration dictionary with model parameters
        
        Returns:
            model: Model instance
        
        Example:
            >>> config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 64}
            >>> model = Separator1.from_config(config)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information for logging and debugging
        
        Returns:
            info: Dictionary with model information
        """
        return {
            'model_class': self.__class__.__name__,
            'seq_len': self.seq_len,
            'num_ports': self.num_ports,
            'normalize_energy': self.normalize_energy,
            'num_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def get_unshifted_channels(self, separated: torch.Tensor, pos_values: list) -> torch.Tensor:
        """
        Post-processing: Un-shift separated channels
        
        Args:
            separated: (B, P, L) or (B, P, L*2) separated shifted channels
            pos_values: List of port positions, e.g., [0, 3, 6, 9]
        
        Returns:
            channels: (B, P, L) or (B, P, L*2) un-shifted channels
        """
        channels = []
        for p_idx, pos in enumerate(pos_values):
            h_p = torch.roll(separated[:, p_idx], shifts=-pos, dims=-1)
            channels.append(h_p)
        return torch.stack(channels, dim=1)
