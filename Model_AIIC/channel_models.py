"""
Channel Models for SRS Channel Estimation

Note: TDL channel models have been moved to tdl_channel.py.
This file only contains legacy/simple channel models.

For TDL channels, use:
    from Model_AIIC.tdl_channel import TDLChannel
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple


class SimpleRayleighChannel:
    """
    Simple Rayleigh fading channel
    
    Generates frequency-selective Rayleigh fading channels using a tap-delay line model.
    
    Args:
        num_taps: Number of channel taps (default: 3)
        tap_power_profile: Power profile for taps (default: exponential decay)
        normalize_channel: Whether to normalize channel energy (default: True)
        dtype: Data type (default: torch.complex64)
        
    Example:
        >>> channel = SimpleRayleighChannel(num_taps=3)
        >>> h = channel.generate(batch_size=32, num_users=4, seq_len=12)
        >>> print(h.shape)  # (32, 4, 12)
    """
    
    def __init__(
        self,
        num_taps: int = 3,
        tap_power_profile: Optional[List[float]] = None,
        normalize_channel: bool = True,
        dtype: torch.dtype = torch.complex64
    ):
        self.num_taps = num_taps
        self.normalize_channel = normalize_channel
        self.dtype = dtype
        
        # Default exponential power decay profile
        if tap_power_profile is None:
            self.tap_power_profile = np.exp(-np.arange(num_taps))
            self.tap_power_profile /= self.tap_power_profile.sum()
        else:
            self.tap_power_profile = np.array(tap_power_profile)
            if len(self.tap_power_profile) != num_taps:
                raise ValueError(f"Power profile length {len(self.tap_power_profile)} != num_taps {num_taps}")
    
    def generate(
        self,
        batch_size: int = 1,
        num_users: int = 1,
        seq_len: int = 12,
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate Rayleigh fading channels
        
        Args:
            batch_size: Number of channel realizations
            num_users: Number of users/ports
            seq_len: Sequence length (time samples)
            return_numpy: If True, return numpy array
            
        Returns:
            h: Channel tensor (batch_size, num_users, seq_len)
        """
        # Generate channel taps (complex Gaussian)
        h_taps = (np.random.randn(batch_size, num_users, self.num_taps) + 
                  1j * np.random.randn(batch_size, num_users, self.num_taps))
        
        # Apply power profile
        h_taps = h_taps * np.sqrt(self.tap_power_profile[None, None, :])
        
        # Zero-pad and take FFT to get frequency response
        h = np.fft.fft(h_taps, n=seq_len, axis=-1)
        
        # Normalize if requested
        if self.normalize_channel:
            h_power = np.mean(np.abs(h) ** 2)
            h = h / np.sqrt(h_power)
        
        if not return_numpy:
            h = torch.from_numpy(h).to(self.dtype)
        
        return h
    
    def __repr__(self):
        return f"SimpleRayleighChannel(num_taps={self.num_taps})"


if __name__ == "__main__":
    """Quick test"""
    
    print("="*80)
    print("Simple Rayleigh Channel Test")
    print("="*80)
    
    # Test: Simple Rayleigh channel
    rayleigh_gen = SimpleRayleighChannel(num_taps=3, normalize_channel=True)
    print(f"Generator: {rayleigh_gen}")
    
    h_rayleigh = rayleigh_gen.generate(batch_size=4, num_users=4, seq_len=12)
    print(f"Generated channels: {h_rayleigh.shape}")
    print(f"Channel power: {h_rayleigh.abs().pow(2).mean():.4f}")
    print(f"dtype: {h_rayleigh.dtype}")
    
    print("\n" + "="*80)
    print("✓ Test completed!")
    print("="*80)
    print("\nFor TDL channels, use Model_AIIC.tdl_channel.TDLChannel instead.")
