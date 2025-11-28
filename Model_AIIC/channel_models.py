"""
Channel Models for SRS Channel Estimation

TDL Channel Generation with Random Timing Offset:

Physical Parameters:
    - seq_len: Number of time-domain samples (default: 12)
    - scs: Subcarrier spacing in Hz (e.g., 15e3, 30e3)
    - Ktc: Comb factor (default: 4, every 4th subcarrier has pilot)
    - T_total = 1/(scs*Ktc): Total time duration
    - Ts = 1/(scs*Ktc*seq_len): Sampling interval
    - Tc = 1/(480e3*4096): 3GPP basic time unit

Timing Offset:
    - Maximum offset: ±256*Tc
    - Normalized: delta = 256*Tc/Ts
    - Applied via frequency domain phase rotation:
      h' = IFFT(FFT(h) .* exp(j*2π/seq_len*k*delta)), k=0,1,...,seq_len-1

Uses Sionna library for standard-compliant TDL channel modeling.
"""

import torch
import numpy as np
from typing import Optional, Union, List, Tuple

try:
    import tensorflow as tf
    # Sionna TDL channel model
    from sionna.phy.channel.tr38901 import TDL
    SIONNA_AVAILABLE = True
except ImportError:
    SIONNA_AVAILABLE = False
    print("Warning: Sionna not available. Install with: pip install sionna tensorflow")


class TDLChannelGenerator:
    """
    TDL Channel Generator with Random Timing Offset for SRS
    
    Generates time-domain channel impulse responses using Sionna's TDL models,
    then applies random timing offset for realistic channel estimation scenarios.
    
    Args:
        tdl_model: TDL model type ('A', 'B', 'C', 'D', 'E') from 3GPP TR 38.901
        delay_spread: RMS delay spread in seconds (default: 30e-9 for 30ns)
        carrier_frequency: Carrier frequency in Hz (default: 3.5e9 for 3.5 GHz)
        scs: Subcarrier spacing in Hz (default: 30e3 for 30 kHz)
        Ktc: Comb factor - pilot every Ktc subcarriers (default: 4)
        seq_len: Number of time samples (default: 12)
        max_timing_offset: Maximum timing offset in units of Tc (default: 256)
                          Tc = 1/(480e3*4096) ≈ 0.509 ns
        add_timing_offset: Whether to add random timing offset (default: True)
        normalize_energy: Whether to normalize channel energy (default: True)
        dtype: PyTorch data type (default: torch.complex64)
        
    Example:
        >>> channel_gen = TDLChannelGenerator(
        ...     model='A',
        ...     delay_spread=30e-9,
        ...     sampling_rate=30.72e6,
        ...     num_time_samples=12
        ... )
        >>> h = channel_gen.generate(batch_size=32, num_users=4)
        >>> print(h.shape)  # (32, 4, 12)
    """
    
    def __init__(
        self,
        tdl_model: str = 'A',
        delay_spread: float = 30e-9,
        carrier_frequency: float = 3.5e9,
        scs: float = 30e3,
        Ktc: int = 4,
        seq_len: int = 12,
        max_timing_offset: float = 256.0,
        add_timing_offset: bool = True,
        normalize_energy: bool = True,
        dtype: torch.dtype = torch.complex64
    ):
        if not SIONNA_AVAILABLE:
            raise ImportError("Sionna is required. Install with: pip install sionna tensorflow")
        
        self.tdl_model = tdl_model.upper()
        self.delay_spread = delay_spread
        self.carrier_frequency = carrier_frequency
        self.scs = scs
        self.Ktc = Ktc
        self.seq_len = seq_len
        self.max_timing_offset = max_timing_offset
        self.add_timing_offset = add_timing_offset
        self.normalize_energy = normalize_energy
        self.dtype = dtype
        
        # Calculate timing parameters
        # Tc: 3GPP basic time unit
        self.Tc = 1.0 / (480e3 * 4096)  # ≈ 0.509 ns
        
        # Ts: Sampling interval
        self.Ts = 1.0 / (scs * Ktc * seq_len)  # seconds
        
        # T_total: Total time duration
        self.T_total = 1.0 / (scs * Ktc)  # seconds
        
        # Effective sampling rate
        self.effective_sampling_rate = seq_len / self.T_total  # = scs * Ktc * seq_len
        
        # Normalized timing offset: delta = max_timing_offset * Tc / Ts
        self.delta_max = max_timing_offset * self.Tc / self.Ts
        self.normalize_channel = normalize_channel
        self.dtype = dtype
        
        # Validate model type
        valid_models = ['A', 'B', 'C', 'D', 'E']
        if self.tdl_model not in valid_models:
            raise ValueError(f"Invalid TDL model '{tdl_model}'. Must be one of {valid_models}")
        
        # Initialize Sionna TDL channel (will be created in generate method)
        self.channel = None
    
    def generate(
        self, 
        batch_size: int = 1, 
        num_ports: int = 4,
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate time-domain channel realizations with random timing offset
        
        Steps:
        1. Generate TDL channel using Sionna (returns CIR in time domain)
        2. Sample/interpolate to seq_len points
        3. Add random timing offset via DFT -> phase rotation -> IDFT
        
        Args:
            batch_size: Number of channel realizations (batches)
            num_ports: Number of ports/users per batch
            return_numpy: If True, return numpy array instead of torch tensor
            
        Returns:
            h: Channel tensor of shape (batch_size, num_ports, seq_len)
               Complex-valued time-domain channel with timing offset
        """
        import tensorflow as tf
        
        # Initialize TDL channel if not already done
        if self.channel is None:
            self.channel = TDL(
                model=self.tdl_model,
                delay_spread=self.delay_spread,
                carrier_frequency=self.carrier_frequency,
                num_rx_ant=1,
                num_tx_ant=1
            )
        
        # Generate channels for all batches and ports
        h_all = []
        
        for _ in range(batch_size):
            h_ports = []
            for _ in range(num_ports):
                # Generate one channel realization using Sionna
                # We generate a longer sequence and then sample to seq_len
                num_samples_sionna = max(self.seq_len * 4, 64)  # Over-sample for better quality
                
                a, tau = self.channel(
                    batch_size=1,
                    num_time_steps=1,
                    sampling_frequency=self.effective_sampling_rate
                )
                # a: complex path gains, shape [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
                # tau: path delays in seconds, shape [batch, num_rx, num_tx, num_paths]
                
                # Extract path gains and delays
                a = a[0, 0, 0, 0, 0, :, 0].numpy()  # [num_paths]
                tau = tau[0, 0, 0, :].numpy()  # [num_paths]
                
                # Generate time-domain CIR by placing path gains at corresponding time indices
                # Time indices for each tap
                t = np.arange(self.seq_len) * self.Ts
                
                # Initialize CIR
                h = np.zeros(self.seq_len, dtype=np.complex64)
                
                # Place each path at its delay position (with interpolation if needed)
                for path_gain, path_delay in zip(a, tau):
                    # Find closest time index
                    idx = np.argmin(np.abs(t - path_delay))
                    if idx < self.seq_len:
                        h[idx] += path_gain
                
                # If no paths landed in the window, put first path at index 0
                if np.abs(h).sum() == 0:
                    h[0] = a[0] if len(a) > 0 else 1.0
                
                # Add random timing offset
                if self.add_timing_offset:
                    # Random offset: delta uniformly in [-delta_max, delta_max]
                    delta = np.random.uniform(-self.delta_max, self.delta_max)
                    
                    # Apply timing offset via frequency domain phase rotation
                    # h' = IFFT(FFT(h) * exp(j*2π*k*delta/seq_len))
                    H = np.fft.fft(h)
                    k = np.arange(self.seq_len)
                    phase_shift = np.exp(1j * 2 * np.pi * k * delta / self.seq_len)
                    H_shifted = H * phase_shift
                    h = np.fft.ifft(H_shifted)
                
                h_ports.append(h)
            
            h_all.append(np.stack(h_ports, axis=0))
        
        # Stack all batches: (batch_size, num_ports, seq_len)
        h = np.stack(h_all, axis=0)
        
        # Normalize energy if requested
        if self.normalize_energy:
            # Normalize per port
            for b in range(batch_size):
                for p in range(num_ports):
                    power = np.mean(np.abs(h[b, p]) ** 2)
                    if power > 0:
                        h[b, p] = h[b, p] / np.sqrt(power)
        
        # Convert to torch tensor
        if not return_numpy:
            h = torch.from_numpy(h).to(self.dtype)
        
        return h
    
    def generate_frequency_response(
        self,
        batch_size: int = 1,
        num_users: int = 1,
        num_subcarriers: int = 12,
        return_numpy: bool = False
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate frequency-domain channel response
        
        Args:
            batch_size: Number of channel realizations
            num_users: Number of users/ports
            num_subcarriers: Number of subcarriers
            return_numpy: If True, return numpy array
            
        Returns:
            H: Frequency response (batch_size, num_users, num_subcarriers)
        """
        # Generate time-domain channel
        h = self.generate(batch_size, num_users, return_numpy=True)
        
        # Convert to frequency domain via FFT
        H = np.fft.fft(h, n=num_subcarriers, axis=-1)
        
        if not return_numpy:
            H = torch.from_numpy(H).to(self.dtype)
        
        return H
    
    def __repr__(self):
        return (f"TDLChannelGenerator(model='TDL-{self.model}', "
                f"delay_spread={self.delay_spread*1e9:.1f}ns, "
                f"fc={self.carrier_frequency/1e9:.2f}GHz, "
                f"fs={self.sampling_rate/1e6:.2f}MHz, "
                f"L={self.num_time_samples})")


class SimpleRayleighChannel:
    """
    Simple Rayleigh fading channel (fallback when Sionna not available)
    
    Args:
        num_taps: Number of channel taps (default: 3)
        tap_power_profile: Power profile for taps (default: exponential decay)
        normalize_channel: Whether to normalize channel energy (default: True)
        dtype: Data type (default: torch.complex64)
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


def create_channel_generator(
    model: str = 'rayleigh',
    **kwargs
) -> Union[TDLChannelGenerator, SimpleRayleighChannel]:
    """
    Factory function to create channel generator
    
    Args:
        model: Channel model type
               - 'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E': 3GPP TDL models
               - 'rayleigh': Simple Rayleigh fading
        **kwargs: Additional arguments for specific channel model
        
    Returns:
        Channel generator instance
        
    Example:
        >>> # TDL channel
        >>> gen = create_channel_generator('TDL-A', delay_spread=30e-9)
        >>> 
        >>> # Rayleigh channel
        >>> gen = create_channel_generator('rayleigh', num_taps=3)
    """
    model_upper = model.upper()
    
    if model_upper.startswith('TDL'):
        # Extract TDL type (A, B, C, D, E)
        tdl_type = model_upper.split('-')[-1] if '-' in model_upper else 'A'
        return TDLChannelGenerator(model=tdl_type, **kwargs)
    elif model_upper == 'RAYLEIGH':
        return SimpleRayleighChannel(**kwargs)
    else:
        raise ValueError(f"Unknown channel model: {model}")


if __name__ == "__main__":
    """Quick test"""
    
    print("="*80)
    print("Channel Models Test")
    print("="*80)
    
    # Test 1: Simple Rayleigh channel (always available)
    print("\n" + "="*80)
    print("Test 1: Simple Rayleigh Channel")
    print("="*80)
    
    rayleigh_gen = SimpleRayleighChannel(num_taps=3, normalize_channel=True)
    print(f"Generator: {rayleigh_gen}")
    
    h_rayleigh = rayleigh_gen.generate(batch_size=4, num_users=4, seq_len=12)
    print(f"Generated channels: {h_rayleigh.shape}")
    print(f"Channel power: {h_rayleigh.abs().pow(2).mean():.4f}")
    print(f"dtype: {h_rayleigh.dtype}")
    
    # Test 2: TDL channel (if Sionna available)
    if SIONNA_AVAILABLE:
        print("\n" + "="*80)
        print("Test 2: TDL-A Channel")
        print("="*80)
        
        try:
            tdl_gen = TDLChannelGenerator(
                model='A',
                delay_spread=30e-9,
                carrier_frequency=3.5e9,
                sampling_rate=30.72e6,
                num_time_samples=12
            )
            print(f"Generator: {tdl_gen}")
            
            h_tdl = tdl_gen.generate(batch_size=4, num_users=4)
            print(f"Generated channels: {h_tdl.shape}")
            print(f"Channel power: {h_tdl.abs().pow(2).mean():.4f}")
            print(f"dtype: {h_tdl.dtype}")
        except Exception as e:
            print(f"TDL generation failed: {e}")
    else:
        print("\n" + "="*80)
        print("Test 2: TDL Channel - SKIPPED (Sionna not available)")
        print("="*80)
        print("Install Sionna to use TDL channels: pip install sionna")
    
    # Test 3: Factory function
    print("\n" + "="*80)
    print("Test 3: Factory Function")
    print("="*80)
    
    gen1 = create_channel_generator('rayleigh', num_taps=5)
    print(f"Created: {gen1}")
    
    if SIONNA_AVAILABLE:
        try:
            gen2 = create_channel_generator('TDL-B', delay_spread=100e-9)
            print(f"Created: {gen2}")
        except Exception as e:
            print(f"TDL-B creation failed: {e}")
    
    print("\n" + "="*80)
    print("✓ Tests completed!")
    print("="*80)
