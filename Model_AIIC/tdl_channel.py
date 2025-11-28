"""
Custom TDL (Tapped Delay Line) Channel Model
Based on 3GPP TR 38.901

Pure NumPy implementation with independent fading per sample.
No Doppler correlation - each time sample has independent random phase.
"""

import numpy as np
import torch
from typing import Optional, Union


class TDLChannel:
    """
    TDL Channel Model from 3GPP TR 38.901
    
    Generates time-domain channel impulse responses with independent fading.
    Each sample has independent random phases (no time correlation).
    
    Args:
        model: TDL model type ('A', 'B', 'C', 'D', 'E')
        delay_spread: RMS delay spread in seconds
        carrier_frequency: Carrier frequency in Hz
        normalize: Normalize channel to unit power
    """
    
    # TDL-A model parameters from 3GPP TR 38.901 Table 7.7.2-1
    TDL_A_DELAYS_NS = np.array([
        0.0, 3.0, 15.0, 31.0, 70.0, 91.0, 113.0, 127.0, 149.0, 175.0,
        189.0, 217.0, 234.0, 273.0, 297.0, 356.0, 382.0, 423.0, 492.0,
        538.0, 593.0, 640.0
    ])  # Normalized delays in nanoseconds
    
    TDL_A_POWERS_DB = np.array([
        -13.4, 0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9,
        -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3,
        -18.9, -16.6, -19.9
    ])
    
    # TDL-B model parameters
    TDL_B_DELAYS_NS = np.array([
        0.0, 10.0, 190.0, 410.0, 730.0, 1050.0, 1370.0, 1690.0, 2010.0, 3000.0
    ])
    
    TDL_B_POWERS_DB = np.array([
        0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0
    ])
    
    # TDL-C model parameters
    TDL_C_DELAYS_NS = np.array([
        0, 65, 130, 195, 260, 325, 390, 455, 520, 585, 650, 715, 780,
        845, 910, 975, 1040, 1105, 1170, 1235, 1300, 1365, 1430, 1495
    ])
    
    TDL_C_POWERS_DB = np.array([
        -4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1,
        -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
        -17.1, -16.0, -15.7, -21.6, -22.8
    ])
    
    # TDL-D model parameters
    TDL_D_DELAYS_NS = np.array([
        0, 35, 612, 1426, 1794, 2636, 3590, 4726, 5665, 6956,
        8885, 10610, 12050, 13610, 15200, 16360, 18030, 20390
    ])
    
    TDL_D_POWERS_DB = np.array([
        -0.2, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -21.9, -22.9,
        -27.8, -23.6, -24.8, -30.0, -27.7, -28.2, -29.0, -29.9, -30.0
    ])
    
    # TDL-E model parameters
    TDL_E_DELAYS_NS = np.array([
        0, 5, 15, 20, 40, 60, 90, 115, 145, 175, 210, 245, 280, 315,
        355, 400, 450, 510, 575, 655, 730, 810, 895, 985
    ])
    
    TDL_E_POWERS_DB = np.array([
        -0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8,
        -22.6, -22.3, -25.6, -20.2, -29.8, -29.2, -28.9, -28.0, -29.0,
        -28.9, -28.0, -27.8, -30.0, -30.0, -30.0
    ])
    
    MODELS = {
        'A': (TDL_A_DELAYS_NS, TDL_A_POWERS_DB),
        'B': (TDL_B_DELAYS_NS, TDL_B_POWERS_DB),
        'C': (TDL_C_DELAYS_NS, TDL_C_POWERS_DB),
        'D': (TDL_D_DELAYS_NS, TDL_D_POWERS_DB),
        'E': (TDL_E_DELAYS_NS, TDL_E_POWERS_DB),
    }
    
    def __init__(
        self,
        model: str = 'A',
        delay_spread: float = 30e-9,
        carrier_frequency: float = 3.5e9,
        normalize: bool = True
    ):
        """
        Initialize TDL channel
        
        Args:
            model: 'A', 'B', 'C', 'D', or 'E'
            delay_spread: RMS delay spread in seconds (e.g., 30e-9 for 30ns)
            carrier_frequency: Carrier frequency in Hz (e.g., 3.5e9 for 3.5 GHz)
            normalize: Normalize channel to unit power
        """
        if model not in self.MODELS:
            raise ValueError(f"Model must be one of {list(self.MODELS.keys())}, got {model}")
        
        self.model = model
        self.delay_spread = delay_spread
        self.carrier_frequency = carrier_frequency
        self.normalize = normalize
        
        # Get model parameters
        delays_norm_ns, powers_db = self.MODELS[model]
        
        # Scale delays by delay spread (normalized delays are in nanoseconds)
        # Actual delays = normalized_delays * (delay_spread / reference_delay)
        reference_delay_ns = delays_norm_ns[-1]  # Last tap delay
        self.delays = delays_norm_ns * 1e-9 * (delay_spread * 1e9) / reference_delay_ns
        
        # Convert power from dB to linear
        self.powers = 10 ** (powers_db / 10)
        
        # Normalize power to sum to 1
        if self.normalize:
            self.powers = self.powers / self.powers.sum()
        
        self.num_paths = len(self.delays)
    
    def generate(
        self,
        batch_size: int,
        num_ports: int,
        seq_len: int,
        sampling_rate: float,
        return_torch: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Generate independent TDL channel realizations
        
        Each sample (batch, port, time) has independent random phases.
        No time correlation - pure independent fading.
        
        Args:
            batch_size: Number of independent realizations
            num_ports: Number of antenna ports
            seq_len: Number of time samples
            sampling_rate: Sampling rate in Hz
            return_torch: If True, return torch.Tensor, else np.ndarray
        
        Returns:
            h: Channel impulse response
               Shape: (batch_size, num_ports, seq_len)
               Type: complex64
        """
        dt = 1.0 / sampling_rate
        
        # Initialize output
        h = np.zeros((batch_size, num_ports, seq_len), dtype=np.complex64)
        
        # Time indices for each tap
        delay_indices = np.round(self.delays / dt).astype(int)
        
        # Filter valid delays (within seq_len)
        valid_mask = delay_indices < seq_len
        valid_delays = delay_indices[valid_mask]
        valid_powers = self.powers[valid_mask]
        
        # For each batch and port, generate independent channel
        for b in range(batch_size):
            for p in range(num_ports):
                # Random phases for each path (independent per batch/port)
                phases = np.random.uniform(0, 2*np.pi, len(valid_delays))
                
                # Complex gains: sqrt(power) * exp(j*phase)
                gains = np.sqrt(valid_powers) * np.exp(1j * phases)
                
                # Place gains at delay taps
                for delay_idx, gain in zip(valid_delays, gains):
                    h[b, p, delay_idx] = gain
        
        # Convert to torch if requested
        if return_torch:
            h = torch.from_numpy(h)
        
        return h
    
    def generate_batch_parallel(
        self,
        batch_size: int,
        num_ports: int,
        seq_len: int,
        sampling_rate: float,
        return_torch: bool = True
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Parallel vectorized generation (faster for large batches)
        
        Uses NumPy broadcasting for efficient computation.
        """
        dt = 1.0 / sampling_rate
        
        # Initialize output
        h = np.zeros((batch_size, num_ports, seq_len), dtype=np.complex64)
        
        # Delay indices
        delay_indices = np.round(self.delays / dt).astype(int)
        valid_mask = delay_indices < seq_len
        valid_delays = delay_indices[valid_mask]
        valid_powers = self.powers[valid_mask]
        
        # Generate all random phases at once
        # Shape: (batch_size, num_ports, num_valid_paths)
        phases = np.random.uniform(0, 2*np.pi, (batch_size, num_ports, len(valid_delays)))
        
        # Complex gains: (batch_size, num_ports, num_paths)
        gains = np.sqrt(valid_powers[None, None, :]) * np.exp(1j * phases)
        
        # Place gains at delay taps (vectorized)
        for path_idx, delay_idx in enumerate(valid_delays):
            h[:, :, delay_idx] = gains[:, :, path_idx]
        
        # Convert to torch if requested
        if return_torch:
            h = torch.from_numpy(h)
        
        return h


def test_tdl_channel():
    """Test TDL channel generation"""
    print("Testing TDL Channel Models")
    print("=" * 80)
    
    # Test parameters
    batch_size = 2048
    num_ports = 4
    seq_len = 12
    scs = 30e3
    Ktc = 4
    sampling_rate = scs * Ktc * seq_len
    
    for model in ['A', 'B', 'C', 'D', 'E']:
        print(f"\nTDL-{model}:")
        
        # Create channel
        tdl = TDLChannel(
            model=model,
            delay_spread=30e-9,
            carrier_frequency=3.5e9,
            normalize=True
        )
        
        print(f"  Number of paths: {tdl.num_paths}")
        print(f"  Path delays (ns): {tdl.delays * 1e9}")
        print(f"  Path powers (linear): {tdl.powers}")
        print(f"  Total power: {tdl.powers.sum():.4f}")
        
        # Generate channels
        import time
        start = time.time()
        h = tdl.generate_batch_parallel(
            batch_size=batch_size,
            num_ports=num_ports,
            seq_len=seq_len,
            sampling_rate=sampling_rate,
            return_torch=True
        )
        elapsed = time.time() - start
        
        print(f"  Generation time: {elapsed*1000:.2f} ms")
        print(f"  Throughput: {batch_size*num_ports/elapsed:.0f} channels/s")
        print(f"  Output shape: {h.shape}")
        print(f"  Mean power: {h.abs().pow(2).mean():.4f}")
        
        # Verify independence (check first tap across different samples)
        h_np = h.numpy()
        if batch_size >= 2:
            # Correlation between different batch samples (should be ~0)
            corr_matrix = np.corrcoef(h_np[:min(10, batch_size), 0, 0].real)
            if corr_matrix.size > 1:
                corr_batch = corr_matrix[0, 1]
                print(f"  Batch correlation: {corr_batch:.4f} (should be ~0)")
        if seq_len > 1:
            # Correlation across time (should be ~0 for independent fading)
            corr_matrix = np.corrcoef(h_np[0, 0, :].real)
            if corr_matrix.size > 1:
                corr_time = corr_matrix[0, 1]
                print(f"  Time correlation: {corr_time:.4f} (should be ~0)")


if __name__ == "__main__":
    test_tdl_channel()
