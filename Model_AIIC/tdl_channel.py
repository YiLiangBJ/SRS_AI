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
    
    Supports TDL-A, TDL-B, and TDL-C models with accurate normalized delays.
    
    Args:
        model: TDL model type ('A', 'B', or 'C')
        delay_spread: RMS delay spread in seconds (e.g., 30e-9 for 30ns)
                     Actual delays = normalized_delay × delay_spread
                     Max delay = max_normalized_delay × delay_spread
                     (e.g., TDL-C: max_delay = 8.6523 × 30ns = 259.569ns)
        carrier_frequency: Carrier frequency in Hz (e.g., 3.5e9 for 3.5 GHz)
        normalize: Normalize channel to unit power
    """
    
    # TDL-A model parameters from 3GPP TR 38.901 Table 7.7.2-1
    # Normalized delays (when RMS delay spread = 1ns)
    TDL_A_NORMALIZED_DELAYS = np.array([
        0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708, 0.5750,
        0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942, 2.5119, 3.0582,
        4.0810, 4.4579, 4.5695, 4.7966, 5.0066, 5.3043, 9.6586
    ])
    
    TDL_A_POWERS_DB = np.array([
        -13.4, 0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5, -15.9,
        -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7, -16.2, -18.3,
        -18.9, -16.6, -19.9, -29.7
    ])
    
    # TDL-B model parameters from 3GPP TR 38.901 Table 7.7.2-2
    # Normalized delays (when RMS delay spread = 1ns)
    TDL_B_NORMALIZED_DELAYS = np.array([
        0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055,
        0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756, 1.5474, 1.7842,
        2.0169, 2.8294, 3.0219, 3.6187, 4.1067, 4.2790, 4.7834
    ])
    
    TDL_B_POWERS_DB = np.array([
        0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0,
        -8.9, -9.0, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8, -11.4,
        -14.9, -9.2, -11.3
    ])
    
    # TDL-C model parameters from 3GPP TR 38.901 Table 7.7.2-3
    # Normalized delays (when RMS delay spread = 1ns)
    TDL_C_NORMALIZED_DELAYS = np.array([
        0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560,
        0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105,
        4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523
    ])
    
    TDL_C_POWERS_DB = np.array([
        -4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1,
        -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
        -17.1, -16.0, -15.7, -21.6, -22.8
    ])
    
    MODELS = {
        'A': (TDL_A_NORMALIZED_DELAYS, TDL_A_POWERS_DB),
        'B': (TDL_B_NORMALIZED_DELAYS, TDL_B_POWERS_DB),
        'C': (TDL_C_NORMALIZED_DELAYS, TDL_C_POWERS_DB),
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
            model: 'A', 'B', or 'C' (from 3GPP TR 38.901 Table 7.7.2-{1,2,3})
            delay_spread: RMS delay spread in seconds (e.g., 30e-9 for 30ns)
                         This is the RMS value, not the maximum delay
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
        normalized_delays, powers_db = self.MODELS[model]
        
        # Calculate actual delays
        # normalized_delays are for RMS delay spread = 1ns
        # Actual delays = normalized_delays × delay_spread
        # Example: if delay_spread = 30ns, max delay for TDL-C = 8.6523 × 30ns = 259.569ns
        self.delays = normalized_delays * delay_spread
        
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
    
    for model in ['A', 'B', 'C']:
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
