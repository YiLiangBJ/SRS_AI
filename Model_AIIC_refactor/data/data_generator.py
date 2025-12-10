"""
Data generation module for training channel separator models.

Generates synthetic SRS data with:
- TDL channel models (3GPP standard)
- Configurable SNR
- Random timing offsets
- Circular shifts for port separation
"""

import torch
import numpy as np
from typing import Tuple, List, Union, Optional


def generate_training_batch(
    batch_size: int = 32,
    seq_len: int = 12,
    pos_values: List[int] = None,
    snr_db: Union[float, List[float], Tuple[float, float]] = 20.0,
    tdl_config: Union[str, List[str]] = 'A-30',
    snr_sampler = None,
    snr_per_sample: bool = False,
    return_complex: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor, float]:
    """
    Generate a batch of training data with TDL channel and SNR control
    
    Args:
        batch_size: Number of samples in batch
        seq_len: Sequence length (default: 12)
        pos_values: Port positions (shifts). Default: [0, 3, 6, 9]
                   Examples: [0, 3, 6, 9] (4 ports), [0, 2, 4, 6, 8, 10] (6 ports)
        snr_db: SNR configuration
               - Scalar: same SNR for all ports
               - List: per-port SNR if len == num_ports, else random selection
               - Tuple (min, max): random SNR per batch or sample
        tdl_config: TDL configuration
                   - String: 'MODEL-DELAY_NS' e.g., 'A-30', 'B-100', 'C-300'
                   - List: random selection per sample
        snr_sampler: Optional SNRSampler for smart sampling (stratified/round-robin)
        snr_per_sample: If True, each sample gets different SNR (for SNR-invariant learning)
        return_complex: If True, return complex tensors; else real stacked [real; imag]
    
    Returns:
        y: Mixed signal
           - If return_complex=False: (B, L*2) real stacked [y_R; y_I]
           - If return_complex=True: (B, L) complex
        h_targets: Shifted channel targets
           - If return_complex=False: (B, P, L*2) real stacked
           - If return_complex=True: (B, P, L) complex
        pos_values: Port positions used
        h_true: Original channels (before shifting)
           - If return_complex=False: (B, P, L*2) real stacked
           - If return_complex=True: (B, P, L) complex
        actual_snr: Actual SNR used for this batch (for logging)
    
    Example:
        >>> y, h_targets, pos, h_true, snr = generate_training_batch(
        ...     batch_size=128,
        ...     snr_db=(0, 30),
        ...     tdl_config='A-30'
        ... )
        >>> print(y.shape, h_targets.shape)
        torch.Size([128, 24]) torch.Size([128, 4, 24])
    """
    # Default port positions
    if pos_values is None:
        pos_values = [0, 3, 6, 9]
    
    num_ports = len(pos_values)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generate TDL channel with random timing offsets
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Calculate timing parameters
    scs = 30e3  # Subcarrier spacing (Hz)
    Ktc = 4
    Tc = 1.0 / (480e3 * 4096)  # 3GPP basic time unit (~0.509 ns)
    Ts = 1.0 / (scs * Ktc * seq_len)  # Sampling interval
    
    # Random timing offsets: ±256*Tc for each port and sample
    timing_offset_Tc = np.random.uniform(-256, 256, (batch_size, num_ports))
    timing_offset_samples = timing_offset_Tc * Tc / Ts
    
    # Parse TDL configuration
    if isinstance(tdl_config, list):
        tdl_config = np.random.choice(tdl_config)
    
    parts = tdl_config.split('-')
    tdl_model = parts[0].upper()
    delay_ns = float(parts[1]) if len(parts) > 1 else 30
    delay_spread = delay_ns * 1e-9
    
    # Generate TDL channel
    try:
        from Model_AIIC.tdl_channel import TDLChannel
    except ImportError:
        # Fallback: try parent directory
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from Model_AIIC.tdl_channel import TDLChannel
    
    tdl = TDLChannel(
        model=tdl_model,
        delay_spread=delay_spread,
        carrier_frequency=3.5e9,
        normalize=True
    )
    
    sampling_rate = scs * Ktc * seq_len
    h_base = tdl.generate_batch_parallel(
        batch_size=batch_size,
        num_ports=num_ports,
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        return_torch=True
    )
    
    # Apply timing offset via frequency domain phase rotation
    H_fft = torch.fft.fft(h_base, dim=-1)
    k = torch.arange(seq_len, dtype=torch.float32)
    timing_offset_tensor = torch.from_numpy(timing_offset_samples).float()
    phase_shift = torch.exp(
        1j * 2 * np.pi * k[None, None, :] * timing_offset_tensor[:, :, None] / seq_len
    )
    H_shifted = H_fft * phase_shift.to(torch.complex64)
    h_base = torch.fft.ifft(H_shifted, dim=-1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Generate noise (unit power)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Adjust signal power based on SNR
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if isinstance(snr_db, tuple) and len(snr_db) == 2:
        snr_min, snr_max = snr_db
        
        if snr_per_sample:
            # Per-sample SNR: each sample gets different SNR
            sample_snrs = np.random.uniform(snr_min, snr_max, batch_size)
            signal_powers = torch.tensor([10 ** (snr / 10) for snr in sample_snrs])
            h_true = h_base * signal_powers.sqrt().view(batch_size, 1, 1)
            actual_snr = float(np.mean(sample_snrs))
        else:
            # Per-batch SNR: all samples get same SNR
            if snr_sampler is not None:
                batch_snr = snr_sampler.sample()
            else:
                batch_snr = np.random.uniform(snr_min, snr_max)
            
            signal_power = torch.tensor(10 ** (batch_snr / 10))
            h_true = h_base * signal_power.sqrt()
            actual_snr = batch_snr
    
    elif isinstance(snr_db, list):
        if len(snr_db) == num_ports:
            # Per-port SNR
            signal_powers = torch.tensor([10 ** (snr / 10) for snr in snr_db])
            h_true = h_base * signal_powers.sqrt().view(1, num_ports, 1)
            actual_snr = sum(snr_db) / len(snr_db)
        else:
            # Random selection from list
            batch_snr = np.random.choice(snr_db)
            signal_power = torch.tensor(10 ** (batch_snr / 10))
            h_true = h_base * signal_power.sqrt()
            actual_snr = batch_snr
    
    else:
        # Scalar SNR
        signal_power = torch.tensor(10 ** (snr_db / 10))
        h_true = h_base * signal_power.sqrt()
        actual_snr = float(snr_db)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Create mixed signal with circular shifts
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    y_clean = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    h_targets = []
    
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_true[:, i], shifts=-pos, dims=-1)
        y_clean += shifted
        h_targets.append(shifted)
    
    h_targets = torch.stack(h_targets, dim=1)
    y = y_clean + noise
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Format output (complex or real stacked)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if return_complex:
        return y, h_targets, pos_values, h_true, actual_snr
    else:
        # Convert to real stacked format [real; imag]
        y_stacked = torch.cat([y.real, y.imag], dim=-1)
        h_targets_stacked = torch.cat([h_targets.real, h_targets.imag], dim=-1)
        h_true_stacked = torch.cat([h_true.real, h_true.imag], dim=-1)
        return y_stacked, h_targets_stacked, pos_values, h_true_stacked, actual_snr
