import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Tuple, Dict, Optional, Union


def is_prime(n: int) -> bool:
    """
    Check if a number is prime
    
    Args:
        n: Number to check
        
    Returns:
        True if n is prime, False otherwise
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_largest_prime_less_than_or_equal_to(n: int) -> int:
    """
    Find the largest prime number less than or equal to n
    
    Args:
        n: Upper bound
        
    Returns:
        Largest prime number less than or equal to n
    """
    if n < 2:
        raise ValueError("No prime number exists below 2")
    
    # Start from n and move downward
    for i in range(n, 1, -1):
        if is_prime(i):
            return i
    
    return 2  # Fallback, but this shouldn't happen for n >= 2


def generate_base_sequence(length: int, root_index: int = 25) -> torch.Tensor:
    """
    Generate a SRS sequence according to 3GPP specifications
    
    Args:
        length: Length of the SRS sequence (L)
        root_index: Root index of the ZC sequence (hardcoded default = 25)
        
    Returns:
        Complex tensor representing the SRS sequence of length L
    """
    # Find the largest prime p less than or equal to length
    prime_length = find_largest_prime_less_than_or_equal_to(length)
    
    print(f"Generating ZC sequence with prime length {prime_length}, then extending to {length}")
    
    # Ensure root index is coprime with the prime length
    while math.gcd(root_index, prime_length) != 1:
        root_index += 1
        if root_index >= prime_length:
            root_index = 1
    
    # Generate ZC sequence of length p
    # ZC sequence generation formula: x(n) = exp(-j * π * u * n * (n+1) / prime_length)
    n = torch.arange(prime_length, dtype=torch.float32)
    exponent = -1j * math.pi * root_index * n * (n + 1) / prime_length
    zc_sequence = torch.exp(exponent)
    
    # Extend to length L through periodic extension if needed
    if length > prime_length:
        # Create extended sequence
        extended_sequence = torch.zeros(length, dtype=torch.complex64)
        for i in range(length):
            extended_sequence[i] = zc_sequence[i % prime_length]
        return extended_sequence
    else:
        # If length is already prime, return the ZC sequence directly
        return zc_sequence


def apply_cyclic_shift(base_seq: torch.Tensor, n: int, K: int) -> torch.Tensor:
    """
    Apply cyclic shift to base sequence to generate user/port specific sequence
    
    Args:
        base_seq: Base SRS sequence
        n: Cyclic shift parameter
        K: Total number of cyclic shifts
        
    Returns:
        Cyclically shifted sequence
    """
    L = len(base_seq)
    n_range = torch.arange(L)
    phasor = torch.exp(1j * 2 * np.pi * n * n_range / K)
    return base_seq * phasor


def apply_channel(seq: torch.Tensor, channel_taps: torch.Tensor) -> torch.Tensor:
    """
    Apply a channel to the sequence
    
    Args:
        seq: Input sequence in frequency domain
        channel_taps: Channel impulse response in time domain
        
    Returns:
        Sequence after passing through the channel (frequency domain)
        
    Note:
        In traditional channel estimation approaches:
        - Timing offsets and delay spread are typically pre-determined or estimated
          through conventional methods like correlation
        - MMSE matrices (C and R) are derived analytically based on channel statistics
        
        In AI-based channel estimation:
        - Neural networks can implicitly learn timing and delay characteristics
        - C and R matrices can be trainable parameters in a neural network, adapting to
          various channel conditions without explicit modeling
    """
    # Convert sequence to time domain
    seq_time = torch.fft.ifft(seq)
    
    # Apply channel (circular convolution)
    L = len(seq)
    channel_taps_padded = torch.zeros(L, dtype=torch.complex64)
    channel_taps_padded[:len(channel_taps)] = channel_taps
    
    # Circular convolution in time domain
    seq_after_channel_time = torch.fft.ifft(
        torch.fft.fft(seq_time) * torch.fft.fft(channel_taps_padded)
    )
    
    # Convert back to frequency domain
    return torch.fft.fft(seq_after_channel_time)


def add_noise(seq: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add complex Gaussian noise to a sequence
    
    Args:
        seq: Input sequence
        snr_db: Signal-to-noise ratio in dB
        
    Returns:
        Sequence with added noise
    """
    # Calculate signal power
    signal_power = torch.mean(torch.abs(seq) ** 2)
    
    # Calculate noise power
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    noise_real = torch.randn_like(torch.real(seq)) * np.sqrt(noise_power / 2)
    noise_imag = torch.randn_like(torch.imag(seq)) * np.sqrt(noise_power / 2)
    noise = torch.complex(noise_real, noise_imag)
    
    return seq + noise


def generate_channel_taps(
    num_taps: int,
    max_delay_samples: int,
    power_delay_profile: str = 'exponential',
    delay_offset: int = 0
) -> torch.Tensor:
    """
    Generate random channel impulse response taps
    
    Args:
        num_taps: Number of channel taps
        max_delay_samples: Maximum delay spread in samples
        power_delay_profile: Type of power delay profile
        delay_offset: Timing offset in samples
        
    Returns:
        Complex tensor representing the channel impulse response
    """
    if power_delay_profile == 'exponential':
        # Exponential power delay profile
        delays = torch.arange(num_taps)
        powers = torch.exp(-delays / (num_taps / 4))
        
    elif power_delay_profile == 'uniform':
        # Uniform power delay profile
        powers = torch.ones(num_taps)
        
    else:
        raise ValueError(f"Unknown power delay profile: {power_delay_profile}")
    
    # Normalize powers
    powers = powers / torch.sum(powers)
    
    # Generate complex Gaussian taps
    taps_real = torch.randn(num_taps) * torch.sqrt(powers / 2)
    taps_imag = torch.randn(num_taps) * torch.sqrt(powers / 2)
    taps = torch.complex(taps_real, taps_imag)
      # Apply delay offset
    channel_length = max_delay_samples + num_taps
    h = torch.zeros(channel_length, dtype=torch.complex64)
    
    # Ensure delay_offset is within valid range
    delay_offset = max(0, min(delay_offset, channel_length - num_taps))
    
    # Now assign taps with validated offset
    h[delay_offset:delay_offset + num_taps] = taps
    
    return h


def visualize_channel_estimate(
    true_channel: torch.Tensor,
    estimated_channel: torch.Tensor,
    title: str = "Channel Estimation Comparison"
) -> None:
    """
    Visualize the true and estimated channel
    
    Args:
        true_channel: True channel frequency response
        estimated_channel: Estimated channel frequency response
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    # Plot magnitude
    plt.subplot(2, 1, 1)
    plt.plot(torch.abs(true_channel).numpy(), 'b-', label='True Channel')
    plt.plot(torch.abs(estimated_channel).numpy(), 'r--', label='Estimated Channel')
    plt.title(title + " - Magnitude")
    plt.legend()
    plt.grid(True)
    
    # Plot phase
    plt.subplot(2, 1, 2)
    plt.plot(torch.angle(true_channel).numpy(), 'b-', label='True Channel')
    plt.plot(torch.angle(estimated_channel).numpy(), 'r--', label='Estimated Channel')
    plt.title(title + " - Phase")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def calculate_nmse(true_channel: torch.Tensor, estimated_channel: torch.Tensor) -> float:
    """
    Calculate the Normalized Mean Square Error between true and estimated channels
    
    Args:
        true_channel: True channel frequency response
        estimated_channel: Estimated channel frequency response
        
    Returns:
        NMSE value in dB
    """
    error = true_channel - estimated_channel
    nmse = torch.sum(torch.abs(error) ** 2) / torch.sum(torch.abs(true_channel) ** 2)
    return 10 * torch.log10(nmse).item()  # Convert to dB
