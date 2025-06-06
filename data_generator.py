import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import random

from config import SRSConfig
from utils import generate_base_sequence, apply_cyclic_shift, apply_channel, add_noise, generate_channel_taps


class SRSDataGenerator:
    """
    Generate synthetic data for SRS channel estimation training and testing
    """
    def __init__(
        self,
        config: SRSConfig,
        snr_range: Tuple[float, float] = (0, 30),
        num_taps_range: Tuple[int, int] = (5, 6),
        delay_offset_range: Tuple[int, int] = (0, 1),
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the SRS data generator
        
        Args:
            config: SRS configuration
            snr_range: Range of SNR values in dB
            num_taps_range: Range of number of channel taps (represents delay spread)
            delay_offset_range: Range of timing offsets
            device: Computation device
        """
        self.config = config
        self.snr_range = snr_range
        self.num_taps_range = num_taps_range
        self.delay_offset_range = delay_offset_range
        self.device = device
        
        # Validate config
        self.config.validate_config()
        
        # Generate base sequence
        self.base_sequence = generate_base_sequence(self.config.seq_length).to(device)
    
    def generate_sample(self) -> Dict:
        """
        Generate a single training/testing sample
        
        Returns:
            Dictionary containing:
            - ls_estimate: LS channel estimate
            - true_channels: List of true channel responses for each user's ports
            - noise_power: True noise power
            - snr: SNR in dB
        """
        L = self.config.seq_length
        K = self.config.K
        num_users = self.config.num_users
        
        # Select random SNR
        snr_db = random.uniform(*self.snr_range)
        
        # Initialize received signal
        y = torch.zeros(L, dtype=torch.complex64, device=self.device)
        
        # Store true channels for each user/port
        true_channels = []
        
        # For each user and port
        for u in range(num_users):
            for p in range(self.config.ports_per_user[u]):
                # Get cyclic shift
                n_u_p = self.config.cyclic_shifts[u][p]
                
                # Generate cyclically shifted sequence
                x_u_p = apply_cyclic_shift(self.base_sequence, n_u_p, K).to(self.device)
                
                # Generate random channel
                num_taps = random.randint(*self.num_taps_range)
                delay_offset = random.randint(*self.delay_offset_range)
                  # Generate channel taps
                h_taps = generate_channel_taps(
                    num_taps=num_taps,
                    power_delay_profile='exponential',
                    delay_offset=delay_offset
                ).to(self.device)
                
                # Apply channel to sequence
                y_u_p = apply_channel(x_u_p, h_taps)
                
                # Add to received signal
                y += y_u_p
                
                # Calculate true frequency domain channel
                h_true = torch.fft.fft(h_taps, n=L)
                true_channels.append((u, p, h_true))
        
        # Add noise
        y_noisy = add_noise(y, snr_db)
        
        # Calculate noise power
        signal_power = torch.mean(torch.abs(y) ** 2).item()
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Calculate LS estimate
        ls_estimate = y_noisy / self.base_sequence
        
        # Return generated sample
        return {
            'ls_estimate': ls_estimate,
            'true_channels': true_channels,
            'noise_power': noise_power,
            'snr': snr_db
        }
    
    def generate_batch(self, batch_size: int) -> Dict:
        """
        Generate a batch of training/testing samples
        
        Args:
            batch_size: Number of samples to generate
            
        Returns:
            Dictionary containing batched data
        """
        # Generate individual samples
        samples = [self.generate_sample() for _ in range(batch_size)]
        
        # Batch ls_estimates
        ls_estimates = torch.stack([sample['ls_estimate'] for sample in samples])
        
        # Batch noise_powers
        noise_powers = torch.tensor([sample['noise_power'] for sample in samples], device=self.device)
        
        # Batch true_channels (this is more complex as different samples may have different orderings)
        # We'll create a fixed ordering based on user and port
        true_channels_batched = {}
        for u in range(self.config.num_users):
            for p in range(max(self.config.ports_per_user)):
                if p < self.config.ports_per_user[u]:
                    channels = []
                    for sample in samples:
                        for user, port, channel in sample['true_channels']:
                            if user == u and port == p:
                                channels.append(channel)
                                break
                    if channels:
                        true_channels_batched[(u, p)] = torch.stack(channels)
        
        # Batch SNRs
        snrs = torch.tensor([sample['snr'] for sample in samples], device=self.device)
        
        return {
            'ls_estimates': ls_estimates,
            'true_channels': true_channels_batched,
            'noise_powers': noise_powers,
            'snrs': snrs
        }
