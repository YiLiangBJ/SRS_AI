import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal
import random
import math

from config import SRSConfig
from utils import generate_base_sequence, apply_cyclic_shift


class TDLChannelModel:
    """
    Implementation of 3GPP TDL (Tapped Delay Line) channel models
    
    Based on 3GPP TR 38.901 V16.1.0 (2019-12), Section 7.7.2:
    - TDL-A: Rural macrocell, with a DS (Delay Spread) of 100 ns to 1000 ns
    - TDL-B: Urban macrocell, with a DS of 100 ns to 1000 ns
    - TDL-C: Urban microcell, with a DS of 100 ns to 1000 ns
    - TDL-D: Urban macrocell LoS (Line of Sight) with a DS of 10 ns to 100 ns
    - TDL-E: Urban microcell LoS with a DS of 10 ns to 100 ns
    """
    
    def __init__(
        self, 
        model_type: Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"] = "TDL-A",
        num_rx_antennas: int = 1,      # Number of receive antennas
        delay_spread: float = 100e-9,  # in seconds
        sampling_rate: float = 122.88e6, # in Hz (samples per second)
        k_factor: float = 0.0,         # Linear scale (0 for TDL-A, B, C)
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize TDL channel model according to 3GPP specifications
        
        Args:
            model_type: Type of TDL channel model (TDL-A, TDL-B, TDL-C, TDL-D, TDL-E)
            num_rx_antennas: Number of receive antennas
            delay_spread: RMS delay spread in seconds (default 100 ns)
            sampling_rate: Sampling rate in Hz (default 122.88 MHz)
            k_factor: Ricean K-factor in linear scale (0 for NLOS models, >0 for LOS models)
            device: Computation device
        """
        self.model_type = model_type
        self.num_rx_antennas = num_rx_antennas
        self.delay_spread = delay_spread
        self.sampling_rate = sampling_rate
        self.k_factor = k_factor
        self.device = device
        
        # Define model parameters based on 3GPP TR 38.901
        if model_type == "TDL-A":
            # Delay in ns (normalized to 1), relative power in dB
            self.delays = torch.tensor([0.0000, 0.3819, 0.4025, 0.5868, 0.6016, 0.6172, 0.6309, 0.6478, 0.6907, 0.7415, 0.7585, 0.7754, 0.7831, 0.8071, 0.9141, 1.0000])
            self.powers = torch.tensor([-13.4, 0, -2.2, -4, -6, -8.2, -9.9, -10.5, -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -16.2])
        elif model_type == "TDL-B":
            self.delays = torch.tensor([0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752, 0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.0000])
            self.powers = torch.tensor([-6.9, -0.9, -1.7, -2.5, -3.9, -3.3, -6.7, -9.9, -3.4, -5.7, -7.3, -10.4, -18.4])
        elif model_type == "TDL-C":
            self.delays = torch.tensor([0, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448, 0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083, 2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065, 6.6374, 7.0427, 8.6523])
            self.powers = torch.tensor([-4.4, -1.2, -3.5, -5.2, -2.5, 0, -2.2, -3.9, -7.4, -7.1, -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8, -17.1, -16, -15.7, -21.6, -22.8])
        elif model_type == "TDL-D":
            self.delays = torch.tensor([0.000, 0.035, 0.612, 1.363, 1.405, 1.804, 2.596, 1.775, 4.042, 7.937, 9.424, 9.708, 12.525])
            self.powers = torch.tensor([-0.2, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -21.9, -22.9, -27.8, -23.6, -24.8, -30.0])
            if k_factor <= 0:
                self.k_factor = 7.0  # Default K-factor for TDL-D in linear scale
        elif model_type == "TDL-E":
            self.delays = torch.tensor([0.000, 0.089, 0.419, 0.573, 0.680, 1.180, 2.055, 2.510, 3.390, 4.266, 4.865, 5.267, 5.374])
            self.powers = torch.tensor([-0.03, -22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.8, -20.8, -22.6, -22.3, -25.6, -20.2])
            if k_factor <= 0:
                self.k_factor = 4.0  # Default K-factor for TDL-E in linear scale
        else:
            raise ValueError(f"Unknown TDL model type: {model_type}")
        
        # Convert powers from dB to linear
        self.powers_linear = 10 ** (self.powers / 10)
        
        # Normalize powers if needed
        # self.powers_linear = self.powers_linear / torch.sum(self.powers_linear)
          # Scale delays by delay spread to get actual time delays
        self.delays_scaled = self.delays * delay_spread
        
        # Convert time delays to sample indices based on sampling rate
        initial_tap_indices = torch.round(self.delays_scaled * sampling_rate).to(torch.long)
        
        # 处理映射到相同采样点索引的taps，合并它们的功率
        unique_indices, inverse_indices = torch.unique(initial_tap_indices, return_inverse=True)
        
        # 创建新的功率数组，用于合并相同索引的功率
        merged_powers = torch.zeros(len(unique_indices), device=self.powers_linear.device)
        
        # 合并映射到相同采样点的功率
        for i, idx in enumerate(inverse_indices):
            merged_powers[idx] += self.powers_linear[i]
            
        # 更新tap索引和对应的功率
        self.tap_indices = unique_indices
        self.powers_linear = merged_powers
        
        # 确保合并后的功率仍然是归一化的
        self.powers_linear = self.powers_linear / torch.sum(self.powers_linear)
        
        # 更新dB形式的功率（仅用于参考）
        self.powers = 10 * torch.log10(self.powers_linear + 1e-20)  # 加入小值防止log(0)
        
        # Move to device
        self.delays = self.delays.to(device)
        self.powers = self.powers.to(device)
        self.powers_linear = self.powers_linear.to(device)
        self.delays_scaled = self.delays_scaled.to(device)
        self.tap_indices = self.tap_indices.to(device)
    
    def generate_channel_taps(
        self, 
        num_rx_antennas: Optional[int] = None,
        num_taps_max: Optional[int] = None,
        timing_offset: int = 0
    ) -> torch.Tensor:
        """
        Generate channel impulse response taps according to TDL model
        
        Args:
            num_rx_antennas: Number of receive antennas for the channel taps
            num_taps_max: Maximum number of taps to generate (truncates longer channels)
            timing_offset: Timing offset in samples (+ for delayed signal, - for early signal)
            
        Returns:
            Complex tensor representing the channel impulse response
            Shape: [num_rx_antennas, num_taps] if num_rx_antennas > 1 else [num_taps]
        """
        # Use default number of antennas if not specified
        if num_rx_antennas is None:
            num_rx_antennas = self.num_rx_antennas
        
        # Determine maximum delay index
        max_index = torch.max(self.tap_indices).item()
        
        if num_taps_max is not None:
            max_taps = num_taps_max
        else:
            max_taps = max_index + 1
        
        # Create channel impulse response vector for multiple receive antennas
        if num_rx_antennas > 1:
            h = torch.zeros((num_rx_antennas, max_taps), dtype=torch.complex64, device=self.device)
            
            # Generate independent channel realizations for each receive antenna
            for rx_ant in range(num_rx_antennas):
                # Generate complex Gaussian taps (Rayleigh fading)
                for i, (idx, power) in enumerate(zip(self.tap_indices, self.powers_linear)):
                    # Apply timing offset
                    effective_idx = idx + timing_offset
                    
                    # Skip if out of bounds
                    if effective_idx < 0 or effective_idx >= max_taps:
                        continue
                    
                    # Generate complex Gaussian tap with appropriate power
                    tap_real = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(1,), device=self.device)
                    tap_imag = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(1,), device=self.device)
                    
                    # For LOS components (first tap of TDL-D and TDL-E)
                    if (self.model_type in ["TDL-D", "TDL-E"]) and i == 0 and self.k_factor > 0:
                        # Add LOS component (Rician fading)
                        los_amplitude = np.sqrt(power * self.k_factor / (self.k_factor + 1))
                        los_phase = torch.rand(1, device=self.device) * 2 * np.pi
                        los_real = los_amplitude * torch.cos(los_phase)
                        los_imag = los_amplitude * torch.sin(los_phase)
                        
                        # Scale NLOS component
                        nlos_scale = np.sqrt(1 / (self.k_factor + 1))
                        tap_real *= nlos_scale
                        tap_imag *= nlos_scale
                        
                        # Combine LOS and NLOS
                        tap_real += los_real
                        tap_imag += los_imag
                    
                    # Add to channel impulse response for this antenna
                    h[rx_ant, effective_idx] += torch.complex(tap_real, tap_imag)
        else:
            # For single antenna, keep original implementation
            h = torch.zeros(max_taps, dtype=torch.complex64, device=self.device)
            
            # Generate complex Gaussian taps (Rayleigh fading)
            for i, (idx, power) in enumerate(zip(self.tap_indices, self.powers_linear)):
                # Apply timing offset
                effective_idx = idx + timing_offset
                
                # Skip if out of bounds
                if effective_idx < 0 or effective_idx >= max_taps:
                    continue
                
                # Generate complex Gaussian tap with appropriate power
                tap_real = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(1,), device=self.device)
                tap_imag = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(1,), device=self.device)
                
                # For LOS components (first tap of TDL-D and TDL-E)
                if (self.model_type in ["TDL-D", "TDL-E"]) and i == 0 and self.k_factor > 0:
                    # Add LOS component (Rician fading)
                    los_amplitude = np.sqrt(power * self.k_factor / (self.k_factor + 1))
                    los_phase = torch.rand(1, device=self.device) * 2 * np.pi
                    los_real = los_amplitude * torch.cos(los_phase)
                    los_imag = los_amplitude * torch.sin(los_phase)
                    
                    # Scale NLOS component
                    nlos_scale = np.sqrt(1 / (self.k_factor + 1))
                    tap_real *= nlos_scale
                    tap_imag *= nlos_scale
                    
                    # Combine LOS and NLOS
                    tap_real += los_real
                    tap_imag += los_imag
                
                # Add to channel impulse response
                h[effective_idx] += torch.complex(tap_real, tap_imag)
        
        return h
    
    def apply_channel(
        self, 
        signal: torch.Tensor,
        timing_offset: int = 0,
        rx_antenna_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply the TDL channel model to a time domain signal
        
        Args:
            signal: Time domain signal
            timing_offset: Timing offset in samples
            rx_antenna_idx: If provided, applies only the channel for the specified receive antenna
            
        Returns:
            Time domain signal after passing through the channel.
            If rx_antenna_idx is None and num_rx_antennas > 1, returns a tensor of shape [num_rx_antennas, signal_length]
        """
        # Generate channel impulse response
        if rx_antenna_idx is not None:
            # Generate for specific antenna
            h = self.generate_channel_taps(num_rx_antennas=1, timing_offset=timing_offset)
            
            # Perform circular convolution
            output = torch.nn.functional.conv1d(
                signal.reshape(1, 1, -1),  # Reshape to [batch, channels, length]
                h.reshape(1, 1, -1),       # Reshape to [out_channels, in_channels, kernel_size]
                padding='same'             # 'same' for circular convolution
            )
            
            return output.reshape(signal.shape)
        elif self.num_rx_antennas > 1:
            # Generate for all receive antennas
            h_multi = self.generate_channel_taps(timing_offset=timing_offset)
            
            # Initialize output tensor for all antennas
            output = torch.zeros((self.num_rx_antennas,) + signal.shape, dtype=torch.complex64, device=self.device)
            
            # Apply channel for each antenna
            for rx_ant in range(self.num_rx_antennas):
                h = h_multi[rx_ant]
                
                # Perform circular convolution
                ant_output = torch.nn.functional.conv1d(
                    signal.reshape(1, 1, -1),  # Reshape to [batch, channels, length]
                    h.reshape(1, 1, -1),       # Reshape to [out_channels, in_channels, kernel_size]
                    padding='same'             # 'same' for circular convolution
                )
                
                output[rx_ant] = ant_output.reshape(signal.shape)
            
            return output
        else:
            # Single antenna case (original implementation)
            h = self.generate_channel_taps(timing_offset=timing_offset)
            
            # Perform circular convolution
            output = torch.nn.functional.conv1d(
                signal.reshape(1, 1, -1),  # Reshape to [batch, channels, length]
                h.reshape(1, 1, -1),       # Reshape to [out_channels, in_channels, kernel_size]
                padding='same'             # 'same' for circular convolution
            )
            
            return output.reshape(signal.shape)


class SRSDataGenerator:
    """
    Generate synthetic data for SRS channel estimation training and testing
    """
    def __init__(
        self,
        config: SRSConfig,
        num_rx_antennas: int = 4,       # Number of receive antennas at BS
        snr_range: Tuple[float, float] = (-10, 40),
        channel_model: Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"] = "TDL-A",
        delay_spread: float = 100e-9,  # in seconds
        sampling_rate: float = 122.88e6, # in Hz (default 122.88 MHz for 5G)
        delta_f: float = 30e3,  # subcarrier spacing in Hz (default 30 kHz)
        ifft_size: int = 4096,  # IFFT size (default 4096)
        cp_length: int = 288,   # Cyclic prefix length in samples
        ktc: int = 4,          # Parameter to determine subcarrier distance
        start_pos: int = 0,    # Starting position for SRS mapping
        delay_offset_range: Tuple[float, float] = (-130e-9, 130e-9),  # in seconds (-130ns to 130ns)
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None
    ):
        """
        Initialize the SRS data generator
        
        Args:
            config: SRS configuration
            num_rx_antennas: Number of receive antennas at the base station
            snr_range: Range of SNR values in dB
            channel_model: Channel model type (TDL-A, TDL-B, TDL-C, TDL-D, TDL-E from 3GPP)
            delay_spread: RMS delay spread in seconds (for TDL models)
            sampling_rate: Sampling rate in Hz
            delta_f: Subcarrier spacing in Hz
            ifft_size: Size of IFFT for OFDM modulation
            cp_length: Length of cyclic prefix in samples
            ktc: Parameter to determine subcarrier distance (distance = 12/ktc)
            start_pos: Starting position for SRS sequence mapping
            delay_offset_range: Range of timing offsets in seconds (-130ns to 130ns)
            device: Computation device
            seed: Random seed for reproducibility
        """
        self.config = config
        self.num_rx_antennas = num_rx_antennas
        self.snr_range = snr_range
        self.channel_model = channel_model
        self.delay_spread = delay_spread
        self.sampling_rate = sampling_rate        
        self.delta_f = delta_f
        self.ifft_size = ifft_size
        self.cp_length = cp_length
        self.ktc = ktc
        self.start_pos = start_pos
        self.delay_offset_range = delay_offset_range
        self.device = device
        self.initial_seed = seed
        
        # Calculate subcarrier distance based on ktc parameter
        self.distance = 12 // self.ktc
        
        # Set seed if provided
        if seed is not None:
            self.reset_seed(seed)
        
        # Validate config
        self.config.validate_config()
        
        # Generate base sequence
        self.base_sequence = generate_base_sequence(self.config.seq_length).to(device)
        
        # Create mapping indices for SRS sequence to subcarriers
        self.mapping_indices = torch.arange(
            self.start_pos, 
            self.start_pos + self.config.seq_length * self.distance, 
            self.distance
        ) % self.ifft_size
        self.mapping_indices = self.mapping_indices.to(device)
        
        # Initialize TDL channel model
        self.tdl_model = TDLChannelModel(
            model_type=channel_model,
            num_rx_antennas=num_rx_antennas,
            delay_spread=delay_spread,
            sampling_rate=sampling_rate,
            device=device
        )

    def reset_seed(self, seed: Optional[int] = None):
        """
        Reset random seeds to ensure reproducible data generation
        
        Args:
            seed: Random seed to use. If None, uses the initial seed from __init__
        """
        if seed is None:
            seed = self.initial_seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    def generate_sample(self, fixed_snr: Optional[float] = None) -> Dict:
        """
        Generate a single training/testing sample
        
        Args:
            fixed_snr: If provided, uses this fixed SNR value instead of random from range.
                       This is a UE-specific parameter.
        
        Returns:
            Dictionary containing:
            - ls_estimate: LS channel estimate for each receive antenna
            - true_channels: List of true channel responses for each user's ports
            - noise_power: True noise power
            - snr: SNR in dB for each UE
            - mapping_indices: Indices for subcarrier mapping
        """
        L = self.config.seq_length
        K = self.config.K
        num_users = self.config.num_users
        
        # Initialize received signal for each receive antenna at the basestation
        y_freq_rx_antennas = torch.zeros(
            (self.num_rx_antennas, L),
            dtype=torch.complex64, 
            device=self.device
        )
        
        # Store true channels for each user/port/rx_antenna
        true_channels = []
        
        # Store SNR per user
        user_snrs = []
        
        # User-specific signals (to calculate per-user powers)
        user_signals = {}
        
        # First, generate signals for all users and ports
        for u in range(num_users):
            # Select SNR for this UE
            if fixed_snr is not None:
                ue_snr_db = fixed_snr
            else:
                ue_snr_db = random.uniform(*self.snr_range)
            
            # Store the UE's SNR
            user_snrs.append(ue_snr_db)
            
            # Account for multiple ports: the power is divided among ports
            num_ports = self.config.ports_per_user[u]
            port_power_factor = 1.0 / math.sqrt(num_ports)  # sqrt(N) reduction in amplitude
            
            # Initialize combined signal for this user (for power calculation)
            user_signals[u] = [
                torch.zeros(L, dtype=torch.complex64, device=self.device)
                for _ in range(self.num_rx_antennas)
            ]
            
            # Generate one timing offset per user (all ports of the same UE use the same timing offset)
            delay_offset_seconds = random.uniform(*self.delay_offset_range)
            delay_offset_samples = int(delay_offset_seconds * self.sampling_rate)
            
            # For each port of this user
            for p in range(num_ports):
                # Get cyclic shift
                n_u_p = self.config.cyclic_shifts[u][p]
                
                # Generate cyclically shifted sequence (frequency domain)
                x_u_p_freq = apply_cyclic_shift(self.base_sequence, n_u_p, K).to(self.device)
                
                # Apply port power scaling (divide power equally among ports)
                x_u_p_freq = x_u_p_freq * port_power_factor
                
                # Map sequence to IFFT inputs (comb-like structure with distance)
                x_mapped = torch.zeros(self.ifft_size, dtype=torch.complex64, device=self.device)
                x_mapped[self.mapping_indices] = x_u_p_freq
                
                # Convert to time domain with IFFT
                x_time = torch.fft.ifft(x_mapped) * math.sqrt(self.ifft_size)
                
                # Add cyclic prefix - take last cp_length samples and prepend
                x_with_cp = torch.cat([
                    x_time[-self.cp_length:],  # CP from end of symbol
                    x_time,                    # Full symbol
                    x_time[:self.cp_length]    # CP for the end (to handle negative timing offsets)
                ])
                
                # Apply TDL channel in time domain with UE-specific timing offset across all receive antennas
                y_time_with_cp = self.tdl_model.apply_channel(
                    x_with_cp,
                    timing_offset=delay_offset_samples
                )  # Shape: [num_rx_antennas, signal_length]
                
                # Generate channel impulse response for reference
                h_taps = self.tdl_model.generate_channel_taps(
                    num_taps_max=self.ifft_size//4,
                    timing_offset=delay_offset_samples
                )  # Shape: [num_rx_antennas, num_taps]
                
                # Now process each receive antenna separately
                for rx_ant in range(self.num_rx_antennas):
                    # Remove cyclic prefix - extract only the main symbol part
                    y_time = y_time_with_cp[rx_ant, self.cp_length:self.cp_length+self.ifft_size]
                    
                    # Convert back to frequency domain
                    y_freq = torch.fft.fft(y_time) / math.sqrt(self.ifft_size)
                    
                    # Extract only the mapped positions to get back to original sequence length
                    y_u_p = y_freq[self.mapping_indices]
                    
                    # Add to user-specific combined signal for this receive antenna (for power calculation)
                    user_signals[u][rx_ant] += y_u_p
                    
                    # Calculate true frequency domain channel at SRS positions
                    h_freq = torch.fft.fft(h_taps[rx_ant], n=self.ifft_size)
                    h_true = h_freq[self.mapping_indices]
                    
                    # Store the original channel for now (scaling will be applied later)
                    true_channels.append((u, p, rx_ant, h_true * port_power_factor))
        
        # Now combine all user signals at each receive antenna
        for u in range(num_users):
            for rx_ant in range(self.num_rx_antennas):
                # Add this user's signal to the combined receive signal for this antenna
                y_freq_rx_antennas[rx_ant] += user_signals[u][rx_ant]
        
        # Fixed noise power (nominal value)
        nominal_noise_power = 1e-3
        
        # Calculate user-specific signal powers for each antenna
        user_powers = {}
        for u in range(num_users):
            # Average power across all receive antennas
            user_power = 0
            for rx_ant in range(self.num_rx_antennas):
                user_power += torch.mean(torch.abs(user_signals[u][rx_ant]) ** 2).item()
            user_power /= self.num_rx_antennas
            user_powers[u] = user_power
        
        # Calculate scaling factor for each user to achieve desired SNR
        user_scale_factors = {}
        
        # Track actual SNR values
        actual_snrs = []
        
        for u, snr_db in enumerate(user_snrs):
            snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear
            
            # Calculate scale factor for this user's signal
            if user_powers[u] > 0:
                scale_factor = np.sqrt(snr_linear * nominal_noise_power / user_powers[u])
            else:
                scale_factor = 0
            
            user_scale_factors[u] = scale_factor
            
            # Update actual SNR for this user
            actual_snr = 10 * math.log10(user_powers[u] * (scale_factor ** 2) / nominal_noise_power) if user_powers[u] > 0 else -float('inf')
            actual_snrs.append(actual_snr)
        
        # Scale all users' signals and combine
        y_freq_scaled = torch.zeros_like(y_freq_rx_antennas)
        for u in range(num_users):
            scale_factor = user_scale_factors[u]
            # Apply scaling to all antennas
            for rx_ant in range(self.num_rx_antennas):
                y_freq_scaled[rx_ant] += user_signals[u][rx_ant] * scale_factor
        
        # Generate noise and add to each receive antenna
        y_noisy = torch.zeros_like(y_freq_scaled)
        for rx_ant in range(self.num_rx_antennas):
            # Generate noise with fixed power for this antenna
            noise = torch.complex(
                torch.randn_like(torch.real(y_freq_scaled[rx_ant])) * np.sqrt(nominal_noise_power / 2),
                torch.randn_like(torch.imag(y_freq_scaled[rx_ant])) * np.sqrt(nominal_noise_power / 2)
            )
            
            # Add noise to the combined signal
            y_noisy[rx_ant] = y_freq_scaled[rx_ant] + noise
        
        # Apply user scale factors to true channels for accurate comparison with estimates
        scaled_true_channels = []
        for u, p, rx_ant, h_true in true_channels:
            scale_factor = user_scale_factors[u]
            scaled_true_channels.append((u, p, rx_ant, h_true * scale_factor))
        
        # Calculate LS estimate for each antenna
        ls_estimates = torch.zeros_like(y_noisy)
        for rx_ant in range(self.num_rx_antennas):
            ls_estimates[rx_ant] = y_noisy[rx_ant] / self.base_sequence
        
        # Return generated sample
        return {
            'ls_estimates': ls_estimates,  # [num_rx_antennas, L]
            'true_channels': scaled_true_channels,  # List of (u, p, rx_ant, h)
            'noise_power': nominal_noise_power,
            'snr': actual_snrs,  # Returns actual SNRs achieved for each user
            'mapping_indices': self.mapping_indices  # Include mapping indices for reference
        }
        
    def generate_batch(self, batch_size: int, fixed_snr: Optional[float] = None) -> Dict:
        """
        Generate a batch of training/testing samples
        
        Args:
            batch_size: Number of samples to generate
            fixed_snr: If provided, uses this fixed SNR value instead of random from range
            
        Returns:
            Dictionary containing batched data
        """
        # Generate individual samples
        samples = [self.generate_sample(fixed_snr=fixed_snr) for _ in range(batch_size)]
        
        # Batch ls_estimates - now has shape [batch_size, num_rx_antennas, L]
        ls_estimates = torch.stack([sample['ls_estimates'] for sample in samples])
        
        # Batch noise_powers
        noise_powers = torch.tensor([sample['noise_power'] for sample in samples], device=self.device)
        
        # Batch true_channels (this is more complex as different samples may have different orderings)
        # We'll create a fixed ordering based on user, port, and rx_antenna
        true_channels_batched = {}
        for u in range(self.config.num_users):
            for p in range(max(self.config.ports_per_user)):
                if p < self.config.ports_per_user[u]:
                    for rx_ant in range(self.num_rx_antennas):
                        channels = []
                        for sample in samples:
                            for user, port, ant, channel in sample['true_channels']:
                                if user == u and port == p and ant == rx_ant:
                                    channels.append(channel)
                                    break
                        if channels:
                            true_channels_batched[(u, p, rx_ant)] = torch.stack(channels)
        
        # Batch SNRs (now a list of lists, one list per sample, each inner list has one SNR per user)
        snrs = [sample['snr'] for sample in samples]
        
        # Include mapping indices (same for all samples)
        mapping_indices = samples[0]['mapping_indices']
        
        return {
            'ls_estimates': ls_estimates,
            'true_channels': true_channels_batched,
            'noise_powers': noise_powers,
            'snrs': snrs,
            'mapping_indices': mapping_indices
        }
