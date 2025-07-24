"""
⚠️  已弃用：此文件为旧版数据生成器，保留用于兼容性
✅ 新版本：请使用 data_generator_refactored.py 中的模块化设计
📝 新版本特点：
   - Uniformly use SNR and timing_offset configuration from user_config.py
   - 模块化架构，支持动态信道切换
   - 更好的PyTorch/TensorFlow边界管理
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal
import random
import math

from user_config import SRSConfig
from utils import generate_base_sequence, apply_cyclic_shift
import matplotlib.pyplot as plt

# Try to import SIONNA - the only professional channel library we support
try:
    from professional_channels import SIONNAChannelModel, SIONNA_AVAILABLE
    if SIONNA_AVAILABLE:
        print("✅ SIONNA professional channel library available for data generation")
    else:
        print("⚠️  SIONNA not available, will use simplified fallback channel")
except ImportError:
    SIONNA_AVAILABLE = False
    print("⚠️  Professional channels module not available, using simplified fallback")

class SimpleFallbackChannelModel:
    """
    Simplified fallback channel model - used only when SIONNA is not available
    
    This provides minimal functionality for development/testing when SIONNA cannot be installed.
    For production use, always install SIONNA for proper 3GPP-compliant channel modeling.
    
    WARNING: This is NOT a complete 3GPP implementation!
    """
    
    def __init__(
        self, 
        model_type: Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"] = "TDL-A",
        num_rx_antennas: int = 1,
        delay_spread: float = 100e-9,
        sampling_rate: float = 122.88e6,
        k_factor: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Simple fallback channel - basic implementation for testing only
        
        WARNING: This is a simplified model for fallback use only.
        For proper 3GPP channel modeling, install SIONNA:
        python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow
        """
        print("⚠️  WARNING: Using simplified fallback channel model")
        print("   This is NOT a complete 3GPP implementation!")
        print("   Install SIONNA for proper channel modeling:")
        print("   python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow")
        
        self.model_type = model_type
        self.num_rx_antennas = num_rx_antennas
        self.delay_spread = delay_spread
        self.sampling_rate = sampling_rate
        self.device = device
        
        # Simplified single-tap model (NOT 3GPP compliant)
        self.tap_indices = torch.tensor([0], device=device)
        self.powers_linear = torch.tensor([1.0], device=device)
    
    def generate_channel_taps(self, num_tx_antennas: int = 1, num_rx_antennas: Optional[int] = None, num_taps_max: Optional[int] = None) -> torch.Tensor:
        """Generate simplified channel taps (single tap only)"""
        if num_rx_antennas is None:
            num_rx_antennas = self.num_rx_antennas
        
        # Single tap at delay 0 (simplified)
        h = torch.zeros((num_rx_antennas, num_tx_antennas, 1), dtype=torch.complex64, device=self.device)
        
        for rx_ant in range(num_rx_antennas):
            for tx_ant in range(num_tx_antennas):
                # Simple complex Gaussian tap
                tap_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(), device=self.device)
                tap_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(), device=self.device)
                h[rx_ant, tx_ant, 0] = torch.complex(tap_real, tap_imag)
        
        return h
    
    def apply_channel(self, signals, delay_offset_samples: int = 0, mapping_indices: Optional[torch.Tensor] = None, ifft_size: Optional[int] = None, debug_dict: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply simplified channel (single tap only)"""
        # Convert input to tensor format  
        if isinstance(signals, dict):
            # 保持原始输入顺序，不使用sorted()
            original_keys = list(signals.keys())
            signals_list = [signals[key] for key in original_keys]
            signals = torch.stack(signals_list)
        elif isinstance(signals, list):
            signals = torch.stack(signals)
        elif signals.dim() == 1:
            signals = signals.unsqueeze(0)
        
        num_tx_antennas = signals.shape[0]
        signal_length = signals.shape[1]
        
        # Generate simple channel taps
        h_taps = self.generate_channel_taps(num_tx_antennas=num_tx_antennas, num_rx_antennas=self.num_rx_antennas)
        
        # Apply channel (single tap, no convolution)
        y_signals = torch.zeros((self.num_rx_antennas, signal_length), dtype=torch.complex64, device=self.device)
        
        for rx_ant in range(self.num_rx_antennas):
            for tx_ant in range(num_tx_antennas):
                h = h_taps[rx_ant, tx_ant, 0]  # Single tap
                y_signals[rx_ant] += h * signals[tx_ant]  # No convolution, just multiplication
        
        # Apply timing offset by circular shifting
        if delay_offset_samples != 0:
            y_signals = torch.roll(y_signals, shifts=delay_offset_samples, dims=1)
        
        # Simple frequency domain channel (flat fading)
        h_freq_multi = None
        if mapping_indices is not None:
            h_freq_multi = torch.zeros((self.num_rx_antennas, num_tx_antennas, len(mapping_indices)), 
                                     dtype=torch.complex64, device=self.device)
            for rx_ant in range(self.num_rx_antennas):
                for tx_ant in range(num_tx_antennas):
                    h = h_taps[rx_ant, tx_ant, 0]
                    h_freq_multi[rx_ant, tx_ant] = h.expand(len(mapping_indices))
        
        return y_signals, h_freq_multi


class SRSDataGenerator:
    """
    Generate synthetic data for SRS channel estimation training and testing
    """
    def __init__(
        self,
        config: SRSConfig,
        num_rx_antennas: int = 4,       # Number of receive antennas at BS
        snr_range: Tuple[float, float] = (-10, 40),  # DEPRECATED: 应使用config.snr_range
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
        
        # Initialize channel model: prefer SIONNA, fallback to simple model
        if SIONNA_AVAILABLE:
            print(f"🚀 Using SIONNA professional {channel_model} channel model")
            self.tdl_model = SIONNAChannelModel(
                model_type=channel_model,
                num_rx_antennas=num_rx_antennas,
                num_tx_antennas=1,  # Will be set dynamically per user
                delay_spread=delay_spread,
                carrier_frequency=3.5e9,  # Default 5G frequency
                sampling_rate=sampling_rate,
                device=device
            )
        else:
            print(f"⚠️  SIONNA not available, using simplified fallback channel")
            print("   Install SIONNA for proper 3GPP channel modeling:")
            print("   python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow")
            self.tdl_model = SimpleFallbackChannelModel(
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
            enable_debug: If True, stores all intermediate signals and processing steps
        
        Returns:
            Dictionary containing:
            - ls_estimate: LS channel estimate for each receive antenna
            - true_channels: List of true channel responses for each user's ports
            - noise_power: True noise power
            - snr: SNR in dB for each UE
            - mapping_indices: Indices for subcarrier mapping
            - debug_info: (if enable_debug=True) All intermediate signals and processing steps
        """
        L = self.config.seq_length
        K = self.config.K
        num_users = self.config.num_users
        
        # Initialize received signal for each receive antenna at the basestation
        y_freq_rx_antennas = {
            rx_ant: torch.zeros(L, dtype=torch.complex64, device=self.device)
            for rx_ant in range(self.num_rx_antennas)
        }
        
        # Store true channels for each user/port/rx_antenna
        true_channels = {}  # [user][port][rx_ant] -> channel
        
        # Store SNR per user using dictionary structure
        user_snrs = {}  # [user] -> SNR in dB
        
        # User-specific signals storage
        user_signals = {}  # [user][rx_ant] -> signal after channel
        
        # Generate signals for all users and apply channel
        for u in range(num_users):
            # Select SNR for this UE
            if fixed_snr is not None:
                ue_snr_db = fixed_snr
            else:
                ue_snr_db = random.uniform(*self.snr_range)
            
            # Store the UE's SNR using dictionary structure
            user_snrs[u] = ue_snr_db
            
            
            # Account for multiple ports: the power is divided among ports
            num_ports = self.config.ports_per_user[u]
            port_power_factor = 1.0 / math.sqrt(num_ports)  # sqrt(N) reduction in amplitude
            
            # Initialize signals for this user using dictionary structure
            user_signals[u] = {
                rx_ant: torch.zeros(L, dtype=torch.complex64, device=self.device)
                for rx_ant in range(self.num_rx_antennas)
            }
            
            # Generate one timing offset per user (all ports of the same UE use the same timing offset)
            delay_offset_seconds = random.uniform(*self.delay_offset_range)
            delay_offset_samples = int(delay_offset_seconds * self.sampling_rate)
            
            
            # Initialize true channels dictionary for this user with correct structure: [user][rx_ant][tx_ant]
            true_channels[u] = {}
            for rx_ant in range(self.num_rx_antennas):
                true_channels[u][rx_ant] = {}
            
            # Generate separate signals for each port (transmit antenna)
            tx_signals = {}  # [port] -> signal
            for p in range(num_ports):
                # Get cyclic shift
                n_u_p = self.config.cyclic_shifts[u][p]
                
                # Generate cyclically shifted sequence (frequency domain)
                x_u_p_freq = apply_cyclic_shift(self.base_sequence, n_u_p, K).to(self.device)
                
                # Create full OFDM symbol in frequency domain
                x_port_mapped = torch.zeros(self.ifft_size, dtype=torch.complex64, device=self.device)
                x_port_mapped[self.mapping_indices] = x_u_p_freq
                
                # Convert to time domain with IFFT
                x_time = torch.fft.ifft(x_port_mapped) * math.sqrt(self.ifft_size)
                
                # Add cyclic prefix
                x_with_cp = torch.cat([
                    x_time[-self.cp_length:],  # CP from end of symbol
                    x_time,                    # Full symbol
                    x_time[:self.cp_length]    # CP for the end (to handle negative timing offsets)
                ])
                
                tx_signals[p] = x_with_cp
                
            
            # Apply TDL channel: multiple transmit antennas to multiple receive antennas
            y_time_with_cp_shifted, h_freq_channels = self.tdl_model.apply_channel(
                tx_signals,  # Dictionary of signals from each port (transmit antenna)
                delay_offset_samples=delay_offset_samples,
                mapping_indices=self.mapping_indices,
                ifft_size=self.ifft_size,
            )  # Shape: [num_rx_antennas, signal_length], [num_rx_antennas, num_tx_antennas, L]
            
            
            # Process each receive antenna separately
            for rx_ant in range(self.num_rx_antennas):
                # Remove cyclic prefix - extract only the main symbol part
                y_time = y_time_with_cp_shifted[rx_ant, self.cp_length:self.cp_length+self.ifft_size]
                
                # Convert back to frequency domain
                y_freq = torch.fft.fft(y_time) / math.sqrt(self.ifft_size)
                
                # Extract only the mapped positions to get back to original sequence length
                y_user_combined = y_freq[self.mapping_indices]
                
                # Store user's combined signal after channel
                user_signals[u][rx_ant] = y_user_combined
                
                
                # Store channel information for each transmit antenna (port)
                for tx_ant in range(num_ports):
                    # Store the channel from tx_ant to rx_ant
                    true_channels[u][rx_ant][tx_ant] = h_freq_channels[rx_ant, tx_ant]
        
        # Fixed noise power (nominal value)
        nominal_noise_power = 1e-3
        
        
        # Calculate user-specific signal powers and scale factors
        user_powers = {}
        user_scale_factors = {}
        actual_snrs = {}
        
        for u in range(num_users):
            # Calculate average power across all receive antennas for this user
            user_power = 0
            for rx_ant in range(self.num_rx_antennas):
                user_power += torch.mean(torch.abs(user_signals[u][rx_ant]) ** 2).item()
            user_power /= self.num_rx_antennas
            user_powers[u] = user_power
            
            # Calculate scale factor to achieve desired SNR
            snr_db = user_snrs[u]
            snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear
            
            if user_power > 0:
                scale_factor = np.sqrt(snr_linear * nominal_noise_power / user_power)
            else:
                scale_factor = 0
            
            user_scale_factors[u] = scale_factor
            
            # Calculate actual achieved SNR
            actual_snr = 10 * math.log10(user_power * (scale_factor ** 2) / nominal_noise_power) if user_power > 0 else -float('inf')
            actual_snrs[u] = actual_snr
            
        
        # Scale each user's signals to achieve desired SNR and combine at receiver
        for u in range(num_users):
            scale_factor = user_scale_factors[u]
            # Scale and add to combined received signal
            for rx_ant in range(self.num_rx_antennas):
                scaled_signal = user_signals[u][rx_ant] * scale_factor
                y_freq_rx_antennas[rx_ant] += scaled_signal
                
        
        # Add noise to each receive antenna
        y_noisy = {}
        for rx_ant in range(self.num_rx_antennas):
            # Generate noise with fixed power for this antenna
            noise = torch.complex(
                torch.randn_like(torch.real(y_freq_rx_antennas[rx_ant])) * np.sqrt(nominal_noise_power / 2),
                torch.randn_like(torch.imag(y_freq_rx_antennas[rx_ant])) * np.sqrt(nominal_noise_power / 2)
            )
            
            # Add noise to the combined signal
            y_noisy[rx_ant] = y_freq_rx_antennas[rx_ant] + noise
            
        
        # Apply user scale factors to true channels for accurate comparison with estimates
        scaled_true_channels = {}
        for u in range(num_users):
            scaled_true_channels[u] = {}
            scale_factor = user_scale_factors[u]
            for rx_ant in range(self.num_rx_antennas):
                scaled_true_channels[u][rx_ant] = {}
                for tx_ant in range(self.config.ports_per_user[u]):
                    scaled_true_channels[u][rx_ant][tx_ant] = true_channels[u][rx_ant][tx_ant] * scale_factor
        
        # Calculate LS estimate for each antenna (receiver processing)
        ls_estimates = {}
        for rx_ant in range(self.num_rx_antennas):
            ls_estimates[rx_ant] = y_noisy[rx_ant] / self.base_sequence
        
        
        # Prepare return dictionary
        result = {
            'ls_estimates': ls_estimates,  # [rx_ant] -> signal
            'true_channels': scaled_true_channels,  # [user][rx_ant][tx_ant] -> channel
            'noise_power': nominal_noise_power,
            'snr': actual_snrs,  # [user] -> SNR in dB
            'mapping_indices': self.mapping_indices  # Include mapping indices for reference
        }
        
        
        return result
        
    def generate_batch(self, batch_size: int, fixed_snr: Optional[float] = None, enable_debug: bool = False) -> Dict:
        """
        Generate a batch of training/testing samples
        
        Args:
            batch_size: Number of samples to generate
            fixed_snr: If provided, uses this fixed SNR value instead of random from range
            enable_debug: If True, stores debug information for the first sample in the batch
            
        Returns:
            Dictionary containing batched data
        """
        # Generate individual samples
        samples = []
        for i in range(batch_size):
            # Only enable debug for the first sample to avoid excessive memory usage
            sample_enable_debug = enable_debug and (i == 0)
            sample = self.generate_sample(fixed_snr=fixed_snr, enable_debug=sample_enable_debug)
            samples.append(sample)
        
        # Batch ls_estimates - convert dictionary structure to tensor
        ls_estimates_list = []
        for sample in samples:
            # Convert dictionary to tensor for this sample
            sample_estimates = torch.stack([
                sample['ls_estimates'][rx_ant] 
                for rx_ant in range(self.num_rx_antennas)
            ])
            ls_estimates_list.append(sample_estimates)
        
        ls_estimates = torch.stack(ls_estimates_list)  # Shape: [batch_size, num_rx_antennas, L]
        
        # Batch noise_powers
        noise_powers = torch.tensor([sample['noise_power'] for sample in samples], device=self.device)
        
        # Batch true_channels (now using dictionary structure: [u][rx_ant][tx_ant])
        true_channels_batched = {}
        for u in range(self.config.num_users):
            true_channels_batched[u] = {}
            for rx_ant in range(self.num_rx_antennas):
                true_channels_batched[u][rx_ant] = {}
                for tx_ant in range(self.config.ports_per_user[u]):
                    channels = []
                    for sample in samples:
                        if (u in sample['true_channels'] and 
                            rx_ant in sample['true_channels'][u] and 
                            tx_ant in sample['true_channels'][u][rx_ant]):
                            channels.append(sample['true_channels'][u][rx_ant][tx_ant])
                    if channels:
                        true_channels_batched[u][rx_ant][tx_ant] = torch.stack(channels)
        
        # Batch SNRs (convert dictionary structure to list for compatibility)
        snrs = []
        for sample in samples:
            sample_snrs = [sample['snr'][u] for u in range(self.config.num_users)]
            snrs.append(sample_snrs)
        
        # Include mapping indices (same for all samples)
        mapping_indices = samples[0]['mapping_indices']
        
        # Prepare result dictionary
        result = {
            'ls_estimates': ls_estimates,
            'true_channels': true_channels_batched,
            'noise_powers': noise_powers,
            'snrs': snrs,
            'mapping_indices': mapping_indices
        }
        
        # Add debug information from first sample if available
        if enable_debug and 'debug_info' in samples[0]:
            result['debug_info'] = samples[0]['debug_info']
        
        return result
    
    def print_debug_summary(self, debug_info: Dict):
        """
        Print a summary of debug information for easy inspection
        
        Args:
            debug_info: Debug dictionary returned from generate_sample(enable_debug=True)
        """
        print("=== SRS Data Generation Debug Summary ===")
        print(f"Configuration:")
        print(f"  - Sequence length: {debug_info['seq_length']}")
        print(f"  - Number of users: {debug_info['num_users']}")
        print(f"  - Number of RX antennas: {debug_info['num_rx_antennas']}")
        print(f"  - IFFT size: {debug_info['ifft_size']}")
        print(f"  - CP length: {debug_info['cp_length']}")
        print(f"  - Sampling rate: {debug_info['sampling_rate']:.2e} Hz")
        print(f"  - Noise power: {debug_info['nominal_noise_power']:.2e}")
        
        print(f"\nBase sequence energy: {debug_info['base_sequence'].abs().pow(2).sum().item():.6f}")
        
        print(f"\nPer-user information:")
        for u in range(debug_info['num_users']):
            print(f"  User {u}:")
            print(f"    - Target SNR: {debug_info['user_snrs'][u]:.2f} dB")
            print(f"    - Actual SNR: {debug_info['actual_snrs'][u]:.2f} dB")
            print(f"    - Signal power: {debug_info['user_powers'][u]:.6e}")
            print(f"    - Scale factor: {debug_info['user_scale_factors'][u]:.6f}")
            print(f"    - Delay offset: {debug_info['user_delay_offsets'][u]['delay_offset_samples']} samples "
                  f"({debug_info['user_delay_offsets'][u]['delay_offset_seconds']*1e9:.2f} ns)")
            
            # Show channel tap energy
            if u in debug_info['channel_debug']:
                h_taps_energy = debug_info['channel_debug'][u]['h_taps_energy'].item()
                print(f"    - Channel taps energy: {h_taps_energy:.6f}")
        
        print(f"\nSignal energies:")
        print(f"  - Combined signal (before noise): {debug_info['combined_signal_energy_before_noise']:.6f}")
        print(f"  - Total noise energy: {debug_info['total_noise_energy']:.6f}")
        
        print(f"\nReceive antenna energies:")
        for rx_ant in range(debug_info['num_rx_antennas']):
            ls_energy = debug_info['ls_estimates'][rx_ant].abs().pow(2).sum().item()
            print(f"  - RX antenna {rx_ant} LS estimate energy: {ls_energy:.6f}")
    
    def get_debug_tensor(self, debug_info: Dict, key_path: str):
        """
        Extract a specific tensor from debug_info using a dot-separated path
        
        Args:
            debug_info: Debug dictionary
            key_path: Dot-separated path, e.g., "user_signals_after_channel.0.1.y_user_combined"
            
        Returns:
            The requested tensor or None if not found
        """
        keys = key_path.split('.')
        current = debug_info
        
        try:
            for key in keys:
                if key.isdigit():
                    current = current[int(key)]
                else:
                    current = current[key]
            return current
        except (KeyError, IndexError, TypeError):
            print(f"Key path '{key_path}' not found in debug_info")
            return None
    
    def list_debug_keys(self, debug_info: Dict, prefix: str = "", max_depth: int = 3):
        """
        List all available keys in debug_info for easy navigation
        
        Args:
            debug_info: Debug dictionary
            prefix: Current prefix for recursive listing
            max_depth: Maximum depth to recurse
        """
        if max_depth <= 0:
            return
            
        for key, value in debug_info.items():
            current_path = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                print(f"{current_path}/ (dict)")
                if max_depth > 1:
                    self.list_debug_keys(value, current_path, max_depth - 1)
            elif isinstance(value, torch.Tensor):
                print(f"{current_path} (tensor: {list(value.shape)}, {value.dtype})")
            else:
                print(f"{current_path} ({type(value).__name__})")
