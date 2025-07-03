import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal
import random
import math

from config import SRSConfig
from utils import generate_base_sequence, apply_cyclic_shift
import matplotlib.pyplot as plt

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
        num_tx_antennas: int = 1,
        num_rx_antennas: Optional[int] = None,
        num_taps_max: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate channel impulse response taps according to TDL model (without timing offset)
        
        Args:
            num_tx_antennas: Number of transmit antennas (ports)
            num_rx_antennas: Number of receive antennas for the channel taps
            num_taps_max: Maximum number of taps to generate (truncates longer channels)
            
        Returns:
            Complex tensor representing the channel impulse response
            Shape: [num_rx_antennas, num_tx_antennas, num_taps]
        """
        # Use default number of antennas if not specified
        if num_rx_antennas is None:
            num_rx_antennas = self.num_rx_antennas
        
        # Determine maximum delay index (without timing offset)
        max_index = torch.max(self.tap_indices).item()
        
        if num_taps_max is not None:
            max_taps = num_taps_max
        else:
            max_taps = max_index + 1
        
        # Create channel impulse response tensor: [num_rx_antennas, num_tx_antennas, num_taps]
        h = torch.zeros((num_rx_antennas, num_tx_antennas, max_taps), dtype=torch.complex64, device=self.device)
        
        # Generate independent channel realizations for each rx-tx antenna pair
        for rx_ant in range(num_rx_antennas):
            for tx_ant in range(num_tx_antennas):
                # Generate complex Gaussian taps (Rayleigh fading) with random scaling
                for i, (idx, power) in enumerate(zip(self.tap_indices, self.powers_linear)):
                    
                    # Generate random scaling factor (mean=1, std=1 complex Gaussian)
                    # For complex Gaussian: real and imag parts are independent N(0, 0.5)
                    # This gives |z|^2 ~ Exponential(1) with mean=1
                    random_real = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(), device=self.device)
                    random_imag = torch.normal(mean=0.0, std=np.sqrt(0.5), size=(), device=self.device)
                    random_factor = torch.complex(random_real, random_imag)
                    
                    # Generate complex Gaussian tap with appropriate power and randomness
                    tap_real = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(), device=self.device)
                    tap_imag = torch.normal(mean=0.0, std=np.sqrt(power / 2), size=(), device=self.device)
                    base_tap = torch.complex(tap_real, tap_imag)
                    
                    # Apply random scaling
                    final_tap = base_tap * random_factor
                    
                    # For LOS components (first tap of TDL-D and TDL-E)
                    if (self.model_type in ["TDL-D", "TDL-E"]) and i == 0 and self.k_factor > 0:
                        # Add LOS component (Rician fading)
                        los_amplitude = np.sqrt(power * self.k_factor / (self.k_factor + 1))
                        los_phase = torch.rand((), device=self.device) * 2 * np.pi
                        los_real = los_amplitude * torch.cos(los_phase)
                        los_imag = los_amplitude * torch.sin(los_phase)
                        los_component = torch.complex(los_real, los_imag)
                        
                        # Scale NLOS component
                        nlos_scale = np.sqrt(1 / (self.k_factor + 1))
                        final_tap *= nlos_scale
                        
                        # Combine LOS and NLOS
                        final_tap += los_component
                    
                    # Add to channel impulse response for this rx-tx antenna pair at the correct delay index
                    h[rx_ant, tx_ant, idx] += final_tap
        
        return h
    
    def apply_channel(
        self, 
        signals: Union[torch.Tensor, List[torch.Tensor], Dict[int, torch.Tensor]],
        delay_offset_samples: int = 0,
        mapping_indices: Optional[torch.Tensor] = None,
        ifft_size: Optional[int] = None,
        debug_dict: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the TDL channel model to time domain signals from multiple transmit antennas
        
        Args:
            signals: Time domain signals from transmit antennas
                    - If tensor: shape [num_tx_antennas, signal_length] 
                    - If list: list of signals, one per transmit antenna
                    - If dict: dictionary of signals, keys are port indices
            delay_offset_samples: Timing offset in samples (+ for delay, - for advance)
            mapping_indices: Subcarrier mapping indices for frequency domain channel calculation
            ifft_size: IFFT size for frequency domain channel calculation
            debug_dict: Optional dictionary to store intermediate signals for debugging
            
        Returns:
            Tuple of (processed_signals, frequency_domain_channels):
            - processed_signals: [num_rx_antennas, signal_length] - combined received signals
            - frequency_domain_channels: [num_rx_antennas, num_tx_antennas, len(mapping_indices)] (if mapping_indices provided)
        """
        # Convert input to tensor format
        if isinstance(signals, dict):
            # Dictionary input: convert to list in order of keys
            signals_list = [signals[key] for key in sorted(signals.keys())]
            signals = torch.stack(signals_list)  # [num_tx_antennas, signal_length]
        elif isinstance(signals, list):
            signals = torch.stack(signals)  # [num_tx_antennas, signal_length]
        elif signals.dim() == 1:
            # Single signal case - add tx antenna dimension
            signals = signals.unsqueeze(0)  # [1, signal_length]
        
        num_tx_antennas = signals.shape[0]
        signal_length = signals.shape[1]
        
        # Store intermediate results in debug dictionary if provided
        if debug_dict is not None:
            debug_dict['input_signals'] = signals.clone()
            debug_dict['num_tx_antennas'] = num_tx_antennas
            debug_dict['signal_length'] = signal_length
            debug_dict['delay_offset_samples'] = delay_offset_samples
        
        # Generate channel impulse response for all tx-rx antenna pairs
        h_taps = self.generate_channel_taps(
            num_tx_antennas=num_tx_antennas,
            num_rx_antennas=self.num_rx_antennas
        )  # Shape: [num_rx_antennas, num_tx_antennas, num_taps]
        
        if debug_dict is not None:
            debug_dict['h_taps'] = h_taps.clone()
            debug_dict['h_taps_energy'] = h_taps.abs().pow(2).sum()
            debug_dict['h_taps_shape'] = h_taps.shape
        
        # Initialize output signals for each receive antenna
        y_signals = torch.zeros((self.num_rx_antennas, signal_length), dtype=torch.complex64, device=self.device)
        
        # Apply channel for each rx-tx antenna pair
        for rx_ant in range(self.num_rx_antennas):
            for tx_ant in range(num_tx_antennas):
                # Get channel taps for this rx-tx pair
                h = h_taps[rx_ant, tx_ant, :]
                
                # Apply channel convolution
                tx_signal = signals[tx_ant]
                
                # Apply circular convolution
                # For circular convolution, output length equals input length
                # y[n] = sum_{k=0}^{K-1} h[k] * x[(n-k) mod N]
                # where N is the signal length and K is the number of channel taps
                
                conv_output = torch.zeros_like(tx_signal, dtype=torch.complex64)
                
                # Manual circular convolution implementation
                for n in range(signal_length):
                    for k, h_k in enumerate(h):
                        if torch.abs(h_k) > 1e-10:  # Skip negligible taps for efficiency
                            # Circular indexing: (n - k) mod N
                            src_idx = (n - k) % signal_length
                            conv_output[n] += h_k * tx_signal[src_idx]
                
                # Apply timing offset by circular shifting
                if delay_offset_samples != 0:
                    conv_output = torch.roll(conv_output, shifts=delay_offset_samples, dims=0)
                
                # Accumulate signal from this transmit antenna to this receive antenna
                y_signals[rx_ant] += conv_output
        
        # Store processed signals in debug dictionary if provided
        if debug_dict is not None:
            debug_dict['y_signals'] = y_signals.clone()
            debug_dict['y_signals_energy'] = y_signals.abs().pow(2).sum()
        
        # Calculate frequency domain channels if requested
        h_freq_multi = None
        if mapping_indices is not None and ifft_size is not None:
            h_freq_multi = torch.zeros((self.num_rx_antennas, num_tx_antennas, len(mapping_indices)), 
                                     dtype=torch.complex64, device=self.device)
            
            for rx_ant in range(self.num_rx_antennas):
                for tx_ant in range(num_tx_antennas):
                    # Step 1: FFT to get ideal frequency domain channel
                    h = h_taps[rx_ant, tx_ant, :]
                    h_freq_full = torch.fft.fft(h, n=ifft_size) / math.sqrt(ifft_size)
                    h_ideal = h_freq_full[mapping_indices]
                    
                    # Step 2: Apply phase rotation due to timing offset
                    if delay_offset_samples != 0:
                        # Create frequency indices tensor for phase rotation
                        freq_indices = torch.arange(ifft_size, device=self.device, dtype=torch.float32)
                        phase_rotation_full = torch.exp(-1j * 2 * np.pi * freq_indices * delay_offset_samples / ifft_size)
                        phase_rotation = phase_rotation_full[mapping_indices]
                        h_freq_multi[rx_ant, tx_ant] = h_ideal * phase_rotation
                    else:
                        h_freq_multi[rx_ant, tx_ant] = h_ideal
            
            # Store frequency domain channels in debug dictionary if provided
            if debug_dict is not None:
                debug_dict['h_freq_multi'] = h_freq_multi.clone()
                debug_dict['h_freq_energy'] = h_freq_multi.abs().pow(2).sum()
                debug_dict['mapping_indices'] = mapping_indices.clone()
                debug_dict['ifft_size'] = ifft_size
        
        return y_signals, h_freq_multi


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

    def generate_sample(self, fixed_snr: Optional[float] = None, enable_debug: bool = False) -> Dict:
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
        
        # Initialize unified debug dictionary for all intermediate signals
        debug_info = {} if enable_debug else None
        
        if enable_debug:
            debug_info.update({
                # Basic configuration info
                'seq_length': L,
                'K': K,
                'num_users': num_users,
                'num_rx_antennas': self.num_rx_antennas,
                'base_sequence': self.base_sequence.clone(),
                'mapping_indices': self.mapping_indices.clone(),
                'ifft_size': self.ifft_size,
                'cp_length': self.cp_length,
                'sampling_rate': self.sampling_rate,
                'delay_offset_range': self.delay_offset_range,
                
                # Initialize storage for per-user data
                'user_signals_before_channel': {},  # [user][port] -> signal
                'user_signals_after_channel': {},   # [user][rx_ant] -> signal  
                'user_snrs': {},                    # [user] -> SNR
                'user_delay_offsets': {},           # [user] -> delay_offset_samples
                'user_scale_factors': {},           # [user] -> scale_factor
                'user_powers': {},                  # [user] -> signal_power
                'channel_debug': {},                # [user] -> TDL channel debug info
            })
        
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
            
            if enable_debug:
                debug_info['user_snrs'][u] = ue_snr_db
            
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
            
            if enable_debug:
                debug_info['user_delay_offsets'][u] = {
                    'delay_offset_seconds': delay_offset_seconds,
                    'delay_offset_samples': delay_offset_samples
                }
                debug_info['user_signals_before_channel'][u] = {}
                debug_info['user_signals_after_channel'][u] = {}
            
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
                
                if enable_debug:
                    debug_info['user_signals_before_channel'][u][p] = {
                        'cyclic_shift': n_u_p,
                        'x_u_p_freq': x_u_p_freq.clone(),
                        'x_port_mapped': x_port_mapped.clone(),
                        'x_time': x_time.clone(),
                        'x_with_cp': x_with_cp.clone(),
                        'signal_energy': x_with_cp.abs().pow(2).sum().item()
                    }
            
            # Apply TDL channel: multiple transmit antennas to multiple receive antennas
            channel_debug = {} if enable_debug else None
            y_time_with_cp_shifted, h_freq_channels = self.tdl_model.apply_channel(
                tx_signals,  # Dictionary of signals from each port (transmit antenna)
                delay_offset_samples=delay_offset_samples,
                mapping_indices=self.mapping_indices,
                ifft_size=self.ifft_size,
                debug_dict=channel_debug
            )  # Shape: [num_rx_antennas, signal_length], [num_rx_antennas, num_tx_antennas, L]
            
            if enable_debug:
                debug_info['channel_debug'][u] = channel_debug
            
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
                
                if enable_debug:
                    debug_info['user_signals_after_channel'][u][rx_ant] = {
                        'y_time_with_cp': y_time_with_cp_shifted[rx_ant].clone(),
                        'y_time': y_time.clone(),
                        'y_freq': y_freq.clone(),
                        'y_user_combined': y_user_combined.clone(),
                        'signal_energy': y_user_combined.abs().pow(2).sum().item()
                    }
                
                # Store channel information for each transmit antenna (port)
                for tx_ant in range(num_ports):
                    # Store the channel from tx_ant to rx_ant
                    true_channels[u][rx_ant][tx_ant] = h_freq_channels[rx_ant, tx_ant]
        
        # Fixed noise power (nominal value)
        nominal_noise_power = 1e-3
        
        if enable_debug:
            debug_info['nominal_noise_power'] = nominal_noise_power
            debug_info['y_freq_rx_antennas_before_scaling'] = {
                rx_ant: torch.zeros_like(y_freq_rx_antennas[rx_ant])
                for rx_ant in range(self.num_rx_antennas)
            }
        
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
            
            if enable_debug:
                debug_info['user_powers'][u] = user_power
                debug_info['user_scale_factors'][u] = scale_factor
                debug_info['actual_snrs'] = actual_snrs
        
        # Scale each user's signals to achieve desired SNR and combine at receiver
        for u in range(num_users):
            scale_factor = user_scale_factors[u]
            # Scale and add to combined received signal
            for rx_ant in range(self.num_rx_antennas):
                scaled_signal = user_signals[u][rx_ant] * scale_factor
                y_freq_rx_antennas[rx_ant] += scaled_signal
                
                if enable_debug:
                    debug_info['y_freq_rx_antennas_before_scaling'][rx_ant] += user_signals[u][rx_ant]
        
        if enable_debug:
            debug_info['y_freq_rx_antennas_after_scaling'] = {
                rx_ant: y_freq_rx_antennas[rx_ant].clone()
                for rx_ant in range(self.num_rx_antennas)
            }
            debug_info['combined_signal_energy_before_noise'] = sum([
                y_freq_rx_antennas[rx_ant].abs().pow(2).sum().item()
                for rx_ant in range(self.num_rx_antennas)
            ])
        
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
            
            if enable_debug:
                if 'noise_signals' not in debug_info:
                    debug_info['noise_signals'] = {}
                debug_info['noise_signals'][rx_ant] = noise.clone()
        
        if enable_debug:
            debug_info['y_noisy'] = {rx_ant: y_noisy[rx_ant].clone() for rx_ant in range(self.num_rx_antennas)}
            debug_info['total_noise_energy'] = sum([
                debug_info['noise_signals'][rx_ant].abs().pow(2).sum().item()
                for rx_ant in range(self.num_rx_antennas)
            ])
        
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
        
        if enable_debug:
            debug_info['ls_estimates'] = {rx_ant: ls_estimates[rx_ant].clone() for rx_ant in range(self.num_rx_antennas)}
            debug_info['scaled_true_channels'] = {}
            for u in range(num_users):
                debug_info['scaled_true_channels'][u] = {}
                for rx_ant in range(self.num_rx_antennas):
                    debug_info['scaled_true_channels'][u][rx_ant] = {}
                    for tx_ant in range(self.config.ports_per_user[u]):
                        debug_info['scaled_true_channels'][u][rx_ant][tx_ant] = scaled_true_channels[u][rx_ant][tx_ant].clone()
        
        # Prepare return dictionary
        result = {
            'ls_estimates': ls_estimates,  # [rx_ant] -> signal
            'true_channels': scaled_true_channels,  # [user][rx_ant][tx_ant] -> channel
            'noise_power': nominal_noise_power,
            'snr': actual_snrs,  # [user] -> SNR in dB
            'mapping_indices': self.mapping_indices  # Include mapping indices for reference
        }
        
        # Add debug information if enabled
        if enable_debug:
            result['debug_info'] = debug_info
        
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
