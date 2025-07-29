import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal

from user_config import SRSConfig
from system_config import SystemConfig, create_default_system_config
from utils import generate_base_sequence, apply_cyclic_shift





class BaseSRSDataGenerator: 
    def __init__(
        self,
        srs_config: SRSConfig,
        system_config: SystemConfig,  # 系统配置
        # 以下参数如果为None，将从system_config获取
        num_rx_antennas: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        # 其他参数
        device: str = "cpu",  # Force CPU-only execution
        **kwargs
    ):
        self.srs_config = srs_config
        self.system_config = system_config  # 系统配置
        
        # 从系统配置获取参数
        self.num_rx_antennas = num_rx_antennas if num_rx_antennas is not None else system_config.num_rx_antennas
        self.sampling_rate = sampling_rate if sampling_rate is not None else system_config.sampling_rate
        self.ifft_size = system_config.ifft_size
        self.subcarrier_spacing = system_config.subcarrier_spacing
        self.cp_length_samples = system_config.cp_length_samples
        
        # 其他参数
        self.device = device
        
        # 调试信息存储
        self.debug_data = {}
        
        # Generate base sequence based on current configuration
        # self._update_base_sequence()
        
    def _update_base_sequence(self):
        """Update the base sequence when configuration changes"""
        # Use the current sequence length from the configuration
        self.base_seq = generate_base_sequence(length=self.srs_config.current_seq_length)
    
    def generate_srs_sequences(
        self,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        生成SRS序列（频域）
        
        Args:
            user_indices: 用户索引列表
            batch_size: 批次大小
            num_symbols: 符号数量
        Returns:
            sequences_tensor: [batch_size, num_user_ports, num_symbols, seq_length]
        """
        base_seq = self.base_seq
        cyclic_shifts = []
        meta_list = []
        for user_id in range(self.srs_config.num_users):
            num_ports = self.srs_config.ports_per_user[user_id]
            for port_id in range(num_ports):
                cyclic_shifts.append(self.srs_config.current_cyclic_shifts[user_id][port_id])
                meta_list.append((user_id, port_id))
        cyclic_shifts_tensor = torch.tensor(cyclic_shifts, dtype=base_seq.dtype, device=self.device)
        self.cyclic_shifts_tensor = cyclic_shifts_tensor
        self.meta_list = meta_list
        num_user_ports = len(cyclic_shifts_tensor)
        # Broadcast base_seq for all ports and batch
        base_seq_expanded = base_seq.unsqueeze(0).unsqueeze(0).expand(batch_size, num_user_ports, base_seq.shape[0])
        # Compute phase shift matrix: exp(1j * 2pi * cyclic_shift * n / K)
        n = torch.arange(base_seq.shape[0], device=self.device).unsqueeze(0)
        K = self.srs_config.K
        phase = torch.exp(1j * 2 * np.pi * cyclic_shifts_tensor.unsqueeze(0).unsqueeze(2) * n / K)
        phase = phase.expand(batch_size, num_user_ports, base_seq.shape[0])
        shifted_seqs = base_seq_expanded * phase
        sequences_tensor = shifted_seqs.to(dtype=torch.complex64, device=self.device)

        return sequences_tensor
    
    def map_to_subcarriers(
        self,
        sequences: torch.Tensor,
        ifft_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        将SRS序列映射到子载波
        
        Args:
            sequences: [batch_size, num_user_ports, seq_length]
            ifft_size: IFFT大小 (如果为None则使用系统配置)
            
        Returns:
            mapped_signals: [batch_size, num_user_ports, ifft_size]
        """
        if ifft_size is None:
            ifft_size = self.ifft_size
        
        # Generate comb mapping indices - place one sequence element every ktc subcarriers
        # Use current values from the active configuration
        seq_length = self.srs_config.current_seq_length
        ktc = self.srs_config.current_ktc  # Comb spacing
        
        # 简化映射：从索引0开始，步长为ktc
        # 确保映射索引不超出ifft_size范围
        
        mapping_indices = torch.arange(seq_length, dtype=torch.long, device=self.device) * ktc
        
        # sequences: (batch_size, num_user_ports, seq_length)
        batch_size, num_user_ports, seq_length = sequences.shape[:3]
        mapped_signals = torch.zeros((batch_size, num_user_ports, ifft_size), dtype=torch.complex64, device=self.device)
        valid_indices = mapping_indices[:seq_length]
        mapped_signals[:, :, valid_indices] = sequences[:, :, :seq_length]
        self.mapping_indices = mapping_indices
        return mapped_signals
    
    def ofdm_modulate(self, freq_signals: torch.Tensor) -> torch.Tensor:
        """
        OFDM调制 - 频域到时域（为每个端口独立处理）
        
        Args:
            freq_signals: [batch_size, num_user_ports, ifft_size]
            
        Returns:
            time_signals: [batch_size, num_user_ports, time_length]
        """
        # freq_signals: (num_user_ports, ifft_size)
        cp_length = self.cp_length_samples
        # freq_signals: (batch_size, num_user_ports, ifft_size)
        time_signals = torch.fft.ifft(freq_signals, dim=-1)
        if cp_length > 0:
            cp = time_signals[..., -cp_length:]
            cs = time_signals[..., :cp_length]
            time_signals_with_cp = torch.cat([cp, time_signals, cs], dim=-1)
        else:
            time_signals_with_cp = time_signals
        return time_signals_with_cp
    
    def ofdm_demodulate(self, time_signals: torch.Tensor, cp_length: Optional[int] = None) -> torch.Tensor:
        """
        OFDM解调 - 时域到频域
        
        Args:
            time_signals: 时域信号
            cp_length: 循环前缀长度 (如果为None则使用系统配置)
            
        Returns:
            频域信号
        """
        if cp_length is None:
            cp_length = self.cp_length_samples
        
        # 移除循环前缀
        if cp_length > 0:
            time_signals = time_signals[..., cp_length:]
        
        # FFT
        freq_signals = torch.fft.fft(time_signals, dim=-1)
        
        return freq_signals
    
    def add_noise(self, signals: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        添加高斯白噪声
        
        Args:
            signals: 输入信号
            snr_db: 信噪比 (dB)
            
        Returns:
            加噪后的信号
        """
        # 计算信号功率
        signal_power = torch.mean(torch.abs(signals) ** 2)
        
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 生成复高斯噪声
        noise_real = torch.randn_like(signals.real) * torch.sqrt(noise_power / 2)
        noise_imag = torch.randn_like(signals.imag) * torch.sqrt(noise_power / 2)
        noise = torch.complex(noise_real, noise_imag)
        
        return signals + noise
    
    def generate_pure_data_sample(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ifft_size = self.ifft_size
        self._update_base_sequence()
        # 1. 生成SRS序列（每个用户每个端口）
        sequences = self.generate_srs_sequences(batch_size)
        # 2. 映射到子载波（每个端口独立的频域网格）
        freq_grids = self.map_to_subcarriers(sequences, ifft_size)
        # 3. OFDM调制（每个端口独立的时域信号）
        time_signals = self.ofdm_modulate(freq_grids)
        return time_signals, sequences, freq_grids
    
    def get_debug_info(self) -> Dict[str, torch.Tensor]:
        """获取调试信息"""
        return self.debug_data.copy()

class SRSDataGenerator:  
    def __init__(
        self,
        base_generator: BaseSRSDataGenerator,
    ):
        self.base_generator = base_generator
    
    def generate_batch(
        self, 
        batch_size,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # Generate pure data (without channel)
        time_signals, sequences, freq_grids = self.base_generator.generate_pure_data_sample(batch_size)

        rx_signals, freq_channels = \
            self.channel_model._apply_sionna_channel(signals=time_signals, base_generator=self.base_generator)

        ls_estimates_tensor = rx_signals[:,:, self.base_generator.mapping_indices] / self.base_generator.base_seq[None,None,:]
        true_channel_tensor = freq_channels[...,self.base_generator.mapping_indices]

        return ls_estimates_tensor, true_channel_tensor


# ================================================================================
# 辅助函数：端口映射处理
# ================================================================================

def extract_channel_for_user_port(
    frequency_domain_channels: torch.Tensor,
    signal_port_mapping: Dict[int, Tuple[int, int]],
    target_user_id: int,
    target_port_id: int
) -> torch.Tensor:
    # 查找目标用户/端口对应的端口索引
    target_port_index = None
    for port_idx, (user_id, port_id) in signal_port_mapping.items():
        if user_id == target_user_id and port_id == target_port_id:
            target_port_index = port_idx
            break
    
    if target_port_index is None:
        raise ValueError(f"未找到用户{target_user_id}端口{target_port_id}的映射")
    
    # 提取对应的信道响应
    return frequency_domain_channels[:, target_port_index, :]


def get_all_user_port_channels(
    frequency_domain_channels: torch.Tensor,
    signal_port_mapping: Dict[int, Tuple[int, int]]
) -> Dict[Tuple[int, int], torch.Tensor]:

    result = {}
    for port_idx, (user_id, port_id) in signal_port_mapping.items():
        result[(user_id, port_id)] = frequency_domain_channels[:, port_idx, :]
    return result