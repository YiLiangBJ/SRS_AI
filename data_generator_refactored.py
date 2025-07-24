"""
Refactored SRS Data Generation Architecture

Complete separation of data generation and channel modeling:
1. BaseSRSDataGenerator: Pure data generation without channel
2. ChannelModelInterface: Unified channel interface
3. SRSDataGenerator: High-level interface combining data generator + channel model

System-level physical layer parameters (sampling rate, carrier frequency, etc.) are obtained from SystemConfig to ensure system-wide consistency.
User-level SRS parameters (number of users, cyclic shifts, etc.) are obtained from SRSConfig.

这样的设计更清晰，避免了循环依赖和重复调用的问题。
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal, Protocol
from abc import ABC, abstractmethod

from user_config import SRSConfig
from system_config import SystemConfig, create_default_system_config
from utils import generate_base_sequence, apply_cyclic_shift


# 简化版本：不使用Protocol接口，直接使用类型提示
# from typing import Union, Optional
# 
# # 可以接受任何有apply_channel方法的对象
# ChannelModelType = Union['SIONNAChannelModel', 'SimpleFallbackChannelModel', None]


class ChannelModelInterface(Protocol):
    """
    信道模型统一接口
    
    这是一个Protocol接口，定义了所有信道模型必须实现的方法。
    Protocol不包含具体实现，只定义方法签名和文档。
    
    🎯 框架使用策略：
    - 输入：PyTorch张量
    - 输出：PyTorch张量  
    - 内部实现：信道建模部分可以使用TensorFlow/SIONNA，但必须对外透明
    - 信号处理：所有非信道建模的信号处理都使用PyTorch
    
    具体实现在：
    - SIONNAChannelModel (professional_channels.py) - 内部使用TensorFlow/SIONNA
    - SimpleFallbackChannelModel (如果需要) - 纯PyTorch实现
    """
    
    def apply_channel(
        self,
        signals: Union[torch.Tensor, List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        delay_offset_samples: int = 0,
        mapping_indices: Optional[torch.Tensor] = None,
        ifft_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用信道模型到输入信号
        
        🎯 输入输出都是PyTorch张量，内部实现可以使用任何框架
        
        Args:
            signals: 输入信号 (PyTorch张量) - Dict[(user_id, port_id), time_signal] 每个端口的独立时域信号
            delay_offset_samples: 延迟偏移采样点数
            mapping_indices: 子载波映射索引 (PyTorch张量)
            ifft_size: IFFT大小
            debug_dict: 调试信息存储字典，会存储端口映射信息
            
        Returns:
            Tuple of (output_signals, frequency_domain_channels) - 都是PyTorch张量
            - output_signals: 输出信号 [num_rx_antennas, time_samples] (PyTorch)
            - frequency_domain_channels: 频域信道 [num_rx_antennas, num_tx_ports, num_subcarriers] (PyTorch)
            
        端口映射说明：
            当输入signals为字典Dict[(user_id, port_id), time_signal]时：
            1. 信道模型保持输入字典的原始键顺序（不使用sorted()）
            2. 输出的frequency_domain_channels第2维（num_tx_ports）按原始输入顺序对应端口
            3. debug_dict['signal_port_mapping']存储{port_index: (user_id, port_id)}映射
            4. 调用者可通过此映射将信道矩阵正确关联到对应用户/端口
            5. 这确保了每个用户每个端口的信号都明确通过其对应的信道
            
        注意：
        - 所有输入输出都必须是PyTorch张量
        - 内部实现可以使用TensorFlow/SIONNA，但必须转换回PyTorch
        - 这确保了整个项目的框架一致性
        """
        ...


class BaseSRSDataGenerator:
    """
    基础SRS数据生成器 - 纯PyTorch数据生成，不包含任何信道建模
    
    🎯 框架策略：100% PyTorch实现
    
    这个类专注于：
    1. SRS序列生成 (PyTorch)
    2. 资源映射 (PyTorch) 
    3. OFDM调制/解调 (PyTorch)
    4. 噪声添加 (PyTorch)
    5. 数据格式化 (PyTorch)
    
    ⚠️ 不包含任何信道建模逻辑！
    ⚠️ 不使用TensorFlow！所有计算都在PyTorch中完成！
    """
    
    def __init__(
        self,
        config: SRSConfig,  # 用户SRS配置
        system_config: Optional[SystemConfig] = None,  # 系统配置
        # 以下参数如果为None，将从system_config获取
        num_rx_antennas: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        # 其他参数
        device: str = "cpu",  # Force CPU-only execution
        **kwargs
    ):
        """
        初始化基础数据生成器
        
        Args:
            config: 用户SRS配置 (包含SNR配置)
            system_config: 系统配置 (如果为None则使用默认配置)
            num_rx_antennas: 接收天线数量 (如果为None则从system_config获取)
            sampling_rate: 采样率 (如果为None则从system_config获取)
            device: 计算设备
        """
        # 使用系统配置或创建默认配置
        if system_config is None:
            system_config = create_default_system_config()
        
        self.config = config  # 用户SRS配置
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
        self._update_base_sequence()
        
    def _update_base_sequence(self):
        """Update the base sequence when configuration changes"""
        # Use the current sequence length from the configuration
        self.base_seq = generate_base_sequence(length=self.config.current_seq_length)
    
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
        for user_id in range(self.config.num_users):
            num_ports = self.config.ports_per_user[user_id]
            for port_id in range(num_ports):
                cyclic_shifts.append(self.config.current_cyclic_shifts[user_id][port_id])
                meta_list.append((user_id, port_id))
        cyclic_shifts_tensor = torch.tensor(cyclic_shifts, dtype=base_seq.dtype, device=self.device)
        self.cyclic_shifts_tensor = cyclic_shifts_tensor
        self.meta_list = meta_list
        num_user_ports = len(cyclic_shifts_tensor)
        # Broadcast base_seq for all ports and batch
        base_seq_expanded = base_seq.unsqueeze(0).unsqueeze(0).expand(batch_size, num_user_ports, base_seq.shape[0])
        # Compute phase shift matrix: exp(1j * 2pi * cyclic_shift * n / K)
        n = torch.arange(base_seq.shape[0], device=self.device).unsqueeze(0)
        K = self.config.K
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
        seq_length = self.config.current_seq_length
        ktc = self.config.current_ktc  # Comb spacing
        
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
        ifft_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate pure data sample (without channel and noise), batch-parallel
        Args:
            batch_size: 批次大小
            user_indices: User index list
            ifft_size: IFFT size (if None, use system config)
        Returns:
            Dictionary containing all data (without SNR information), batch-parallel
        """
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
    """
    完整的SRS数据生成器 - 组合数据生成器和信道模型
    
    🎯 框架策略：主要PyTorch + 信道部分可选TensorFlow/SIONNA
    
    这个类负责协调：
    1. 纯数据生成（通过BaseSRSDataGenerator - 100% PyTorch）
    2. 信道建模（通过ChannelModelInterface - 内部可能使用TensorFlow/SIONNA）
    3. 端到端的数据流处理（100% PyTorch输入输出）
    
    关键设计原则：
    - 所有接口都是PyTorch张量
    - 信道模型内部实现对用户透明
    - 确保项目整体的PyTorch一致性
    """
    
    def __init__(
        self,
        config: SRSConfig,
        channel_model: Optional[ChannelModelInterface] = None,
        num_rx_antennas: int = 4,
        sampling_rate: float = 122.88e6,
        device: str = "cpu",  # Force CPU-only execution
        **kwargs
    ):
        """
        初始化完整数据生成器
        
        Args:
            config: SRS配置 (包含SNR配置)
            channel_model: 信道模型（如果为None则不应用信道）
            其他参数同BaseSRSDataGenerator
        """
        # 创建基础数据生成器
        self.base_generator = BaseSRSDataGenerator(
            config=config,
            num_rx_antennas=num_rx_antennas,
            sampling_rate=sampling_rate,
            device=device,
            **kwargs
        )
        
        # 设置信道模型
        self.channel_model = channel_model
        self.using_channel = channel_model is not None
    
    def generate_batch(
        self, 
        batch_size,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:

        ifft_size = self.base_generator.ifft_size

        # Generate pure data (without channel)
        time_signals, sequences, freq_grids = self.base_generator.generate_pure_data_sample(
            batch_size, ifft_size
        )
        # If SNR is not explicitly provided, get it from the config
        snr_db = kwargs.get('snr_db', None)
        if snr_db is None:
            snr_db = self.base_generator.config.get_snr_db()
        # 应用信道
        rx_signals, freq_channels = self.channel_model.apply_channel(
            signals=time_signals,
            user_config=self.base_generator.config,
            mapping_indices=self.base_generator.mapping_indices,
            ifft_size=ifft_size,
        )
        base_seq = self.base_generator.base_seq
        ls_estimates_tensor = rx_signals[:, self.base_generator.mapping_indices] / base_seq.unsqueeze(0)
        true_channel_tensor = freq_channels[...,self.base_generator.mapping_indices]

        return ls_estimates_tensor, true_channel_tensor
    
    def get_debug_info(self) -> Dict[str, torch.Tensor]:
        """获取调试信息"""
        debug_info = self.base_generator.get_debug_info()
        if self.using_channel and hasattr(self.channel_model, 'get_debug_info'):
            debug_info['channel'] = self.channel_model.get_debug_info()
        return debug_info


# ================================================================================
# 辅助函数：端口映射处理
# ================================================================================

def extract_channel_for_user_port(
    frequency_domain_channels: torch.Tensor,
    signal_port_mapping: Dict[int, Tuple[int, int]],
    target_user_id: int,
    target_port_id: int
) -> torch.Tensor:
    """
    从信道矩阵中提取特定用户/端口的信道响应
    
    Args:
        frequency_domain_channels: 频域信道矩阵 [num_rx_antennas, num_tx_ports, num_subcarriers]
        signal_port_mapping: 端口索引映射 {port_index: (user_id, port_id)}
        target_user_id: 目标用户ID
        target_port_id: 目标端口ID
        
    Returns:
        torch.Tensor: 特定用户/端口的信道响应 [num_rx_antennas, num_subcarriers]
        
    Example:
        # 从信道调试信息中获取映射
        debug_info = data_sample['channel_debug']
        port_mapping = debug_info['signal_port_mapping']
        h_freq = data_sample['freq_channels']
        
        # 提取用户1端口0的信道响应
        user1_port0_channel = extract_channel_for_user_port(
            h_freq, port_mapping, user_id=1, port_id=0
        )
    """
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
    """
    将信道矩阵按用户/端口重新组织为字典
    
    Args:
        frequency_domain_channels: 频域信道矩阵 [num_rx_antennas, num_tx_ports, num_subcarriers]
        signal_port_mapping: 端口索引映射 {port_index: (user_id, port_id)}
        
    Returns:
        Dict[(user_id, port_id), torch.Tensor]: 每个用户/端口的信道响应
        
    Example:
        debug_info = data_sample['channel_debug']
        port_mapping = debug_info['signal_port_mapping']
        h_freq = data_sample['freq_channels']
        
        # 获取所有用户/端口的信道响应
        all_channels = get_all_user_port_channels(h_freq, port_mapping)
        user1_port0_channel = all_channels[(1, 0)]  # 用户1端口0的信道
    """
    result = {}
    for port_idx, (user_id, port_id) in signal_port_mapping.items():
        result[(user_id, port_id)] = frequency_domain_channels[:, port_idx, :]
    return result


# ================================================================================
# 向后兼容性别名
# ================================================================================

class SRSDataGeneratorLegacy(SRSDataGenerator):
    """向后兼容的数据生成器别名"""
    pass
