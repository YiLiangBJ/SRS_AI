"""
重构后的SRS数据生成架构

将数据生成和信道建模完全分离：
1. BaseSRSDataGenerator: 纯数据生成，不包含信道
2. ChannelModelInterface: 统一的信道接口
3. SRSDataGenerator: 组合数据生成器+信道模型的高级接口

系统级物理层参数 (采样率、载波频率等) 从 SystemConfig 获取，确保全系统一致性。
用户级SRS参数 (用户数量、循环移位等) 从 SRSConfig 获取。

这样的设计更清晰，避免了循环依赖和重复调用的问题。
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Literal, Protocol
from abc import ABC, abstractmethod

from config import SRSConfig
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
        debug_dict: Optional[Dict] = None
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
        snr_range: Tuple[float, float] = (-10, 40),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        初始化基础数据生成器
        
        Args:
            config: 用户SRS配置
            system_config: 系统配置 (如果为None则使用默认配置)
            num_rx_antennas: 接收天线数量 (如果为None则从system_config获取)
            sampling_rate: 采样率 (如果为None则从system_config获取)
            snr_range: 信噪比范围 (dB)
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
        self.snr_range = snr_range
        self.device = device
        
        # 调试信息存储
        self.debug_data = {}
        
        print(f"✅ 基础SRS数据生成器初始化完成 (100% PyTorch)")
        print(f"   框架策略: 纯PyTorch实现，无TensorFlow依赖")
        print(f"   用户SRS配置: {config}")
        print(f"   系统配置:")
        print(f"   - 采样率: {self.sampling_rate/1e6:.2f} MHz")
        print(f"   - IFFT大小: {self.ifft_size}")
        print(f"   - 子载波间隔: {self.subcarrier_spacing/1e3:.0f} kHz")
        print(f"   - 接收天线: {self.num_rx_antennas}")
        print(f"   - SNR范围: {snr_range} dB")
        print(f"   - 设备: {device}")
    
    def generate_srs_sequences(
        self, 
        user_indices: List[int], 
        num_symbols: int = 1
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        生成SRS序列（频域）
        
        Args:
            user_indices: 用户索引列表
            num_symbols: 符号数量
            
        Returns:
            Dict[(user_id, port_id), frequency_domain_sequence]
        """
        sequences = {}
        
        # 首先生成基础序列
        base_seq = generate_base_sequence(
            length=self.config.seq_length
        )
        
        for user_id in user_indices:
            # 确保用户ID有效
            if user_id >= self.config.num_users:
                raise ValueError(f"User ID {user_id} exceeds configured users {self.config.num_users}")
            
            # 为该用户的每个端口生成序列
            num_ports = self.config.ports_per_user[user_id]
            for port_id in range(num_ports):
                # 获取该端口的循环移位
                cyclic_shift = self.config.cyclic_shifts[user_id][port_id]
                
                # 应用循环移位
                shifted_seq = apply_cyclic_shift(
                    base_seq, 
                    cyclic_shift,
                    self.config.K
                )
                
                # 转换为张量并移动到设备
                seq_tensor = torch.tensor(
                    shifted_seq, 
                    dtype=torch.complex64, 
                    device=self.device
                )
                
                # 扩展到多个符号
                if num_symbols > 1:
                    seq_tensor = seq_tensor.unsqueeze(0).repeat(num_symbols, 1)
                
                # 使用 (user_id, port_id) 作为键
                sequences[(user_id, port_id)] = seq_tensor
            
        return sequences
    
    def map_to_subcarriers(
        self, 
        sequences: Dict[Tuple[int, int], torch.Tensor], 
        ifft_size: Optional[int] = None
    ) -> Tuple[Dict[Tuple[int, int], torch.Tensor], torch.Tensor]:
        """
        将SRS序列映射到子载波
        
        Args:
            sequences: 用户端口SRS序列 Dict[(user_id, port_id), sequence]
            ifft_size: IFFT大小 (如果为None则使用系统配置)
            
        Returns:
            Tuple of (mapped_signals_dict, mapping_indices)
            - mapped_signals_dict: Dict[(user_id, port_id), full_freq_grid] 每个端口的完整频域网格
            - mapping_indices: 子载波映射索引 (所有端口共享相同的映射)
        """
        if ifft_size is None:
            ifft_size = self.ifft_size
        # 生成映射索引 (假设所有端口使用相同的子载波映射)
        seq_length = self.config.seq_length
        start_subcarrier = (ifft_size - seq_length) // 2  # 居中映射
        end_subcarrier = start_subcarrier + seq_length
        
        mapping_indices = torch.arange(
            start_subcarrier, end_subcarrier, 
            dtype=torch.long, device=self.device
        )
        
        # 为每个端口创建独立的频域网格
        mapped_signals = {}
        
        for (user_id, port_id), sequence in sequences.items():
            # 创建完整的频域网格
            freq_grid = torch.zeros(ifft_size, dtype=torch.complex64, device=self.device)
            
            # 映射序列到指定子载波
            actual_length = min(len(sequence), len(mapping_indices))
            freq_grid[mapping_indices[:actual_length]] = sequence[:actual_length]
            
            mapped_signals[(user_id, port_id)] = freq_grid
        
        return mapped_signals, mapping_indices
    
    def ofdm_modulate(self, freq_signals_dict: Dict[Tuple[int, int], torch.Tensor]) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        OFDM调制 - 频域到时域（为每个端口独立处理）
        
        Args:
            freq_signals_dict: 频域信号字典 Dict[(user_id, port_id), freq_signal]
            
        Returns:
            时域信号字典 Dict[(user_id, port_id), time_signal]
        """
        time_signals = {}
        
        for (user_id, port_id), freq_signal in freq_signals_dict.items():
            # IFFT
            time_signal = torch.fft.ifft(freq_signal) * torch.sqrt(torch.tensor(len(freq_signal), dtype=torch.float32))
            
            # 添加循环前缀（使用系统配置的CP长度）
            cp_length = self.cp_length_samples
            if cp_length > 0:
                # 添加前缀和后缀CP（用于处理时延偏移）
                time_signal_with_cp = torch.cat([
                    time_signal[-cp_length:],  # 前缀CP
                    time_signal,               # 完整符号
                    time_signal[:cp_length]    # 后缀CP（用于负时延偏移）
                ])
            else:
                time_signal_with_cp = time_signal
            
            time_signals[(user_id, port_id)] = time_signal_with_cp
        
        return time_signals
    
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
        user_indices: List[int], 
        snr_db: Optional[float] = None,
        ifft_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        生成纯数据样本（不包含信道）
        
        Args:
            user_indices: 用户索引列表
            snr_db: 信噪比，如果为None则从范围中随机选择
            ifft_size: IFFT大小 (如果为None则使用系统配置)
            ifft_size: IFFT大小
            
        Returns:
            包含所有数据的字典
        """
        if snr_db is None:
            snr_db = np.random.uniform(self.snr_range[0], self.snr_range[1])
        
        # 1. 生成SRS序列（每个用户每个端口）
        sequences = self.generate_srs_sequences(user_indices)
        
        # 2. 映射到子载波（每个端口独立的频域网格）
        freq_grids_dict, mapping_indices = self.map_to_subcarriers(sequences, ifft_size)
        
        # 3. OFDM调制（每个端口独立的时域信号）
        time_signals_dict = self.ofdm_modulate(freq_grids_dict)
        
        # 存储调试信息
        self.debug_data.update({
            'pure_sequences': sequences,
            'freq_grids_dict': freq_grids_dict,
            'time_signals_dict': time_signals_dict,
            'mapping_indices': mapping_indices,
            'snr_db': snr_db,
            'user_indices': user_indices
        })
        
        return {
            'tx_signals': time_signals_dict,  # Dict[(user_id, port_id), time_signal]
            'sequences': sequences,
            'freq_grids': freq_grids_dict,
            'mapping_indices': mapping_indices,
            'snr_db': snr_db,
            'ifft_size': ifft_size,
            'user_indices': user_indices
        }
    
    def finalize_received_data(
        self, 
        rx_signals: torch.Tensor, 
        data_sample: Dict[str, torch.Tensor],
        add_noise: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        完成接收数据处理（添加噪声、解调等）
        
        Args:
            rx_signals: 经过信道的接收信号 [num_rx_antennas, time_samples]
            data_sample: 原始数据样本
            add_noise: 是否添加噪声
            
        Returns:
            完整的数据样本
        """
        # 添加噪声
        if add_noise:
            rx_signals = self.add_noise(rx_signals, data_sample['snr_db'])
        
        # OFDM解调 - 处理每个接收天线
        rx_freq_list = []
        cp_length = self.cp_length_samples  # 使用系统配置的CP长度
        
        for rx_ant in range(rx_signals.shape[0]):
            rx_freq_ant = self.ofdm_demodulate(rx_signals[rx_ant], cp_length)
            rx_freq_list.append(rx_freq_ant)
        
        rx_freq = torch.stack(rx_freq_list, dim=0)  # [num_rx_antennas, ifft_size]
        
        # 提取映射的子载波
        mapping_indices = data_sample['mapping_indices']
        rx_mapped = rx_freq[..., mapping_indices]  # [num_rx_antennas, num_mapped_subcarriers]
        
        # 更新调试信息
        self.debug_data.update({
            'rx_time_signals': rx_signals,
            'rx_freq_signals': rx_freq,
            'rx_mapped_signals': rx_mapped
        })
        
        # 返回完整数据
        result = data_sample.copy()
        result.update({
            'rx_signals': rx_signals,
            'rx_freq': rx_freq,
            'rx_mapped': rx_mapped,
            'tx_signals': data_sample['tx_signals'],  # 保持字典格式
            'mapping_indices': mapping_indices
        })
        
        return result
    
    def get_debug_info(self) -> Dict[str, torch.Tensor]:
        """获取调试信息"""
        return self.debug_data.copy()
    
    def generate_batch(
        self, 
        batch_size: int, 
        user_indices: Optional[List[int]] = None,
        enable_debug: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成一批纯数据样本（不包含信道建模）
        
        🎯 为动态架构设计：支持BaseSRSDataGenerator独立生成批次数据
        
        Args:
            batch_size: 批次大小
            user_indices: 用户索引列表，如果为None则使用所有用户
            enable_debug: 是否启用调试模式
            **kwargs: 传递给generate_pure_data_sample的其他参数
            
        Returns:
            批次数据字典，每个键对应的值都是堆叠的张量
        """
        if user_indices is None:
            user_indices = list(range(self.config.num_users))
        
        batch_samples = []
        
        for i in range(batch_size):
            sample = self.generate_pure_data_sample(
                user_indices=user_indices,
                **kwargs
            )
            batch_samples.append(sample)
        
        # 合并批次数据
        batch_data = {}
        
        # 处理需要堆叠的张量数据
        tensor_keys = ['mapping_indices', 'snr_db']
        scalar_keys = ['ifft_size']  # 这些可能为None或标量
        dict_keys = ['tx_signals', 'sequences', 'freq_grids']  # 这些保持为列表
        list_keys = ['user_indices']
        
        for key in batch_samples[0].keys():
            if key in tensor_keys:
                # 对于标量或张量，直接堆叠
                values = [s[key] for s in batch_samples]
                if isinstance(values[0], torch.Tensor):
                    batch_data[key] = torch.stack(values)
                else:
                    batch_data[key] = torch.tensor(values)
            elif key in scalar_keys:
                # 对于可能为None的标量，特殊处理
                values = [s[key] for s in batch_samples]
                if values[0] is not None:
                    batch_data[key] = torch.tensor(values) if not isinstance(values[0], torch.Tensor) else torch.stack(values)
                else:
                    batch_data[key] = values  # 保持为None列表
            elif key in dict_keys:
                # 对于字典数据，保持为列表（每个样本一个字典）
                batch_data[key] = [s[key] for s in batch_samples]
            elif key in list_keys:
                # 对于列表数据，保持为列表
                batch_data[key] = [s[key] for s in batch_samples]
            else:
                # 其他数据类型，尝试堆叠或保持为列表
                try:
                    values = [s[key] for s in batch_samples]
                    if isinstance(values[0], torch.Tensor):
                        batch_data[key] = torch.stack(values)
                    else:
                        batch_data[key] = values
                except:
                    batch_data[key] = [s[key] for s in batch_samples]
        
        if enable_debug:
            print(f"✅ BaseSRSDataGenerator批次生成完成:")
            print(f"   批次大小: {batch_size}")
            print(f"   用户索引: {user_indices}")
            print(f"   数据键: {list(batch_data.keys())}")
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor):
                    print(f"   {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"   {key}: list of {len(value)} items")
                else:
                    print(f"   {key}: {type(value)}")
        
        return batch_data


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
        snr_range: Tuple[float, float] = (-10, 40),
        sampling_rate: float = 122.88e6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        初始化完整数据生成器
        
        Args:
            config: SRS配置
            channel_model: 信道模型（如果为None则不应用信道）
            其他参数同BaseSRSDataGenerator
        """
        # 创建基础数据生成器
        self.base_generator = BaseSRSDataGenerator(
            config=config,
            num_rx_antennas=num_rx_antennas,
            snr_range=snr_range,
            sampling_rate=sampling_rate,
            device=device,
            **kwargs
        )
        
        # 设置信道模型
        self.channel_model = channel_model
        self.using_channel = channel_model is not None
        
        print(f"✅ SRS数据生成器初始化完成")
        print(f"   框架策略: PyTorch主导 + 信道部分可选TensorFlow")
        print(f"   数据生成: 100% PyTorch (BaseSRSDataGenerator)")
        print(f"   使用信道模型: {'是' if self.using_channel else '否'}")
        if self.using_channel:
            print(f"   信道模型类型: {type(channel_model).__name__}")
            print(f"   信道内部实现: 对外透明 (输入输出都是PyTorch)")
        else:
            print(f"   信道建模: 跳过，纯数据生成模式")
    
    def generate_sample(
        self, 
        user_indices: Optional[List[int]] = None,
        snr_db: Optional[float] = None,
        ifft_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        生成一个完整的训练样本
        
        Args:
            user_indices: 用户索引列表（如果为None则使用config中定义的所有用户）
            snr_db: 信噪比
            ifft_size: IFFT大小 (如果为None则使用系统配置)
            
        Returns:
            完整的数据样本
        """
        # 使用系统配置的IFFT大小
        if ifft_size is None:
            ifft_size = self.base_generator.ifft_size
        # 确定用户列表 - 从config获取，不再随机生成
        if user_indices is None:
            # 使用config中定义的所有用户
            user_indices = list(range(self.base_generator.config.num_users))
        
        # 1. 生成纯数据（不含信道）
        data_sample = self.base_generator.generate_pure_data_sample(
            user_indices, snr_db, ifft_size
        )
        
        # 2. 应用信道模型（如果有）
        if self.using_channel:
            debug_dict = {}
            
            # 应用信道
            rx_signals, freq_channels = self.channel_model.apply_channel(
                signals=data_sample['tx_signals'],
                user_config=self.base_generator.config,  # 🎯 传入用户配置
                mapping_indices=data_sample['mapping_indices'],
                ifft_size=ifft_size,
                debug_dict=debug_dict
            )
            
            # 存储信道调试信息
            data_sample['channel_debug'] = debug_dict
            data_sample['freq_channels'] = freq_channels
        else:
            # 不使用信道，简单模拟多用户接收
            tx_signals_dict = data_sample['tx_signals']  # Dict[(user_id, port_id), time_signal]
            
            # 将所有发送信号相加作为接收信号
            time_samples = None
            combined_signal = None
            
            for (user_id, port_id), tx_signal in tx_signals_dict.items():
                if time_samples is None:
                    time_samples = len(tx_signal)
                    combined_signal = torch.zeros(time_samples, dtype=torch.complex64, device=tx_signal.device)
                combined_signal += tx_signal
            
            # 复制到多个接收天线
            if combined_signal is not None:
                rx_signals = combined_signal.unsqueeze(0).repeat(self.base_generator.num_rx_antennas, 1)
            else:
                # 如果没有信号，创建零信号
                rx_signals = torch.zeros(
                    self.base_generator.num_rx_antennas, 2048,  # 默认长度
                    dtype=torch.complex64, device=self.base_generator.device
                )
        
        # 3. 完成接收端处理
        final_sample = self.base_generator.finalize_received_data(
            rx_signals, data_sample, add_noise=True
        )
        
        return final_sample
    
    def generate_batch(self, batch_size: int, **kwargs) -> Dict[str, torch.Tensor]:
        """生成一批训练样本"""
        batch_samples = []
        
        for i in range(batch_size):
            sample = self.generate_sample(**kwargs)
            batch_samples.append(sample)
        
        # 合并批次
        batch_data = {}
        for key in batch_samples[0].keys():
            if isinstance(batch_samples[0][key], torch.Tensor):
                batch_data[key] = torch.stack([s[key] for s in batch_samples])
            else:
                batch_data[key] = [s[key] for s in batch_samples]
        
        return batch_data
    
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
