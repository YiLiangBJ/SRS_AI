"""
SIONNA Professional Channel Models for SRS Channel Estimation

This module provides SIONN        print(f"🚀 Initializing SIONNA {self.model_type} channel model")
        print(f"   System configuration parameters:")
        print(f"   - Sampling rate: {self.sampling_rate/1e6:.2f} MHz (= {self.subcarrier_spacing/1e3:.0f} kHz × {self.ifft_size})")
        print(f"   - Carrier frequency: {self.carrier_frequency/1e9:.1f} GHz")
        print(f"   - Subcarrier spacing: {self.subcarrier_spacing/1e3:.0f} kHz")
        print(f"   - IFFT size: {self.ifft_size}")
        self._init_sionna()based professional 3GPP channel models.
SIONNA is the industry standard for wireless communication simulation.

Requirements (install with Intel proxy):
- python -m pip install --proxy http://child-prc.intel.com:913 sionna
- python -m pip install --proxy http://child-prc.intel.com:913 tensorflow>=2.13.0

Alternative installation (without proxy):
- python -m pip install sionna
- python -m pip install tensorflow>=2.13.0
"""
import math
import torch
import numpy as np
from typing import Optional, Literal, Tuple, Dict, Union, List
import warnings

# Import system configuration
from system_config import SystemConfig, create_default_system_config
from data_generator import SRSDataGenerator

import sionna
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
import tensorflow as tf
SIONNA_AVAILABLE = True
print("SIONNA professional channel library loaded successfully")



class SIONNAChannelModel:
    """
    SIONNA-based professional 3GPP channel model with per-UE TDL instantiation.
    
    🔧 Architecture Design (Physically Meaningful TDL Usage):
    
    1. **Per-UE TDL Strategy**:
       - No global TDL instance created at initialization
       - Individual TDL instance created for each UE at runtime
       - Each UE's TDL: num_tx_ant = number of ports for that UE
       - More physically realistic: each UE has independent channel environment
    
    2. **Framework Usage**:
       - 输入：PyTorch张量
       - 信道建模：TensorFlow/SIONNA (内部使用)
       - 输出：PyTorch张量
       - 信号处理：全部PyTorch
    
    3. **Physical Benefits**:
       - 每个UE独立的信道环境
       - UE-specific延迟：同一UE所有port共享延迟
       - 避免维度不匹配问题
       - 内存友好：TDL实例即用即删
    
    Uses NVIDIA/TU Wien SIONNA library for industry-standard channel modeling.
    Supports all 3GPP TR 38.901 TDL and CDL models.
    
    System parameters (采样率、载波频率等) 从 SystemConfig 获取，确保全系统一致性。
    """
    
    def __init__(
        self,
        system_config: Optional[SystemConfig] = None,
        model_type: Optional[Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"]] = None,
        num_rx_antennas: Optional[int] = None,
        delay_spread: Optional[float] = None,  # in seconds
        k_factor: Optional[float] = None,  # Ricean K-factor (linear scale)
        device: str = "cpu"  # Force CPU-only execution
    ):
        """
        Initialize SIONNA channel model
        
        Args:
            system_config: 系统配置 (如果为None则使用默认配置)
            model_type: Channel model type (如果为None则从system_config获取)
            num_rx_antennas: Number of receive antennas (如果为None则从system_config获取)
            num_tx_antennas: Number of transmit antennas  
            delay_spread: RMS delay spread in seconds (如果为None则从system_config获取)
            k_factor: Ricean K-factor (如果为None则从system_config获取)
            device: PyTorch device
        """
        # Enforce required parameters
        if system_config is None:
            raise ValueError("system_config must be provided and cannot be None")
        self.system_config = system_config
        
        # Enforce strict parameter requirements
        if model_type is None:
            raise ValueError("model_type must be provided and cannot be None")
        if num_rx_antennas is None:
            raise ValueError("num_rx_antennas must be provided and cannot be None")
        if delay_spread is None:
            raise ValueError("delay_spread must be provided and cannot be None")
        if k_factor is None:
            # Default K-factor value
            self.k_factor = 0.0  # For non-LOS models
        else:
            self.k_factor = k_factor
        
        self.model_type = model_type
        self.num_rx_antennas = num_rx_antennas
        self.delay_spread = delay_spread
        # k_factor is already set in the condition check above
        
        # Enforce strict device usage
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was specified as device but no CUDA-capable GPU is available. Please specify 'cpu' if CPU execution is intended.")
        
        # 从系统配置获取统一的参数
        self.carrier_frequency = system_config.carrier_frequency
        self.sampling_rate = system_config.sampling_rate
        self.subcarrier_spacing = system_config.subcarrier_spacing
        self.ifft_size = system_config.ifft_size
        
        self._init_sionna()
    
    def _init_sionna(self):
        # Process K-factor for LOS models
        if self.model_type in ["TDL-D", "TDL-E"]:
            # For LOS models, k_factor must be properly defined
            if self.k_factor == 0.0:
                # No automatic defaults - if 0.0 was provided, alert the user
                raise ValueError(f"K-factor must be non-zero for LOS models ({self.model_type}). Please specify a valid K-factor value.")
    
    def apply_channel(
        self,
        signals: Union[torch.Tensor, List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        srs_config,  # 🔧 新增：用户配置，用于确定每个UE的port数量
        delay_offset_samples: Optional[int] = None,  # 如果为None则随机生成
        mapping_indices: Optional[torch.Tensor] = None,
        ifft_size: Optional[int] = None,
        snr_db: Optional[float] = None  # 🔧 新增：SNR参数
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 如果没有指定时间偏移，则从用户配置随机生成
        if delay_offset_samples is None:
            delay_offset_samples = srs_config.get_timing_offset_samples(self.sampling_rate)
        
        # 如果没有指定SNR，则从用户配置获取
        if snr_db is None:
            snr_db = srs_config.get_snr_db()
        
        return self.SRSTrainer(signals, srs_config, delay_offset_samples, mapping_indices, ifft_size, snr_db)
    
    def validate_tdl_parameters(self):
        """验证 TDL 初始化参数的合理性"""
        print("🔍 验证 TDL 参数...")
        
        # 检查延迟扩展
        if self.delay_spread <= 0:
            raise ValueError(f"延迟扩展必须为正数，当前值: {self.delay_spread}")
        if self.delay_spread > 1e-6: # 1 us
            print(f"⚠️  延迟扩展较大: {self.delay_spread*1e9:.1f} ns")
        
        # 检查载波频率
        if self.carrier_frequency <= 0:
            raise ValueError(f"载波频率必须为正数，当前值: {self.carrier_frequency}")
        if self.carrier_frequency < 1e9 or self.carrier_frequency > 100e9:
            print(f"⚠️  载波频率超出常见范围: {self.carrier_frequency/1e9:.1f} GHz")
        
        # 检查天线数量
        if self.num_rx_antennas <= 0 or self.num_tx_antennas <= 0:
            raise ValueError(f"天线数量必须为正整数，RX: {self.num_rx_antennas}, TX: {self.num_tx_antennas}")
        
        print("TDL parameters validated")
    
    def validate_tdl_output(self, h, delays, batch_size, num_time_steps):
        """验证 TDL 输出的合理性"""
        print("🔍 验证 TDL 输出...")
        
        # 检查形状
        expected_h_shape = [batch_size, 1, self.num_rx_antennas, 1, self.num_tx_antennas, -1, num_time_steps]
        expected_delays_shape = [batch_size, 1, 1, -1]
        
        print(f"   信道系数形状: {h.shape} (期望: {expected_h_shape})")
        print(f"   延迟形状: {delays.shape} (期望: {expected_delays_shape})")
        
        # 检查数值范围
        h_magnitude = tf.abs(h)
        h_min = tf.reduce_min(h_magnitude)
        h_max = tf.reduce_max(h_magnitude)
        delays_min = tf.reduce_min(delays)
        delays_max = tf.reduce_max(delays)
        
        print(f"   信道系数幅度范围: [{h_min:.6f}, {h_max:.6f}]")
        print(f"   延迟范围: [{delays_min:.9f}, {delays_max:.9f}] 秒")
        print(f"   延迟范围: [{delays_min*1e9:.1f}, {delays_max*1e9:.1f}] 纳秒")
        
        # 检查 NaN 和无穷大 (对于复数，需要检查实部和虚部)
        h_real = tf.math.real(h)
        h_imag = tf.math.imag(h)
        
        has_nan_h_real = tf.reduce_any(tf.math.is_nan(h_real))
        has_nan_h_imag = tf.reduce_any(tf.math.is_nan(h_imag))
        has_inf_h_real = tf.reduce_any(tf.math.is_inf(h_real))
        has_inf_h_imag = tf.reduce_any(tf.math.is_inf(h_imag))
        
        has_nan_delays = tf.reduce_any(tf.math.is_nan(delays))
        has_inf_delays = tf.reduce_any(tf.math.is_inf(delays))
        
        has_nan_h = has_nan_h_real or has_nan_h_imag
        has_inf_h = has_inf_h_real or has_inf_h_imag
        
        if has_nan_h or has_inf_h or has_nan_delays or has_inf_delays:
            raise ValueError(f"TDL 输出包含无效值 - H: NaN={has_nan_h}, Inf={has_inf_h}, Delays: NaN={has_nan_delays}, Inf={has_inf_delays}")
        
        # 检查功率归一化 (可选)
        total_power = tf.reduce_mean(tf.square(h_magnitude))
        print(f"   平均信道功率: {total_power:.6f}")
        
        print("TDL output validated")
        return h, delays
    
    def debug_create_user_tdl(self, user_id, num_ports, batch_size=1, num_time_steps=1, sampling_frequency=None):
        """
        调试为特定用户创建和测试TDL实例的过程
        
        在per-UE TDL设计中，我们不再有全局TDL实例，而是为每个用户创建专用实例。
        此方法用于调试单个用户的TDL创建和调用过程。
        
        Args:
            user_id: 用户ID
            num_ports: 该用户的端口数量（将设置为TDL的num_tx_ant）
            batch_size: 批次大小
            num_time_steps: 时间步数
            sampling_frequency: 采样频率
        """
        print(f"🐛 调试用户 {user_id} 的TDL创建过程 (num_ports={num_ports})...")
        
        # 使用系统配置的采样率
        if sampling_frequency is None:
            sampling_frequency = self.sampling_rate
        
        try:
            # 创建用户专用TDL实例
            model_letter = self.model_type.split("-")[1]
            print(f"🔧 为用户 {user_id} 创建TDL:")
            print(f"   model: {model_letter}")
            print(f"   delay_spread: {self.delay_spread}")
            print(f"   carrier_frequency: {self.carrier_frequency}")
            print(f"   num_rx_ant: {self.num_rx_antennas}")
            print(f"   num_tx_ant: {num_ports}")
            print(f"   precision: 'single'")
            
            user_channel = TDL(
                model=model_letter,
                delay_spread=self.delay_spread,
                carrier_frequency=self.carrier_frequency,
                num_rx_ant=self.num_rx_antennas,
                num_tx_ant=num_ports,  # 关键：设置为用户的port数
                precision='single'
            )
            print(f"User {user_id} TDL instance created successfully")
            
            # 测试TDL调用
            print(f"🎯 测试用户 {user_id} TDL调用:")
            print(f"   batch_size: {batch_size}")
            print(f"   num_time_steps: {num_time_steps}")
            print(f"   sampling_frequency: {sampling_frequency/1e6:.2f} MHz")
            
            h, delays = user_channel(batch_size, num_time_steps, sampling_frequency)
            print(f"🎯 用户 {user_id} TDL调用成功")
            
            # 验证输出维度
            print(f"📊 用户 {user_id} 输出维度验证:")
            print(f"   h.shape: {h.shape}")
            print(f"   delays.shape: {delays.shape}")
            print(f"   期望tx_ant维度: {num_ports}")
            print(f"   实际tx_ant维度: {h.shape[4]}")
            
            if h.shape[4] != num_ports:
                print(f"⚠️  TX天线维度不匹配！")
            else:
                print(f"TX antenna dimensions perfectly matched")
            
            # 清理TDL实例
            del user_channel
            print(f"🧹 用户 {user_id} TDL实例已清理")
            
            print(f"User {user_id} TDL debugging successful")
            return h, delays
            
        except Exception as e:
            print(f"❌ 用户 {user_id} TDL调试失败:")
            print(f"   错误类型: {type(e).__name__}")
            print(f"   错误信息: {str(e)}")
            
            # 尝试提供有用的诊断信息
            if "sampling_frequency" in str(e):
                print("💡 提示: 检查采样频率参数")
            elif "batch_size" in str(e):
                print("💡 提示: 检查批次大小参数")
            elif "Out of memory" in str(e):
                print("💡 提示: 尝试减少批次大小或时间步数")
            
            raise
    
    def _apply_sionna_channel(
        self,
        base_generator: SRSDataGenerator,
        signals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # signals: [batch_size, num_ports_total, signal_length]
        batch_size, num_ports_total, signal_length = signals.shape

        ifft_size = base_generator.ifft_size
        srs_config = base_generator.srs_config
        mapping_indices = base_generator.mapping_indices
        snr_db = base_generator.srs_config.get_snr_db(batch_size=batch_size)
        delay_offset_samples = base_generator.srs_config.get_timing_offset_samples(base_generator.sampling_rate, batch_size=batch_size)

        
        num_rx_ant = self.num_rx_antennas

        tf_device = '/CPU:0' if self.device == 'cpu' else '/GPU:0'
        with tf.device(tf_device):
            model_letter = self.model_type.split("-")[1]
            user_channel = TDL(
                model=model_letter,
                delay_spread=self.delay_spread,
                carrier_frequency=self.carrier_frequency,
                num_rx_ant=num_rx_ant,
                num_tx_ant=num_ports_total,
                precision='single'
            )
            cir = user_channel(batch_size=batch_size, num_time_steps=1, sampling_frequency=self.sampling_rate)
            h_time_tf, delays_tf = cir
            h_time = torch.from_numpy(h_time_tf.numpy()).to(self.device)  # [batch_size, 1, num_rx_ant, 1, num_ports_total, num_paths, 1]
            delays = torch.from_numpy(delays_tf.numpy()).to(self.device)  # [batch_size, 1, 1, num_paths]
            del user_channel, cir, h_time_tf, delays_tf

        # 构建时域信道冲激响应 (全batch tensor化)
        h_impulse_tensor = self._construct_time_domain_channel_pytorch(h_time, delays, signal_length)  # [batch_size, num_rx_ant, num_ports_total, signal_length]

        # 并行卷积 (全batch tensor化)
        received_tensor = self._time_domain_convolution(signals, h_impulse_tensor)  # [batch_size, num_rx_ant, num_ports_total, signal_length]

        # timing offset
        received_tensor_to = torch.stack([
            torch.roll(received_tensor[b], shifts=int(delay_offset_samples[b].item()), dims=-1)
            for b in range(received_tensor.shape[0])
        ], dim=0)

        # 并行频域信道计算 (全batch tensor化)
        h_freq_tensor = self._cir_to_ofdm_channel_pytorch_fft(h_impulse_tensor, ifft_size)  # [batch_size, num_rx_ant, num_ports_total, num_subcarriers]
        h_freq_tensor_to = self._apply_timing_offset_to_freq_channel(h_freq_tensor, delay_offset_samples, ifft_size)  # [batch_size, num_rx_ant, num_ports_total, num_subcarriers]

        # 功率归一化
        snr_linear = 10 ** (snr_db / 10.0)  # [batch_size]
        signal_power = snr_linear  # [batch_size]

        scales = []
        for num_ports in srs_config.ports_per_user:
            scales.extend([1/np.sqrt(num_ports)] * num_ports)
        scales = torch.tensor(scales, dtype=torch.float32, device=h_freq_tensor_to.device)
        scales = scales.view(1, 1, -1, 1)

        # signal_power: [batch_size]，需要 reshape 成 [batch_size, 1, 1, 1] 以便广播
        signal_power = signal_power.view(-1, 1, 1, 1)
        received_tensor_to = received_tensor_to * scales * torch.sqrt(signal_power)
        h_freq_tensor_to = h_freq_tensor_to * scales * torch.sqrt(signal_power)

        received_tensor_to_rx = torch.sum(received_tensor_to, dim=2)
        # 加噪声
        noise_var = 1
        noise = math.sqrt(noise_var / 2) / math.sqrt(ifft_size) * (
            torch.randn_like(received_tensor_to_rx) + 1j * torch.randn_like(received_tensor_to_rx)
        )
        noisy_received = received_tensor_to_rx + noise
        cp_length = self.system_config.cp_length_samples
        data_part = noisy_received[..., cp_length:cp_length + ifft_size]
        received_freq = torch.fft.fft(data_part, dim=-1)  # [batch_size, num_rx_ant, num_ports_total, num_subcarriers]

        # 输出全部为张量
        return received_freq, h_freq_tensor_to
    
    def _construct_time_domain_channel_pytorch(
        self,
        h_time: torch.Tensor,
        delays: torch.Tensor,
        signal_length: int
    ) -> torch.Tensor:
        """
        并行构建时域信道冲激响应 (全batch tensor化)
        Args:
            h_time: [batch_size, 1, num_rx_ant, 1, num_tx_ant, num_paths, 1]
            delays: [batch_size, 1, 1, num_paths]
            signal_length: 信号长度
        Returns:
            torch.Tensor: [batch_size, num_rx_ant, num_tx_ant, signal_length]
        """
        batch_size, _, num_rx_ant, _, num_tx_ant, num_paths, _ = h_time.shape
        h_coeff = h_time[:, 0, :, 0, :, :, 0]  # [batch_size, num_rx_ant, num_tx_ant, num_paths]
        delay_samples = (delays[:, 0, 0, :] * self.sampling_rate).round().long()  # [batch_size, num_paths]
        delay_samples = torch.clamp(delay_samples, 0, signal_length - 1)
        h_impulse = torch.zeros(
            (batch_size, num_rx_ant, num_tx_ant, signal_length),
            dtype=torch.complex64,
            device=self.device
        )
        # 矢量化放置每个路径的响应
        # batch_idx: [batch_size, num_paths]
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, num_paths)
        path_idx = torch.arange(num_paths, device=self.device).unsqueeze(0).expand(batch_size, -1)
        # h_coeff: [batch_size, num_rx_ant, num_tx_ant, num_paths]
        # delay_samples: [batch_size, num_paths]
        # 需要将每个路径的响应放到对应的delay位置
        # 先reshape为 [batch_size * num_paths, num_rx_ant, num_tx_ant]
        h_coeff_flat = h_coeff.permute(0, 3, 1, 2).reshape(-1, num_rx_ant, num_tx_ant)  # [batch_size*num_paths, num_rx_ant, num_tx_ant]
        delay_flat = delay_samples.reshape(-1)  # [batch_size*num_paths]
        batch_flat = batch_idx.reshape(-1)  # [batch_size*num_paths]
        # 构造索引
        for i in range(h_coeff_flat.shape[0]):
            h_impulse[batch_flat[i], :, :, delay_flat[i]] += h_coeff_flat[i]
        return h_impulse
    
    def _cir_to_ofdm_channel_pytorch_fft(
        self,
        h_impulse: torch.Tensor,
        ifft_size: int
    ) -> torch.Tensor:
        """
        并行化: 支持batch输入的PyTorch FFT频域信道响应
        Args:
            h_impulse: [batch_size, num_rx_ant, num_tx_ant, signal_length]
            mapping_indices: 子载波映射索引
            ifft_size: IFFT大小
        Returns:
            torch.Tensor: [batch_size, num_rx_ant, num_tx_ant, num_subcarriers]
        """
        batch_size, num_rx_ant, num_tx_ant, signal_length = h_impulse.shape
        # 零填充或截断到ifft_size
        if signal_length < ifft_size:
            h_padded = torch.zeros(
                (batch_size, num_rx_ant, num_tx_ant, ifft_size),
                dtype=h_impulse.dtype,
                device=h_impulse.device
            )
            h_padded[..., :signal_length] = h_impulse
        else:
            h_padded = h_impulse[..., :ifft_size]
        # FFT并行
        h_freq = torch.fft.fft(h_padded, dim=-1)  # [batch_size, num_rx_ant, num_tx_ant, ifft_size]
        # h_freq_mapped = h_freq_full[..., mapping_indices]  # [batch_size, num_rx_ant, num_tx_ant, num_subcarriers]
        return h_freq
    
    def _apply_timing_offset_to_freq_channel(
        self,
        h_freq: torch.Tensor,
        timing_offset_samples: torch.Tensor,
        ifft_size: int
    ) -> torch.Tensor:

        batch_size = h_freq.shape[0]
        if torch.all(timing_offset_samples == 0):
            return h_freq
        mapping_indices = torch.arange(ifft_size, device=h_freq.device)
        # 计算每个样本的相位旋转
        phase_rotation = torch.stack([
            torch.exp(-2.0j * np.pi * mapping_indices.float() * timing_offset_samples[b] / ifft_size)
            for b in range(batch_size)
        ], dim=0)  # [batch_size, num_subcarriers]
        # 应用相位旋转
        h_freq_with_offset = h_freq * phase_rotation[:, None, None, :]
        return h_freq_with_offset
    
    def apply_channel_to_signals(
        self, 
        signals_dict: Dict[int, torch.Tensor], 
        user_port_mapping: Dict[int, List[int]],
        ifft_size: int = 128,
        snr_db: float = 30.0,
        timing_offset_samples: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, any]]:
        """
        应用信道到信号 - 物理正确的实现
        
        实现完整的物理过程：
        1. 时域信号过信道（卷积）
        2. 应用timing offset（循环移位）
        3. 功率归一化和信噪比调整
        4. 接收机累加
        5. 加噪声
        6. 去CP并转频域
        
        Args:
            signals_dict: {user_id: signal} 时域信号字典，signal形状 [num_ports, signal_length]
            user_port_mapping: {user_id: [port_indices]} 用户端口映射
            ifft_size: IFFT大小
            snr_db: 信噪比 (dB)
            timing_offset_samples: timing offset（采样点）
            
        Returns:
            Tuple[torch.Tensor, Dict]: (接收频域信号, 信道信息字典)
            - 接收信号: [num_rx_ant, num_total_ports, num_subcarriers] (频域)
            - 信道信息: {'h_freq': 总频域信道, 'timing_offsets': timing偏移, 'noise_var': 噪声方差}
        """
        
        # 1. 初始化参数
        device = self.device
        batch_size = 1
        cp_length = self.system_config.cp_length_samples  # 使用系统配置的CP长度
        num_subcarriers = ifft_size
        
        # 计算总端口数（用于最终输出维度）
        total_ports = sum(len(ports) for ports in user_port_mapping.values())
        
        # 2. 为每个用户生成信道和应用信道
        all_received_signals = []  # 存储所有接收信号
        all_h_freq = {}  # 存储所有频域信道 {user_id: h_freq_tensor}
        timing_offsets = {}  # 存储timing offset
        
        noise_var = 1

        for user_id, signal in signals_dict.items():
            user_ports = user_port_mapping[user_id]
            num_ports = len(user_ports)
            
            
            # 2.1 确保信号是正确的形状 [num_ports, signal_length]
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)  # [1, signal_length]
            
            signal_length = signal.shape[1]
            
            # 2.2 生成该用户的信道
            try:
                # Use TensorFlow device corresponding to PyTorch device setting
                tf_device = '/CPU:0' if self.device == 'cpu' else '/GPU:0'
                with tf.device(tf_device):
                    # Create user-specific TDL
                    model_letter = self.model_type.split("-")[1]
                    user_channel = TDL(
                        model=model_letter,
                        delay_spread=self.delay_spread,
                        carrier_frequency=self.carrier_frequency,
                        num_rx_ant=self.num_rx_antennas,
                        num_tx_ant=num_ports,
                        precision='single'
                    )
                    
                    # 单个时间快照的信道实现
                    cir = user_channel(batch_size=batch_size, num_time_steps=1, sampling_frequency=self.sampling_rate)
                    h_time_tf, delays_tf = cir  # h_time和delays
                    
                    # 转换为PyTorch
                    h_time = torch.from_numpy(h_time_tf.numpy()).to(device)
                    delays = torch.from_numpy(delays_tf.numpy()).to(device)
                    
                    del user_channel, cir, h_time_tf, delays_tf  # TensorFlow内存管理
                    
            except Exception as e:
                print(f"❌ 用户{user_id}信道生成失败: {e}")
                continue
            

            
            # 2.3 构建时域冲激响应（用于卷积）
            h_impulse = self._construct_time_domain_channel_pytorch(h_time, delays, signal_length)
            # h_impulse: [num_rx_ant, num_tx_ant, signal_length]
            
            # 2.4 时域卷积：信号过信道
            received_signal_user = self._time_domain_convolution(signal, h_impulse)
            # received_signal_user: [num_rx_ant, num_ports, signal_length]
            
            # 2.5 应用timing offset
            timing_offsets[user_id] = timing_offset_samples
            
            received_signal_user_to = torch.roll(received_signal_user, shifts=timing_offset_samples, dims=2)
            

            
            # 2.6 计算频域信道（用于NMSE比较）
            # 创建子载波映射（这里简化为连续映射）
            mapping_indices = torch.arange(num_subcarriers, device=device)
            
            # 使用PyTorch FFT计算频域信道
            h_freq_user = self._cir_to_ofdm_channel_pytorch_fft(h_impulse, mapping_indices, ifft_size)
            # h_freq_user: [num_rx_ant, num_tx_ant, num_subcarriers]
            
            # 在频域信道上应用timing offset（用于真实信道比较）
            h_freq_user_with_offset = self._apply_timing_offset_to_freq_channel(
                h_freq_user, timing_offset_samples, mapping_indices, ifft_size
            )
            
            snr_linear = 10 ** (snr_db / 10.0)
            signal_power_user = snr_linear * noise_var

            received_signal_user_to_adjPower = received_signal_user_to * math.sqrt(signal_power_user) /math.sqrt(num_ports)
            h_freq_user_with_offset_adjPower = h_freq_user_with_offset * math.sqrt(signal_power_user) /math.sqrt(num_ports)

            received_signal_user_rx = torch.sum(received_signal_user_to_adjPower, dim=1)

            all_received_signals.append(received_signal_user_rx)
            all_h_freq[user_id] = h_freq_user_with_offset_adjPower  # 用字典存储，键为user_id
        
        total_received = sum(all_received_signals)  # [num_rx_ant, total_ports, signal_length]
        
        
        # 5. 加噪声
        
        noise = math.sqrt(noise_var / 2) / math.sqrt(ifft_size) * (
            torch.randn_like(total_received) + 1j * torch.randn_like(total_received)
        )
        
        noisy_received = total_received + noise
        data_part = noisy_received[:, cp_length:cp_length + ifft_size]
        received_freq = torch.fft.fft(data_part, dim=-1)  # [num_rx_ant, total_ports, num_subcarriers]

        
        # 7. 组装频域信道矩阵 - 保持字典结构，方便用户级访问
        # all_h_freq现在是字典: {user_id: h_freq_tensor}
        # 每个h_freq_tensor的形状: [num_rx_ant, num_ports_for_user, num_subcarriers]

        
        # 8. 返回结果
        channel_info = {
            'h_freq': all_h_freq,
            'timing_offsets': timing_offsets,
            'snr_db': snr_db,
        }
        
        
        return received_freq, channel_info
    
    def _time_domain_convolution(self, signals: torch.Tensor, h_impulse_tensor: torch.Tensor) -> torch.Tensor:
        """
        并行时域卷积：信号通过信道 (全batch tensor化)
        Args:
            signals: [batch_size, num_ports, signal_length]
            h_impulse_tensor: [batch_size, num_rx_ant, num_tx_ant, signal_length]
        Returns:
            torch.Tensor: [batch_size, num_rx_ant, num_ports, signal_length]
        """
        batch_size, num_ports, signal_length = signals.shape
        _, num_rx_ant, num_tx_ant, _ = h_impulse_tensor.shape
        assert num_tx_ant == num_ports, f"发送天线数({num_tx_ant})与端口数({num_ports})不匹配"
        # FFT并行实现
        X = torch.fft.fft(signals, dim=-1)  # [batch_size, num_ports, signal_length]
        H = torch.fft.fft(h_impulse_tensor, dim=-1)  # [batch_size, num_rx_ant, num_tx_ant, signal_length]
        # 只取对角线（tx_ant==port）
        idx = torch.arange(num_ports, device=signals.device)
        H_diag = H[:, :, idx, :]  # [batch_size, num_rx_ant, num_ports, signal_length]
        X_expand = X.unsqueeze(1)  # [batch_size, 1, num_ports, signal_length]
        Y = X_expand * H_diag  # [batch_size, num_rx_ant, num_ports, signal_length]
        y = torch.fft.ifft(Y, dim=-1)
        return y


def check_sionna_installation() -> bool:
    """Check if SIONNA is properly installed and working"""
    return SIONNA_AVAILABLE


def install_sionna_guide():
    """Print SIONNA installation guide"""
    print("📦 SIONNA Installation Guide")
    print("=" * 40)
    print("SIONNA is the industry standard for 3GPP channel modeling.")
    print("")
    print("💻 Installation commands (Intel网络内使用代理):")
    print("python -m pip install --proxy http://child-prc.intel.com:913 sionna")
    print("python -m pip install --proxy http://child-prc.intel.com:913 tensorflow>=2.13.0")
    print("")
    print("💻 Installation commands (其他网络):")
    print("python -m pip install sionna")
    print("python -m pip install tensorflow>=2.13.0")
    print("")
    print("🔍 Verify installation:")
    print("python -c \"import sionna; print('SIONNA installed successfully!')\"")
    print("")
    print("📖 Documentation: https://nvlabs.github.io/sionna/")
    print("🏢 Developed by: NVIDIA & TU Wien")


def print_sionna_info():
    """Print information about SIONNA"""
    print("📡 SIONNA Channel Modeling")
    print("=" * 40)
    
    if SIONNA_AVAILABLE:
        print("SIONNA - Professional 3GPP TR 38.901 models")
        print("   Features:")
        print("   - Complete 3GPP compliance")
        print("   - TDL models (A, B, C, D, E)")
        print("   - GPU acceleration")
        print("   - Industry standard")
        print("   - End-to-end differentiable")
        print("")
        print("🎯 Recommendation: Use SIONNA for professional channel modeling")
    else:
        print("❌ SIONNA - Not installed")
        print("   Install with proxy (Intel网络): python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow")
        print("   Install without proxy: python -m pip install sionna tensorflow")
        print("")
        print("⚠️  Fallback: Using custom TDL implementation")
        print("   Custom TDL provides basic 3GPP-based channel modeling")
        print("")
        install_sionna_guide()


class SIONNAChannelGenerator:
    """
    SIONNA信道生成器工厂
    这个类负责创建合适的数据生成器+信道模型组合：
    1. 如果SIONNA可用，创建SRSDataGenerator + SIONNAChannelModel
    2. 如果SIONNA不可用，创建SRSDataGenerator + SimpleFallbackChannel
    3. 如果完全不使用信道，创建纯BaseSRSDataGenerator
    """
    @staticmethod
    def create_generator(
        srs_config,  # 用户SRS配置
        system_config: Optional[SystemConfig] = None,  # 系统配置
        use_sionna: bool = True,
        use_channel: bool = True,
        num_rx_antennas: Optional[int] = None,
        channel_model: Optional[Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"]] = None,
        delay_spread: Optional[float] = None,
        carrier_frequency: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        snr_range: Optional[Tuple[float, float]] = None,
        device: str = "cpu",  # Force CPU-only execution
        **kwargs
    ):
        # ...existing code...
        pass

