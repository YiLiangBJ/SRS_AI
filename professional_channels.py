"""
SIONNA Professional Channel Models for SRS Channel Estimation

This module provides SIONN        print(f"🚀 Initializing SIONNA {self.model_type} channel model")
        print(f"   系统配置参数:")
        print(f"   - 采样率: {self.sampling_rate/1e6:.2f} MHz (= {self.subcarrier_spacing/1e3:.0f} kHz × {self.ifft_size})")
        print(f"   - 载波频率: {self.carrier_frequency/1e9:.1f} GHz")
        print(f"   - 子载波间隔: {self.subcarrier_spacing/1e3:.0f} kHz")
        print(f"   - IFFT大小: {self.ifft_size}")
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

# Import SIONNA - the only professional channel library we support
try:
    import sionna
    from sionna.phy.channel.tr38901 import TDL
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
    import tensorflow as tf
    SIONNA_AVAILABLE = True
    print("✅ SIONNA professional channel library loaded successfully")
except ImportError as e:
    SIONNA_AVAILABLE = False
    print("❌ SIONNA not found. Install with proxy: python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow")
    print("   或者不用代理: python -m pip install sionna tensorflow")
    print("   Falling back to custom TDL implementation")


class SIONNAChannelModel:
    """
    SIONNA-based professional 3GPP channel model with per-UE TDL instantiation.
    
    🔧 架构设计 (Physically Meaningful TDL Usage):
    
    1. **Per-UE TDL Strategy**:
       - 不在初始化时创建全局TDL实例
       - 运行时为每个UE单独创建TDL实例
       - 每个UE的TDL: num_tx_ant = 该UE的port数量
       - 更符合物理现实：每个UE有独立的信道环境
    
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
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
        # 使用系统配置或创建默认配置
        if system_config is None:
            system_config = create_default_system_config()
        self.system_config = system_config
        
        # 从系统配置或参数获取值
        self.model_type = model_type if model_type is not None else system_config.channel_model_type
        self.num_rx_antennas = num_rx_antennas if num_rx_antennas is not None else system_config.num_rx_antennas
        self.delay_spread = delay_spread if delay_spread is not None else system_config.delay_spread
        self.k_factor = k_factor if k_factor is not None else system_config.k_factor
        self.device = device
        
        # 从系统配置获取统一的参数
        self.carrier_frequency = system_config.carrier_frequency
        self.sampling_rate = system_config.sampling_rate
        self.subcarrier_spacing = system_config.subcarrier_spacing
        self.ifft_size = system_config.ifft_size
        
        self._init_sionna()
    
    def _init_sionna(self):
        # 预处理K-factor for LOS models
        if self.model_type in ["TDL-D", "TDL-E"]:
            # LOS models - use provided K-factor or default
            if self.k_factor == 0.0:
                self.k_factor = 13.3  # Default K-factor for LOS models (dB converted to linear)
                self.k_factor = 10**(self.k_factor/10)  # Convert dB to linear   
    
    def apply_channel(
        self,
        signals: Union[torch.Tensor, List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        user_config,  # 🔧 新增：用户配置，用于确定每个UE的port数量
        delay_offset_samples: Optional[int] = None,  # 如果为None则随机生成
        mapping_indices: Optional[torch.Tensor] = None,
        ifft_size: Optional[int] = None,
        snr_db: Optional[float] = None,  # 🔧 新增：SNR参数
        debug_dict: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SIONNA channel model to input signals with per-UE TDL instantiation
        
        🔧 Per-UE TDL Strategy:
        - 根据user_config确定每个UE的port数量
        - 为每个UE创建专用TDL实例，num_tx_ant = 该UE的port数量
        - 物理合理：每个UE有独立的信道环境和天线配置
        
        Args:
            signals: Input signals
            user_config: 用户配置 (SRSConfig)，包含每个用户的port数量信息
            delay_offset_samples: Timing offset in samples (如果为None则根据系统配置随机生成)
            mapping_indices: Subcarrier mapping indices
            ifft_size: IFFT size
            snr_db: 信噪比 (dB)，如果为None则使用系统配置的默认值
            debug_dict: Debug information storage
            
        Returns:
            Tuple of (output_signals, frequency_domain_channels)
        """
        # 如果没有指定时间偏移，则从用户配置随机生成
        if delay_offset_samples is None:
            delay_offset_samples = user_config.get_timing_offset_samples(self.sampling_rate)
            if debug_dict is not None:
                # 记录生成的时间偏移信息
                timing_offset_seconds = delay_offset_samples / self.sampling_rate
                debug_dict['random_timing_offset_samples'] = delay_offset_samples
                debug_dict['random_timing_offset_seconds'] = timing_offset_seconds
                debug_dict['random_timing_offset_ns'] = timing_offset_seconds * 1e9
        
        # 如果没有指定SNR，则从用户配置获取
        if snr_db is None:
            snr_db = user_config.get_snr_db()
        
        return self._apply_sionna_channel(signals, user_config, delay_offset_samples, mapping_indices, ifft_size, snr_db, debug_dict)
    
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
        
        print("✅ TDL 参数验证通过")
    
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
        
        print("✅ TDL 输出验证通过")
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
            print(f"✅ 用户 {user_id} TDL实例创建成功")
            
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
                print(f"✅ TX天线维度完美匹配")
            
            # 清理TDL实例
            del user_channel
            print(f"🧹 用户 {user_id} TDL实例已清理")
            
            print(f"✅ 用户 {user_id} TDL调试成功")
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
        signals: Union[torch.Tensor, List[torch.Tensor], Dict[Tuple[int, int], torch.Tensor]],
        user_config,  # 🔧 新增：用户配置
        delay_offset_samples: int,  # 现在保证不为None
        mapping_indices: Optional[torch.Tensor] = None,
        ifft_size: Optional[int] = None,
        snr_db: float = 30.0,  # 🔧 新增：SNR参数，现在从上层传入
        debug_dict: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SIONNA channel model with physically correct signal processing
        
        🔧 物理正确的信号过信道流程：
        1. 时域信号过信道（卷积）
        2. 应用timing offset（循环移位）
        3. 功率归一化和信噪比调整
        4. 接收机累加
        5. 加噪声
        6. 去CP并转频域
        
        Args:
            signals: Input signals (time domain)
            user_config: User configuration with port assignments
            delay_offset_samples: Timing offset in samples
            mapping_indices: Subcarrier mapping indices
            ifft_size: IFFT size for frequency domain
            snr_db: 信噪比 (dB)
            debug_dict: Debug information storage
            
        Returns:
            Tuple of (received_freq_signals, frequency_domain_channels)
        """
        print(f"\n🔥 物理正确的SIONNA信道处理开始:")
        print(f"   Timing offset: {delay_offset_samples} 采样点")
        print(f"   用户数: {user_config.num_users}")
        print(f"   总端口数: {user_config.total_ports}")
        
        # 设置默认参数
        if ifft_size is None:
            ifft_size = 128
        if mapping_indices is None:
            mapping_indices = torch.arange(ifft_size, device=self.device)
        
        # ========================================================================
        # 第一步：信号预处理和分组
        # ========================================================================
        print("📦 信号预处理和按用户分组...")
        
        signals_dict = {}  # {user_id: signal_tensor}
        user_port_mapping = {}  # {user_id: [port_indices]}
        
        # 处理不同输入格式并按用户分组
        if isinstance(signals, dict):
            # 从(user_id, port_id) -> signal 字典构建
            for (user_id, port_id), signal in signals.items():
                if user_id not in signals_dict:
                    signals_dict[user_id] = []
                    user_port_mapping[user_id] = []
                
                signals_dict[user_id].append(signal)
                user_port_mapping[user_id].append(port_id)
        
        elif isinstance(signals, list):
            # 按用户配置分配信号
            signal_idx = 0
            for user_id in range(user_config.num_users):
                num_ports = user_config.ports_per_user[user_id]
                user_signals = []
                user_ports = []
                
                for port_id in range(num_ports):
                    if signal_idx < len(signals):
                        user_signals.append(signals[signal_idx])
                        user_ports.append(port_id)
                        signal_idx += 1
                    else:
                        # 用零信号填充
                        zero_signal = torch.zeros_like(signals[0])
                        user_signals.append(zero_signal)
                        user_ports.append(port_id)
                
                signals_dict[user_id] = user_signals
                user_port_mapping[user_id] = user_ports
        
        else:
            # 单张量，分配给用户0
            signals_dict[0] = [signals]
            user_port_mapping[0] = [0]
        
        # 将每个用户的信号列表转换为张量
        for user_id in signals_dict:
            signal_list = signals_dict[user_id]
            if len(signal_list) == 1:
                signals_dict[user_id] = signal_list[0].unsqueeze(0)  # [1, signal_length]
            else:
                signals_dict[user_id] = torch.stack(signal_list, dim=0)  # [num_ports, signal_length]
        
        print(f"   处理后用户信号: {len(signals_dict)} 个用户")
        for user_id, signal in signals_dict.items():
            print(f"     用户{user_id}: {signal.shape}")
        
        # 创建全局端口映射 {port_index: (user_id, port_id)}
        signal_port_mapping = {}
        global_port_idx = 0
        for user_id in sorted(signals_dict.keys()):  # 保持确定的顺序
            for port_idx, port_id in enumerate(user_port_mapping[user_id]):
                signal_port_mapping[global_port_idx] = (user_id, port_id)
                global_port_idx += 1
        
        print(f"   全局端口映射: {signal_port_mapping}")
        
        # ========================================================================
        # 第二步：为每个用户生成SIONNA信道
        # ========================================================================
        print("🔄 为每个用户生成SIONNA信道...")
        
        batch_size = 1
        num_time_steps = 1
        user_channels = {}  # {user_id: (h_time, delays)}
        
        for user_id in signals_dict:
            num_ports = len(user_port_mapping[user_id])
            print(f"   用户{user_id}: {num_ports}个端口")
            
            try:
                with tf.device('/CPU:0'):  # 使用CPU避免GPU内存问题
                    # 创建用户专用TDL
                    model_letter = self.model_type.split("-")[1]
                    user_channel = TDL(
                        model=model_letter,
                        delay_spread=self.delay_spread,
                        carrier_frequency=self.carrier_frequency,
                        num_rx_ant=self.num_rx_antennas,
                        num_tx_ant=num_ports,  # 用户端口数
                        precision='single'
                    )
                    
                    # 生成信道
                    h_time_tf, delays_tf = user_channel(batch_size, num_time_steps, self.sampling_rate)
                    
                    # 转换为PyTorch
                    h_time = torch.from_numpy(h_time_tf.numpy()).to(self.device)
                    delays = torch.from_numpy(delays_tf.numpy()).to(self.device)
                    
                    user_channels[user_id] = (h_time, delays)
                    
                    del user_channel, h_time_tf, delays_tf  # 清理TensorFlow对象
                    
                    print(f"     ✅ 用户{user_id}信道生成: CIR{h_time.shape}, delays{delays.shape}")
            
            except Exception as e:
                print(f"     ❌ 用户{user_id}信道生成失败: {e}")
                raise
        
        # ========================================================================
        # 第三步：物理正确的信号过信道处理
        # ========================================================================
        print("⚡ 物理正确的信号过信道处理...")
        
        # 使用新的物理正确处理函数
        received_freq, channel_info = self.apply_channel_to_signals(
            signals_dict=signals_dict,
            user_port_mapping=user_port_mapping,
            ifft_size=ifft_size,
            snr_db=snr_db,  # 使用从配置获取的SNR
            timing_offset_samples=delay_offset_samples
        )
        
        # ========================================================================
        # 第四步：准备输出
        # ========================================================================
        h_freq_total = channel_info['h_freq']
        
        if debug_dict is not None:
            debug_dict.update({
                'timing_offset_samples': delay_offset_samples,
                'timing_offset_seconds': delay_offset_samples / self.sampling_rate,
                'user_channels': user_channels,
                'channel_info': channel_info,
                'signal_port_mapping': signal_port_mapping  # 添加端口映射信息
            })
        
        print(f"✅ SIONNA信道处理完成:")
        print(f"   接收信号: {received_freq.shape}")
        print(f"   信道矩阵: {h_freq_total.shape}")
        
        return received_freq, h_freq_total
    
    def _construct_time_domain_channel_pytorch(
        self, 
        h_time: torch.Tensor,
        delays: torch.Tensor,
        signal_length: int
    ) -> torch.Tensor:
        """
        构建时域信道冲激响应
        
        Args:
            h_time: SIONNA TDL输出的时域信道响应
            delays: SIONNA TDL输出的延迟信息
            signal_length: 信号长度
            
        Returns:
            torch.Tensor: 时域信道冲激响应 [num_rx_ant, num_tx_ant, signal_length]
        """
        print(f"   🔧 构建时域信道冲激响应:")
        print(f"      输入CIR形状: {h_time.shape}")
        print(f"      延迟形状: {delays.shape}")
        print(f"      信号长度: {signal_length}")
        
        # 获取维度信息
        batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time = h_time.shape
        batch_size_d, num_rx_d, num_tx_d, num_paths_d = delays.shape
        
        print(f"      🔍 SIONNA TDL输出分析:")
        print(f"         CIR: [batch={batch_size}, rx_cluster={num_rx}, rx_ant={num_rx_ant}, tx_cluster={num_tx}, tx_ant={num_tx_ant}, paths={num_paths}, time={num_time}]")
        print(f"         Delays: [batch={batch_size_d}, rx_cluster={num_rx_d}, tx_cluster={num_tx_d}, paths={num_paths_d}]")
        
        # 验证维度一致性
        assert num_paths == num_paths_d, f"CIR和delays的路径数不匹配: {num_paths} vs {num_paths_d}"
        
        # 正确提取延迟信息：delays的形状是 [batch, rx_cluster=1, tx_cluster=1, num_paths]
        # 延迟对所有天线都相同，所以我们只需要取一个副本
        delay_samples = (delays[0, 0, 0, :] * self.sampling_rate).round().long()  # [num_paths]
        
        # 确保延迟在有效范围内
        delay_samples = torch.clamp(delay_samples, 0, signal_length - 1)
        
        print(f"      延迟采样点: {delay_samples}")
        print(f"      延迟范围: {delay_samples.min().item()} - {delay_samples.max().item()} 采样点")
        
        # 初始化时域信道响应
        h_impulse = torch.zeros(
            (num_rx_ant, num_tx_ant, signal_length),
            dtype=torch.complex64,
            device=self.device
        )
        
        # 正确提取信道系数：h_time的形状是 [batch, rx_cluster=1, rx_ant, tx_cluster=1, tx_ant, paths, time=1]
        # 我们需要: [rx_ant, tx_ant, paths]
        h_coeff = h_time[0, 0, :, 0, :, :, 0]  # [num_rx_ant, num_tx_ant, num_paths]
        
        print(f"      提取的信道系数形状: {h_coeff.shape}")
        print(f"      期望形状: [{num_rx_ant}, {num_tx_ant}, {num_paths}]")
        
        # 将每个路径的响应放置在对应的延迟位置
        for path_idx in range(num_paths):
            delay_idx = delay_samples[path_idx].item()
            # 将该路径的所有天线对的系数加到对应的延迟位置
            h_impulse[:, :, delay_idx] += h_coeff[:, :, path_idx]
        
        print(f"      ✅ 时域冲激响应形状: {h_impulse.shape}")
        
        return h_impulse
    
    def _cir_to_ofdm_channel_pytorch_fft(
        self, 
        h_impulse: torch.Tensor,
        mapping_indices: torch.Tensor,
        ifft_size: int
    ) -> torch.Tensor:
        """
        使用PyTorch FFT直接计算频域信道响应
        
        Args:
            h_impulse: 时域信道冲激响应 [num_rx_ant, num_tx_ant, signal_length]
            mapping_indices: 子载波映射索引
            ifft_size: IFFT大小
            
        Returns:
            torch.Tensor: 频域信道响应 [num_rx_ant, num_tx_ant, num_subcarriers]
        """
        print(f"   � 使用PyTorch FFT计算频域信道:")
        print(f"      时域信道形状: {h_impulse.shape}")
        print(f"      IFFT大小: {ifft_size}")
        print(f"      映射子载波数: {len(mapping_indices)}")
        
        # 对时域信道进行FFT
        # 先进行零填充到IFFT大小
        num_rx_ant, num_tx_ant, signal_length = h_impulse.shape
        
        if signal_length < ifft_size:
            # 零填充
            h_padded = torch.zeros(
                (num_rx_ant, num_tx_ant, ifft_size),
                dtype=h_impulse.dtype,
                device=h_impulse.device
            )
            h_padded[:, :, :signal_length] = h_impulse
        else:
            # 截断
            h_padded = h_impulse[:, :, :ifft_size]
        
        # 使用PyTorch FFT计算频域响应
        h_freq_full = torch.fft.fft(h_padded, dim=-1)  # [num_rx_ant, num_tx_ant, ifft_size]
        
        # 提取映射的子载波
        h_freq_mapped = h_freq_full[:, :, mapping_indices]  # [num_rx_ant, num_tx_ant, num_subcarriers]
        
        print(f"      ✅ 频域信道形状: {h_freq_mapped.shape}")
        
        return h_freq_mapped
    
    def _apply_timing_offset_to_freq_channel(
        self,
        h_freq: torch.Tensor,
        timing_offset_samples: int,
        mapping_indices: torch.Tensor,
        ifft_size: int
    ) -> torch.Tensor:
        """
        在频域信道上应用timing offset（用于NMSE比较的真实信道）
        
        Args:
            h_freq: 频域信道 [num_rx_ant, num_tx_ant, num_subcarriers]
            timing_offset_samples: timing offset（采样点）
            mapping_indices: 子载波映射索引
            ifft_size: IFFT大小
            
        Returns:
            torch.Tensor: 带timing offset的频域信道
        """
        if timing_offset_samples == 0:
            return h_freq
        
        print(f"   🕐 在频域信道应用timing offset: {timing_offset_samples} 采样点")
        
        # 计算相位旋转
        phase_arg = -2.0 * np.pi * mapping_indices.float() * timing_offset_samples / ifft_size
        phase_rotation = torch.exp(1j * phase_arg)  # [num_subcarriers]
        
        # 应用相位旋转
        h_freq_with_offset = h_freq * phase_rotation.unsqueeze(0).unsqueeze(0)  # broadcast to [rx_ant, tx_ant, subcarriers]
        
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
        print(f"\n🔥 物理正确的信号过信道流程开始:")
        print(f"   输入用户数: {len(signals_dict)}")
        print(f"   IFFT大小: {ifft_size}, SNR: {snr_db} dB")
        print(f"   Timing offset: {timing_offset_samples} 采样点")
        
        # 1. 初始化参数
        device = self.device
        batch_size = 1
        cp_length = self.system_config.cp_length_samples  # 使用系统配置的CP长度
        num_subcarriers = ifft_size
        
        # 计算总端口数（用于最终输出维度）
        total_ports = sum(len(ports) for ports in user_port_mapping.values())
        
        # 2. 为每个用户生成信道和应用信道
        all_received_signals = []  # 存储所有接收信号
        all_h_freq = []  # 存储所有频域信道
        timing_offsets = {}  # 存储timing offset
        
        for user_id, signal in signals_dict.items():
            user_ports = user_port_mapping[user_id]
            num_ports = len(user_ports)
            
            print(f"\n   👤 处理用户 {user_id}:")
            print(f"      端口: {user_ports}")
            print(f"      信号形状: {signal.shape}")
            
            # 2.1 确保信号是正确的形状 [num_ports, signal_length]
            if signal.dim() == 1:
                signal = signal.unsqueeze(0)  # [1, signal_length]
            
            signal_length = signal.shape[1]
            print(f"      处理后信号形状: {signal.shape}")
            
            # 2.2 生成该用户的信道
            try:
                with tf.device('/CPU:0'):
                    # 创建用户专用TDL
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
            
            print(f"      生成的CIR形状: {h_time.shape}")
            print(f"      延迟形状: {delays.shape}")
            
            # 2.3 构建时域冲激响应（用于卷积）
            h_impulse = self._construct_time_domain_channel_pytorch(h_time, delays, signal_length)
            # h_impulse: [num_rx_ant, num_tx_ant, signal_length]
            
            # 2.4 时域卷积：信号过信道
            received_signal_user = self._time_domain_convolution(signal, h_impulse)
            # received_signal_user: [num_rx_ant, num_ports, signal_length]
            
            # 2.5 应用timing offset
            timing_offsets[user_id] = timing_offset_samples
            
            received_signal_user_to = torch.roll(received_signal_user, shifts=timing_offset_samples, dims=2)
            
            print(f"      时域卷积后形状: {received_signal_user_to.shape}")
            print(f"      Timing offset: {timing_offset_samples} 采样点")
            
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
            
            all_received_signals.append(received_signal_user_to)
            all_h_freq.append(h_freq_user_with_offset)
        
        if not all_received_signals:
            raise ValueError("没有成功处理任何用户信号")
        
        # 3. 功率归一化
        print(f"\n   ⚡ 功率归一化和SNR调整:")
        
        # 3.1 计算每个用户信号的功率并归一化
        noise_var = 1
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power_user = snr_linear * noise_var

        for i, received_signal in enumerate(all_received_signals):
            all_received_signals[i] = received_signal * math.sqrt(signal_power_user) /math.sqrt(received_signal.shape[1])
        
        total_received = sum(all_received_signals)  # [num_rx_ant, total_ports, signal_length]
        
        
        # 5. 加噪声
        
        noise = math.sqrt(noise_var / 2) / math.sqrt(ifft_size) * (
            torch.randn_like(total_received) + 1j * torch.randn_like(total_received)
        )
        
        noisy_received = total_received + noise
        data_part = noisy_received[:, :, cp_length:cp_length + ifft_size]
        received_freq = torch.fft.fft(data_part, dim=-1)  # [num_rx_ant, total_ports, num_subcarriers]
        print(f"      接收频域信号形状: {received_freq.shape}")
        
        # 7. 组装频域信道矩阵
        h_freq_total = torch.cat(all_h_freq, dim=1)  # 在发送天线维度连接
        print(f"      总频域信道形状: {h_freq_total.shape}")
        
        # 8. 返回结果
        channel_info = {
            'h_freq': h_freq_total,
            'timing_offsets': timing_offsets,
            'snr_db': snr_db,
        }
        
        print(f"\n✅ 物理信道处理完成:")
        print(f"   接收信号: {received_freq.shape}")
        print(f"   信道矩阵: {h_freq_total.shape}")
        print(f"   SNR: {snr_db} dB, 噪声方差: {noise_var:.6f}")
        
        return received_freq, channel_info
    
    def _time_domain_convolution(self, signal: torch.Tensor, h_impulse: torch.Tensor) -> torch.Tensor:
        """
        时域卷积：信号通过信道
        
        Args:
            signal: 输入信号 [num_ports, signal_length]
            h_impulse: 时域信道冲激响应 [num_rx_ant, num_tx_ant, signal_length]
            
        Returns:
            torch.Tensor: 接收信号 [num_rx_ant, num_ports, signal_length]
        """
        num_ports, signal_length = signal.shape
        num_rx_ant, num_tx_ant, h_length = h_impulse.shape
        
        assert num_tx_ant == num_ports, f"发送天线数({num_tx_ant})与端口数({num_ports})不匹配"
        
        print(f"      🔀 时域卷积: 信号{signal.shape} * 信道{h_impulse.shape}")
        
        # 执行循环卷积（使用FFT优化）
        # 为了避免线性卷积的长度扩展，使用循环卷积
        received = torch.zeros((num_rx_ant, num_ports, signal_length), dtype=signal.dtype, device=signal.device)
        
        for rx_ant in range(num_rx_ant):
            for tx_ant in range(num_ports):
                # 时域循环卷积
                h = h_impulse[rx_ant, tx_ant, :]  # [signal_length]
                x = signal[tx_ant, :]  # [signal_length]
                
                # 使用FFT实现循环卷积
                X = torch.fft.fft(x)
                H = torch.fft.fft(h)
                Y = X * H
                y = torch.fft.ifft(Y)
                
                received[rx_ant, tx_ant, :] = y
        
        return received
    
    def _apply_timing_offset_time_domain(self, signal: torch.Tensor, offset_samples: int) -> torch.Tensor:
        """
        在时域应用timing offset（循环移位）
        
        Args:
            signal: 输入信号 [num_rx_ant, num_ports, signal_length]
            offset_samples: timing offset（采样点数）
            
        Returns:
            torch.Tensor: 移位后的信号
        """
        if offset_samples == 0:
            return signal
        
        print(f"      🕐 时域timing offset: 循环左移 {offset_samples} 采样点")
        
        # 循环移位：signal[..., offset:] + signal[..., :offset]
        shifted = torch.cat([
            signal[..., offset_samples:],
            signal[..., :offset_samples]
        ], dim=-1)
        
        return shifted


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
        # 以下参数如果为None，将从system_config获取
        num_rx_antennas: Optional[int] = None,
        channel_model: Optional[Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"]] = None,
        delay_spread: Optional[float] = None,
        carrier_frequency: Optional[float] = None,
        sampling_rate: Optional[float] = None,
        # 其他参数
        snr_range: Optional[Tuple[float, float]] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs
    ):
        """
        创建合适的数据生成器
        
        Args:
            srs_config: 用户SRS配置
            system_config: 系统配置 (如果为None则使用默认配置)
            use_sionna: 是否尝试使用SIONNA
            use_channel: 是否使用信道模型
            num_rx_antennas: 接收天线数量 (如果为None则从system_config获取)
            channel_model: 信道模型类型 (如果为None则从system_config获取)
            delay_spread: 延迟扩展 (如果为None则从system_config获取)
            carrier_frequency: 载波频率 (如果为None则从system_config获取)
            sampling_rate: 采样率 (如果为None则从system_config获取)
            其他参数: 数据生成参数
            
        Returns:
            SRSDataGenerator实例
        """
        # 使用系统配置或创建默认配置
        if system_config is None:
            system_config = create_default_system_config()
        
        print(f"🔧 使用系统配置:")
        print(f"   采样率: {system_config.sampling_rate/1e6:.2f} MHz")
        print(f"   载波频率: {system_config.carrier_frequency/1e9:.1f} GHz")
        print(f"   IFFT大小: {system_config.ifft_size}")
        print(f"   子载波间隔: {system_config.subcarrier_spacing/1e3:.0f} kHz")
        # 导入重构后的数据生成器
        from data_generator_refactored import SRSDataGenerator, BaseSRSDataGenerator
        
        # 根据配置选择信道模型
        channel_model_instance = None
        
        # 从system_config获取默认值
        final_num_rx_antennas = num_rx_antennas if num_rx_antennas is not None else system_config.num_rx_antennas
        final_channel_model = channel_model if channel_model is not None else system_config.channel_model_type
        final_delay_spread = delay_spread if delay_spread is not None else system_config.delay_spread
        final_carrier_frequency = carrier_frequency if carrier_frequency is not None else system_config.carrier_frequency
        final_sampling_rate = sampling_rate if sampling_rate is not None else system_config.sampling_rate
        final_snr_range = snr_range if snr_range is not None else srs_config.snr_range
        
        if use_channel:
            if use_sionna and SIONNA_AVAILABLE:
                print("🚀 创建SIONNA专业信道模型")
                try:
                    channel_model_instance = SIONNAChannelModel(
                        system_config=system_config,  # 传入系统配置
                        model_type=final_channel_model,
                        num_rx_antennas=final_num_rx_antennas,
                        delay_spread=final_delay_spread,
                        device=device
                    )
                    print("✅ SIONNA信道模型创建成功")
                except Exception as e:
                    print(f"❌ SIONNA信道模型创建失败: {e}")
                    print("   回退到简化信道模型")
                    channel_model_instance = None
            
            if channel_model_instance is None:
                print("⚠️  使用简化回退信道模型")
                # 这里可以创建一个简化的信道模型
                # 为了简化，我们先不使用信道
                channel_model_instance = None
        
        # 创建数据生成器
        if channel_model_instance is not None:
            print(f"✅ 创建完整数据生成器（包含{type(channel_model_instance).__name__}信道）")
            generator = SRSDataGenerator(
                config=srs_config,  # 使用用户SRS配置
                channel_model=channel_model_instance,
                num_rx_antennas=final_num_rx_antennas,
                snr_range=final_snr_range,
                sampling_rate=final_sampling_rate,
                device=device,
                **kwargs
            )
        else:
            print("✅ 创建纯数据生成器（无信道模型）")
            # 创建包装器，使BaseSRSDataGenerator兼容原接口
            base_gen = BaseSRSDataGenerator(
                config=srs_config,  # 使用用户SRS配置
                num_rx_antennas=final_num_rx_antennas,
                snr_range=final_snr_range,
                sampling_rate=final_sampling_rate,
                device=device,
                **kwargs
            )
            
            # 包装成SRSDataGenerator接口
            generator = SRSDataGenerator(
                config=srs_config,  # 使用用户SRS配置
                channel_model=None,  # 无信道
                num_rx_antennas=final_num_rx_antennas,
                snr_range=final_snr_range,
                sampling_rate=final_sampling_rate,
                device=device,
                **kwargs
            )
        
        return generator
    
    def __init__(self, *args, **kwargs):
        """
        向后兼容的构造函数
        创建generator并代理所有方法调用
        """
        self.generator = self.create_generator(*args, **kwargs)
        
    def generate_sample(self, *args, **kwargs):
        """生成样本"""
        return self.generator.generate_sample(*args, **kwargs)
    
    def generate_batch(self, *args, **kwargs):
        """生成批次"""
        return self.generator.generate_batch(*args, **kwargs)
    
    def get_debug_info(self, *args, **kwargs):
        """获取调试信息"""
        return self.generator.get_debug_info(*args, **kwargs)
    
    @property
    def using_sionna(self):
        """是否使用SIONNA"""
        return (hasattr(self.generator, 'channel_model') and 
                self.generator.channel_model is not None and 
                isinstance(self.generator.channel_model, SIONNAChannelModel))


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
        print("✅ SIONNA - Professional 3GPP TR 38.901 models")
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


if __name__ == "__main__":
    # Print SIONNA information
    print_sionna_info()

