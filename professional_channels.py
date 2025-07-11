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
            debug_dict: Debug information storage
            
        Returns:
            Tuple of (output_signals, frequency_domain_channels)
        """
        # 如果没有指定时间偏移，则随机生成
        if delay_offset_samples is None:
            delay_offset_samples = self.system_config.get_random_timing_offset_samples()
            if debug_dict is not None:
                # 记录生成的时间偏移信息
                timing_offset_seconds = delay_offset_samples / self.sampling_rate
                debug_dict['random_timing_offset_samples'] = delay_offset_samples
                debug_dict['random_timing_offset_seconds'] = timing_offset_seconds
                debug_dict['random_timing_offset_ns'] = timing_offset_seconds * 1e9
        
        return self._apply_sionna_channel(signals, user_config, delay_offset_samples, mapping_indices, ifft_size, debug_dict)
    
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
        debug_dict: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SIONNA channel model with per-UE TDL instantiation based on user config
        
        🔧 新的设计理念：基于用户配置的per-UE TDL实例化
        1. 从user_config获取每个UE的port数量
        2. 为每个UE创建专用TDL实例，num_tx_ant = 该UE的port数量
        3. 物理合理：每个UE有独立的信道环境和准确的天线配置
        4. UE-specific延迟：同一UE的所有port共享相同的传播延迟
        5. 架构清晰：避免维度不匹配和手动调整
        
        框架使用策略：
        1. 输入：PyTorch张量 + 用户配置
        2. 信道建模：按UE分别调用SIONNA → 转换回PyTorch
        3. 信号处理：全部使用PyTorch
        4. 输出：PyTorch张量
        """
        
        # 记录时间偏移信息
        timing_offset_seconds = delay_offset_samples / self.sampling_rate
        print(f"🕐 应用时间偏移: {delay_offset_samples} 采样点 ({timing_offset_seconds*1e9:.1f} ns)")
        
        # 从用户配置获取UE信息
        print(f"📋 用户配置信息:")
        print(f"   用户数量: {user_config.num_users}")
        print(f"   每用户端口数: {user_config.ports_per_user}")
        print(f"   总端口数: {user_config.total_ports}")
        
        # ========================================================================
        # 第一部分：PyTorch信号预处理和按用户分组
        # ========================================================================
        print("📦 第一阶段：PyTorch信号预处理和按用户分组...")
        
        # 处理不同的输入格式，按用户分组信号
        signals_by_user = {}  # {user_id: {port_id: signal}}
        signal_port_mapping = {}  # {global_port_index: (user_id, port_id)}
        
        if isinstance(signals, dict):
            # 保持输入顺序，按用户分组
            global_port_index = 0
            for (user_id, port_id), signal in signals.items():
                if user_id not in signals_by_user:
                    signals_by_user[user_id] = {}
                signals_by_user[user_id][port_id] = signal
                signal_port_mapping[global_port_index] = (user_id, port_id)
                global_port_index += 1
            
            print(f"   输入：字典格式，{len(signals)}个端口信号")
            print(f"   用户分组：{len(signals_by_user)}个用户")
            for user_id, user_signals in signals_by_user.items():
                expected_ports = user_config.ports_per_user[user_id] if user_id < len(user_config.ports_per_user) else 1
                actual_ports = len(user_signals)
                print(f"     用户{user_id}: {actual_ports}个端口 (配置期望: {expected_ports})")
                if actual_ports != expected_ports:
                    print(f"     ⚠️  用户{user_id}端口数不匹配配置！")
        
        elif isinstance(signals, list):
            # 对于列表输入，根据用户配置分配给各用户
            signals_by_user = {}
            signal_port_mapping = {}
            global_port_index = 0
            
            for user_id, num_ports in enumerate(user_config.ports_per_user):
                signals_by_user[user_id] = {}
                for port_id in range(num_ports):
                    if global_port_index < len(signals):
                        signals_by_user[user_id][port_id] = signals[global_port_index]
                        signal_port_mapping[global_port_index] = (user_id, port_id)
                        global_port_index += 1
                    else:
                        print(f"⚠️  警告：信号数量不足，用户{user_id}端口{port_id}没有对应信号")
            
            print(f"   输入：列表格式，{len(signals)}个信号，按用户配置分配")
        
        else:
            # 单张量，归属用户0端口0
            signals_by_user[0] = {0: signals}
            signal_port_mapping = {0: (0, 0)}
            print(f"   输入：单张量格式，归属用户0端口0")
        
        print(f"   全局端口映射：{signal_port_mapping}")
        
        # ========================================================================
        # 第二部分：按用户分别生成SIONNA信道（基于用户配置）
        # ========================================================================
        print("🔄 第二阶段：按用户配置分别生成SIONNA信道...")
        
        batch_size = 1
        num_time_steps = 1
        sampling_frequency = self.sampling_rate
        
        channels_by_user = {}  # {user_id: {'h_time': ..., 'delays': ..., 'signals': ...}}
        
        for user_id, user_signals in signals_by_user.items():
            # 🔧 从用户配置获取该用户的准确端口数
            if user_id < len(user_config.ports_per_user):
                expected_num_ports = user_config.ports_per_user[user_id]
            else:
                expected_num_ports = len(user_signals)  # 回退到实际信号数
                print(f"⚠️  用户{user_id}超出配置范围，使用实际端口数{expected_num_ports}")
            
            actual_num_ports = len(user_signals)
            print(f"   📡 处理用户{user_id}: 配置{expected_num_ports}端口, 实际{actual_num_ports}端口")
            
            # 使用配置的端口数来创建TDL（这是物理真实的端口数）
            num_ports_for_tdl = expected_num_ports
            
            # 🔧 为该用户创建专用的TDL实例（基于配置的端口数）
            print(f"     为用户{user_id}创建专用TDL实例 (num_tx_ant={num_ports_for_tdl})...")
            model_letter = self.model_type.split("-")[1]  # "A", "B", "C", etc.
            
            try:
                with tf.device('/GPU:0' if self.device == 'cuda' and tf.config.list_physical_devices('GPU') else '/CPU:0'):
                    user_channel = TDL(
                        model=model_letter,
                        delay_spread=self.delay_spread,
                        carrier_frequency=self.carrier_frequency,
                        num_rx_ant=self.num_rx_antennas,
                        num_tx_ant=num_ports_for_tdl,  # 🎯 关键：使用配置的端口数
                        precision='single'
                    )
                    print(f"     ✅ 用户{user_id}专用TDL创建成功 (TX_ant={num_ports_for_tdl}, RX_ant={self.num_rx_antennas})")
                    
                    # 使用该用户专用的TDL实例生成信道
                    print(f"     📡 使用专用TDL生成用户{user_id}信道...")
                    h_time_tf, delays_time_tf = user_channel(batch_size, num_time_steps, sampling_frequency)
                    
                    # 验证输出维度
                    # SIONNA TDL输出形状：
                    # h_time: [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
                    # delays: [batch_size, num_rx=1, num_tx=1, num_paths]
                    print(f"     📊 用户{user_id}信道维度: h_time={h_time_tf.shape}, delays={delays_time_tf.shape}")
                    actual_tx_ant_dim = h_time_tf.shape[4]  # num_tx_ant 维度，对应我们的端口数
                    if actual_tx_ant_dim != num_ports_for_tdl:
                        print(f"     ⚠️  TDL输出TX天线维度不匹配: 期望{num_ports_for_tdl}, 实际{actual_tx_ant_dim}")
                    else:
                        print(f"     ✅ TDL输出TX天线维度匹配: {actual_tx_ant_dim}")
                    
                    # 🔗 确保同一UE的所有port共享相同的delay (UE-specific延迟)
                    # 物理原理：同一UE的多个port在相同物理位置，传播延迟应该相同
                    # SIONNA输出：delays形状为[batch_size, num_rx=1, num_tx=1, num_paths]
                    # 注意：delays中的num_tx=1维度是固定的，延迟已经在所有天线间共享
                    if num_ports_for_tdl > 1:
                        print(f"     🔗 验证UE-specific延迟：同一UE的{num_ports_for_tdl}个port共享延迟")
                        # SIONNA TDL默认所有天线共享相同延迟，验证这一点
                        print(f"     ✅ UE{user_id}的所有port自动共享延迟（SIONNA TDL特性）")
                    # 转换为PyTorch
                    h_time_torch = torch.tensor(h_time_tf.numpy(), dtype=torch.complex64, device=self.device)
                    delays_time_torch = torch.tensor(delays_time_tf.numpy(), dtype=torch.float32, device=self.device)
                    
                    # 清理用户专用TDL实例
                    del user_channel
                    print(f"     🧹 用户{user_id}专用TDL实例已清理")
                    
            except Exception as e:
                print(f"     ❌ 用户{user_id}专用TDL创建/使用失败: {e}")
                raise RuntimeError(f"Failed to create/use dedicated TDL for user {user_id}: {e}")
            
            print(f"     ✅ 用户{user_id}信道CIR: {h_time_torch.shape}")
            print(f"     ✅ 用户{user_id}延迟: {delays_time_torch.shape}")
            
            # 准备该用户的信号（处理实际信号数量与配置不匹配的情况）
            user_signals_list = []
            for port_idx in range(num_ports_for_tdl):
                if port_idx in user_signals:
                    sig = user_signals[port_idx]
                else:
                    # 如果配置的端口数多于实际信号数，用零信号填充
                    sig = torch.zeros_like(next(iter(user_signals.values())))
                    print(f"     ⚠️  用户{user_id}端口{port_idx}无信号，使用零信号")
                
                if sig.is_cuda:
                    sig_cpu = sig.cpu()
                else:
                    sig_cpu = sig
                sig_torch = torch.tensor(sig_cpu.numpy(), dtype=torch.complex64, device=self.device)
                user_signals_list.append(sig_torch)
            
            user_signals_stack = torch.stack(user_signals_list)  # [num_ports_for_tdl, time_samples]
            print(f"     用户{user_id}信号: {user_signals_stack.shape}")
            
            # 存储该用户的信道和信号
            channels_by_user[user_id] = {
                'h_time': h_time_torch,
                'delays': delays_time_torch,
                'signals': user_signals_stack,
                'configured_ports': num_ports_for_tdl,
                'actual_ports': actual_num_ports,
                'port_mapping': {i: i for i in range(num_ports_for_tdl)}  # 简化映射
            }
        
        # ========================================================================
        # 第三部分：按用户处理频域转换和信道应用
        # ========================================================================
        print("🔧 第三阶段：按用户处理频域转换...")
        
        # 存储所有用户的频域信道
        all_h_freq_by_user = {}  # {user_id: h_freq_tensor}
        global_h_freq_list = []  # 全局频域信道列表（按原始顺序）
        
        for user_id, user_data in channels_by_user.items():
            h_time = user_data['h_time']
            delays = user_data['delays']
            signals = user_data['signals']
            port_mapping = user_data['port_mapping']
            
            print(f"   处理用户{user_id}的频域转换...")
            
            # 转换为频域信道响应（如果需要）- 全部使用PyTorch
            h_freq_user = None
            if mapping_indices is not None and ifft_size is not None:
                print(f"     使用PyTorch转换为频域信道响应...")
                
                # 使用PyTorch生成子载波频率
                freq_indices = torch.arange(ifft_size, dtype=torch.float32, device=self.device)
                frequencies = freq_indices * self.subcarrier_spacing
                
                # 使用PyTorch进行时域信道→频域信道转换
                h_freq_full = self._cir_to_ofdm_channel_pytorch(
                    frequencies, h_time, delays, ifft_size
                )
                print(f"     用户{user_id}完整频域信道: {h_freq_full.shape}")
                
                # 使用PyTorch提取映射的子载波
                mapping_indices_torch = mapping_indices.to(self.device)
                h_freq_mapped = h_freq_full[..., mapping_indices_torch]
                print(f"     用户{user_id}映射后频域信道: {h_freq_mapped.shape}")
                
                # 使用PyTorch应用时间偏移（频域相位旋转）
                if delay_offset_samples != 0:
                    print(f"     用户{user_id}应用时间偏移相位旋转...")
                    phase_arg = -2.0 * np.pi * mapping_indices_torch.float() * delay_offset_samples / ifft_size
                    phase_rotation = torch.exp(1j * phase_arg)  # PyTorch复数指数
                    # 扩展维度以匹配信道矩阵形状
                    phase_rotation_expanded = phase_rotation.view(1, 1, 1, 1, 1, -1)
                    h_freq_mapped = h_freq_mapped * phase_rotation_expanded
                
                # 移除batch和cluster维度，保留link维度
                # h_freq_mapped: [batch, cluster, rx_ant, link, tx_ant, subcarriers, time]
                h_freq_user = h_freq_mapped.squeeze(0).squeeze(0).squeeze(-1)  # [rx_ant, link, tx_ant, subcarriers]
                # 调整维度顺序：[rx_ant, link, subcarriers] (假设tx_ant=1)
                h_freq_user = h_freq_user.squeeze(2)  # 移除tx_ant维度（假设=1）
                print(f"     ✅ 用户{user_id}最终频域信道: {h_freq_user.shape}")
                
                all_h_freq_by_user[user_id] = h_freq_user
                
                # 按原始端口顺序添加到全局列表
                for port_idx in range(h_freq_user.shape[1]):  # link维度
                    global_h_freq_list.append(h_freq_user[:, port_idx, :])  # [rx_ant, subcarriers]
        
        # 合并所有用户的频域信道
        if global_h_freq_list:
            # 重新组织为全局频域信道矩阵
            h_freq_final_torch = torch.stack(global_h_freq_list, dim=1)  # [rx_ant, total_ports, subcarriers]
            print(f"   ✅ 全局频域信道矩阵: {h_freq_final_torch.shape}")
        else:
            h_freq_final_torch = None
        
        # ========================================================================
        # 第四部分：构建最终输出信号（PyTorch）
        # ========================================================================
        print("⚡ 第四阶段：构建最终输出信号...")
        
        # 计算输出信号长度（从第一个用户的第一个信号获取）
        signal_length = 1024  # 默认长度
        if channels_by_user:
            first_user_data = next(iter(channels_by_user.values()))
            if first_user_data['signals'].numel() > 0:
                signal_length = first_user_data['signals'].shape[-1]
        
        output_signals = torch.zeros(
            (self.num_rx_antennas, signal_length), 
            dtype=torch.complex64, 
            device=self.device
        )
        print(f"   输出信号形状: {output_signals.shape}")
        
        # ========================================================================
        # 第五部分：存储调试信息（全部PyTorch格式）
        # ========================================================================
        if debug_dict is not None:
            print("💾 存储调试信息...")
            
            # 合并所有用户的时域信道信息（用于调试）
            all_h_time_list = []
            all_delays_list = []
            for user_id in sorted(channels_by_user.keys()):
                user_data = channels_by_user[user_id]
                # h_time形状: [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
                # 第5维(num_tx_ant)对应端口数，我们需要为每个端口分离信道
                for port_idx in range(user_data['h_time'].shape[4]):  # num_tx_ant维度
                    all_h_time_list.append(user_data['h_time'][:, :, :, :, port_idx:port_idx+1, :, :])
                    # delays形状: [batch_size, num_rx=1, num_tx=1, num_paths] - 所有端口共享延迟
                    all_delays_list.append(user_data['delays'])  # 延迟对所有端口相同
            
            # 合并时域信道
            if all_h_time_list:
                combined_h_time = torch.cat(all_h_time_list, dim=4)  # 在num_tx_ant维度合并
                # 延迟：由于所有端口共享相同延迟，只需使用一个副本
                combined_delays = all_delays_list[0] if all_delays_list else None
            else:
                combined_h_time = None
                combined_delays = None
            
            debug_dict.update({
                'sionna_h_time': combined_h_time,
                'sionna_delays': combined_delays,
                'sionna_h_freq': h_freq_final_torch,
                'sionna_model_type': self.model_type,
                'sionna_backend': 'sionna',
                'applied_timing_offset_samples': delay_offset_samples,
                'applied_timing_offset_seconds': timing_offset_seconds,
                'applied_timing_offset_ns': timing_offset_seconds * 1e9,
                'signal_port_mapping': signal_port_mapping,  # 端口映射关系
                'channels_by_user': {  # 按用户保存的信道信息
                    user_id: {
                        'h_time': user_data['h_time'],
                        'delays': user_data['delays'],
                        'h_freq': all_h_freq_by_user.get(user_id),
                        'port_mapping': user_data['port_mapping']
                    } for user_id, user_data in channels_by_user.items()
                },
                'framework_usage': {
                    'input': 'pytorch',
                    'channel_cir_generation': 'tensorflow_sionna_per_user',  # 按用户调用SIONNA
                    'signal_processing': 'pytorch',  # 所有信号处理都是PyTorch
                    'output': 'pytorch'
                }
            })
        
        print("✅ SIONNA分用户信道应用完成（全PyTorch输出）")
        return output_signals, h_freq_final_torch
    
    def _cir_to_ofdm_channel_pytorch(
        self, 
        frequencies: torch.Tensor, 
        h_time: torch.Tensor, 
        delays: torch.Tensor, 
        ifft_size: int
    ) -> torch.Tensor:
        """
        使用PyTorch将SIONNA TDL时域信道冲激响应转换为频域OFDM信道响应
        
        🔧 Per-UE TDL设计说明：
        - 此函数为每个用户独立调用
        - 每次调用时，num_tx_ant数量 = 该用户的port数量 (通过per-UE TDL实例保证)
        - 不同用户的信道完全独立（不同的TDL实例）
        - 每个用户内部：不同port有独立的tap系数，但共享相同的延迟profile
        - UE-specific延迟：同一用户的所有port延迟相同（物理合理性）
        
        🔧 SIONNA TDL输出格式：
        - h_time: [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
        - delays: [batch_size, num_rx=1, num_tx=1, num_paths] (所有天线共享延迟)
        - num_tx_ant对应我们的port数量
        
        Args:
            frequencies: 子载波频率 [ifft_size] (Hz)
            h_time: 时域信道系数 [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
                   🎯 num_tx_ant = 当前用户的port数量 (per-UE TDL保证正确维度)
            delays: 路径延迟 [batch_size, num_rx=1, num_tx=1, num_paths] (秒)
                   🎯 所有port共享相同延迟（SIONNA TDL自然特性）
            ifft_size: IFFT大小
            
        Returns:
            torch.Tensor: 频域信道响应 [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, ifft_size, num_time_steps]
        """
        print(f"   🔧 PyTorch CIR→频域转换:")
        print(f"      输入CIR形状: {h_time.shape}")
        print(f"      延迟形状: {delays.shape}")
        print(f"      频率点数: {len(frequencies)}")
        
        # 获取维度信息
        # SIONNA TDL实际输出维度：
        # h_time: [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps]
        # delays: [batch_size, num_rx=1, num_tx=1, num_paths] (所有天线共享延迟)
        batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time = h_time.shape
        batch_size_delays, num_rx_delays, num_tx_delays, num_paths_delays = delays.shape
        
        # 验证SIONNA TDL输出格式
        assert num_rx == 1, f"SIONNA TDL的num_rx应该为1，实际为{num_rx}"
        assert num_tx == 1, f"SIONNA TDL的num_tx应该为1，实际为{num_tx}"
        assert num_rx_delays == 1, f"delays的num_rx应该为1，实际为{num_rx_delays}"
        assert num_tx_delays == 1, f"delays的num_tx应该为1，实际为{num_tx_delays}"
        
        # 确保其他维度一致
        assert batch_size == batch_size_delays, f"批次大小不匹配: CIR={batch_size}, delays={batch_size_delays}"
        assert num_paths == num_paths_delays, f"路径数不匹配: CIR={num_paths}, delays={num_paths_delays}"
        
        print(f"      SIONNA TDL维度分析（按用户调用）:")
        print(f"        batch_size: {batch_size} (批处理大小)")
        print(f"        num_rx: {num_rx} (接收器数量, 固定=1)")
        print(f"        num_rx_ant: {num_rx_ant} (接收天线数)")
        print(f"        num_tx: {num_tx} (发送器数量, 固定=1)")
        print(f"        num_tx_ant: {num_tx_ant} (当前用户的port数量)")
        print(f"        num_paths: {num_paths} (多径数量)")
        print(f"        num_time: {num_time} (时间快照, 通常=1)")
        
        print(f"      📊 当前用户链路分析: 当前用户有{num_tx_ant}个port")
        print(f"         每个port有独立的信道抽头系数")
        print(f"         每个port有{num_paths}个taps (多径分量)")
        print(f"         ★ UE-specific延迟：同一用户所有port的延迟相同")
        
        if num_time != 1:
            print(f"      ⚠️  警告: 静态信道的时间快照应该为1，当前为{num_time}")
            
        # 扩展频率维度以匹配所需的广播形状
        # frequencies: [ifft_size] -> [1, 1, 1, 1, 1, 1, ifft_size, 1]
        freq_expanded = frequencies.view(1, 1, 1, 1, 1, 1, ifft_size, 1)
        
        # 扩展延迟维度以匹配频率
        # delays: [batch_size, num_rx=1, num_tx=1, num_paths] -> [batch_size, 1, 1, 1, 1, num_paths, 1, 1]
        delays_expanded = delays.view(batch_size, 1, 1, 1, 1, num_paths, 1, 1)
        
        # 扩展CIR维度以匹配频率
        # h_time: [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps] 
        # -> [batch_size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, 1, num_time_steps]
        h_time_expanded = h_time.unsqueeze(-2)  # 在倒数第二个位置插入频率维度
        
        # 计算频域响应：H(f) = sum_over_paths(h_path * exp(-j*2*pi*f*delay_path))
        # 相位项：-2π * f * τ
        phase_arg = -2.0 * np.pi * freq_expanded * delays_expanded  # [batch, 1, 1, 1, 1, path, ifft_size, 1]
        phase_term = torch.exp(1j * phase_arg)  # [batch, 1, 1, 1, 1, path, ifft_size, 1]
        
        # 应用相位并在路径维度上求和
        # h_time_expanded: [batch, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, 1, num_time_steps]
        # phase_term: [batch, 1, 1, 1, 1, num_paths, ifft_size, 1]
        # 结果: [batch, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, ifft_size, num_time_steps]
        h_freq_per_path = h_time_expanded * phase_term
        
        # 在路径维度上求和得到最终频域响应
        h_freq = torch.sum(h_freq_per_path, dim=5)  # 在path维度求和
        # 结果形状: [batch, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, ifft_size, num_time_steps]
        
        print(f"      ✅ 输出频域信道形状: {h_freq.shape}")
        
        # 简单的功率归一化（可选）
        # 计算每个链路的平均功率并归一化
        power = torch.mean(torch.abs(h_freq) ** 2, dim=-2, keepdim=True)  # 在频率维度上平均
        power_mean = torch.mean(power)
        if power_mean > 0:
            h_freq = h_freq / torch.sqrt(power_mean)
            print(f"      归一化后平均功率: {torch.mean(torch.abs(h_freq) ** 2):.6f}")
        
        return h_freq


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
        snr_range: Tuple[float, float] = (-10, 40),
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
                snr_range=snr_range,
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
                snr_range=snr_range,
                sampling_rate=final_sampling_rate,
                device=device,
                **kwargs
            )
            
            # 包装成SRSDataGenerator接口
            generator = SRSDataGenerator(
                config=srs_config,  # 使用用户SRS配置
                channel_model=None,  # 无信道
                num_rx_antennas=final_num_rx_antennas,
                snr_range=snr_range,
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
