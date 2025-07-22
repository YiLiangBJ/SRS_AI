"""
系统级配置 - 所有物理层参数的统一配置

这个配置定义了系统的基础物理层参数，包括：
- OFDM参数 (子载波间隔、IFFT大小等)
- 射频参数 (载波频率、功率等)
- 信道参数 (天线数量、延迟扩展等)
- 采样率计算 (基于子载波间隔和IFFT大小)

所有模块都应该从这里获取系统参数，确保一致性。
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import math


@dataclass
class SystemConfig:
    """
    系统级物理层配置
    
    定义所有与物理层相关的系统参数，确保整个系统的一致性
    """
    
    # =============================================================================
    # OFDM 基础参数
    # =============================================================================
    subcarrier_spacing: float = 30e3  # 子载波间隔 (Hz) - 30 kHz for 5G NR
    ifft_size: int = 4096  # IFFT/FFT 大小 - 4096 points
    cp_length_ratio: float = 1.0  # 循环前缀长度比例 (相对于IFFT大小) - 1.0表示CP长度=IFFT大小
    
    # =============================================================================
    # 射频参数
    # =============================================================================
    carrier_frequency: float = 3.5e9  # 载波频率 (Hz) - 3.5 GHz for 5G NR
    
    # =============================================================================
    # 天线配置
    # =============================================================================
    num_rx_antennas: int = 4  # 默认接收天线数量
    max_tx_antennas_per_user: int = 4  # 每个用户最大发送天线数量
    
    # =============================================================================
    # 信道建模参数
    # =============================================================================
    channel_model_type: Literal["TDL-A", "TDL-B", "TDL-C", "TDL-D", "TDL-E"] = "TDL-A"
    delay_spread: float = 30e-9  # RMS延迟扩展 (秒) - 30 ns
    k_factor: float = 0.0  # Ricean K因子 (线性尺度，0表示NLOS)
    
    # =============================================================================
    # 计算属性
    # =============================================================================
    
    @property
    def sampling_rate(self) -> float:
        """
        计算采样率
        
        采样率 = 子载波间隔 × IFFT大小
        
        Returns:
            采样率 (Hz)
        """
        return self.subcarrier_spacing * self.ifft_size
    
    @property
    def symbol_duration(self) -> float:
        """
        计算OFDM符号持续时间 (不包括CP)
        
        符号持续时间 = 1 / 子载波间隔
        
        Returns:
            符号持续时间 (秒)
        """
        return 1.0 / self.subcarrier_spacing
    
    @property
    def cp_length_samples(self) -> int:
        """
        计算循环前缀长度 (采样点)
        
        CP长度 = IFFT大小 × CP长度比例
        
        Returns:
            循环前缀长度 (采样点)
        """
        return int(self.ifft_size * self.cp_length_ratio)
    
    @property
    def cp_duration(self) -> float:
        """
        计算循环前缀持续时间
        
        CP持续时间 = CP长度采样点 / 采样率
        
        Returns:
            循环前缀持续时间 (秒)
        """
        return self.cp_length_samples / self.sampling_rate
    
    @property
    def total_symbol_duration(self) -> float:
        """
        计算总符号持续时间 (包括CP)
        
        总符号持续时间 = 符号持续时间 + CP持续时间
        
        Returns:
            总符号持续时间 (秒)
        """
        return self.symbol_duration + self.cp_duration
    
    @property
    def total_symbol_samples(self) -> int:
        """
        计算总符号长度 (采样点，包括CP)
        
        总符号采样点 = IFFT大小 + CP长度
        
        Returns:
            总符号长度 (采样点)
        """
        return self.ifft_size + self.cp_length_samples
    
    @property
    def frequency_resolution(self) -> float:
        """
        计算频率分辨率
        
        频率分辨率 = 采样率 / IFFT大小 = 子载波间隔
        
        Returns:
            频率分辨率 (Hz)
        """
        return self.sampling_rate / self.ifft_size
    
    def validate_config(self) -> bool:
        """
        验证系统配置的合理性
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # 检查子载波间隔
        valid_scs = [15e3, 30e3, 60e3, 120e3, 240e3]  # 3GPP标准子载波间隔
        if self.subcarrier_spacing not in valid_scs:
            raise ValueError(f"子载波间隔 {self.subcarrier_spacing/1e3} kHz 不是标准值，支持: {[scs/1e3 for scs in valid_scs]} kHz")
        
        # 检查IFFT大小
        if not self._is_power_of_2(self.ifft_size):
            raise ValueError(f"IFFT大小 {self.ifft_size} 必须是2的幂")
        
        if self.ifft_size < 128 or self.ifft_size > 8192:
            raise ValueError(f"IFFT大小 {self.ifft_size} 超出合理范围 [128, 8192]")
        
        # 检查CP长度比例
        if self.cp_length_ratio <= 0:
            raise ValueError(f"CP长度比例 {self.cp_length_ratio} 必须为正数")
        
        # 检查载波频率
        if self.carrier_frequency <= 0:
            raise ValueError(f"载波频率 {self.carrier_frequency} 必须为正数")
        
        # 检查天线数量
        if self.num_rx_antennas <= 0:
            raise ValueError(f"接收天线数量 {self.num_rx_antennas} 必须为正整数")
        
        if self.max_tx_antennas_per_user <= 0:
            raise ValueError(f"每用户最大发送天线数量 {self.max_tx_antennas_per_user} 必须为正整数")
        
        # 检查延迟扩展
        if self.delay_spread <= 0:
            raise ValueError(f"延迟扩展 {self.delay_spread} 必须为正数")
        
        # 检查K因子
        if self.k_factor < 0:
            raise ValueError(f"K因子 {self.k_factor} 不能为负数")
        
        return True
    
    def _is_power_of_2(self, n: int) -> bool:
        """检查是否为2的幂"""
        return n > 0 and (n & (n - 1)) == 0
    
    def print_summary(self):
        """打印系统配置摘要"""
        print("🔧 系统配置摘要")
        print("=" * 50)
        
        print("📡 OFDM参数:")
        print(f"   子载波间隔: {self.subcarrier_spacing/1e3:.1f} kHz")
        print(f"   IFFT大小: {self.ifft_size}")
        print(f"   采样率: {self.sampling_rate/1e6:.2f} MHz")
        print(f"   符号持续时间: {self.symbol_duration*1e6:.1f} μs")
        print(f"   CP长度: {self.cp_length_samples} 采样点 ({self.cp_duration*1e6:.1f} μs)")
        print(f"   总符号长度: {self.total_symbol_samples} 采样点 ({self.total_symbol_duration*1e6:.1f} μs)")
        
        print(f"\n📻 射频参数:")
        print(f"   载波频率: {self.carrier_frequency/1e9:.1f} GHz")
        
        print(f"\n📶 天线配置:")
        print(f"   接收天线数量: {self.num_rx_antennas}")
        print(f"   每用户最大发送天线: {self.max_tx_antennas_per_user}")
        
        print(f"\n🌊 信道参数:")
        print(f"   信道模型: {self.channel_model_type}")
        print(f"   延迟扩展: {self.delay_spread*1e9:.1f} ns")
        print(f"   K因子: {self.k_factor:.1f} (线性)")
        
        print(f"\n✅ 配置验证: ", end="")
        try:
            self.validate_config()
            print("通过")
        except ValueError as e:
            print(f"失败 - {e}")
    

def create_default_system_config() -> SystemConfig:
    """
    Create default system configuration
    
    Uses 5G NR standard parameters:
    - 30 kHz subcarrier spacing
    - 4096-point IFFT
    - 3.5 GHz carrier frequency
    - TDL-A channel model
    
    WARNING: This function is provided for testing and examples only.
    In production code, always create and configure a SystemConfig 
    explicitly rather than relying on defaults.
    
    Returns:
        SystemConfig object
    """
    config = SystemConfig()
    config.validate_config()
    return config


def create_lte_system_config() -> SystemConfig:
    """
    创建LTE系统配置
    
    使用LTE标准参数：
    - 15 kHz子载波间隔
    - 2048点IFFT
    - 2.6 GHz载波频率
    
    Returns:
        SystemConfig对象
    """
    config = SystemConfig(
        subcarrier_spacing=15e3,
        ifft_size=2048,
        carrier_frequency=2.6e9,
        channel_model_type="TDL-A"
    )
    config.validate_config()
    return config


def create_5g_mmwave_system_config() -> SystemConfig:
    """
    创建5G毫米波系统配置
    
    使用5G毫米波参数：
    - 120 kHz子载波间隔
    - 4096点IFFT
    - 28 GHz载波频率
    
    Returns:
        SystemConfig对象
    """
    config = SystemConfig(
        subcarrier_spacing=120e3,
        ifft_size=4096,
        carrier_frequency=28e9,
        channel_model_type="TDL-D"  # LOS模型适合毫米波
    )
    config.validate_config()
    return config


if __name__ == "__main__":
    # 测试不同的系统配置
    print("🧪 测试系统配置")
    print("=" * 50)
    
    configs = [
        ("5G NR 默认配置", create_default_system_config()),
        ("LTE 配置", create_lte_system_config()),
        ("5G 毫米波配置", create_5g_mmwave_system_config())
    ]
    
    for name, config in configs:
        print(f"\n📋 {name}")
        print("-" * 30)
        config.print_summary()
