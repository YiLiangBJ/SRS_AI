"""
用户级SRS配置

这个配置定义了用户相关的SRS参数：
- 用户数量和端口配置
- 循环移位配置
- 序列参数

系统级物理层参数 (如采样率、载波频率等) 请参考 system_config.py
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class SRSConfig:
    """
    SRS用户配置参数
    
    这个类只包含与用户相关的SRS参数。
    系统级参数 (采样率、载波频率等) 在 SystemConfig 中定义。
    
    🔧 配置策略：
    - cyclic_shifts 是主要配置，num_users 和 ports_per_user 从它推导
    - 避免配置不一致的问题
    """
    
    # Basic parameters
    seq_length: int  # Length of SRS sequence (L)
    ktc: int  # Configuration parameter (ktc=4 -> K=12, ktc=2 -> K=8)
    
    # 🔧 主要配置：循环移位（其他参数从此推导）
    cyclic_shifts: List[List[int]]  # Cyclic shift parameters for each user's ports
    
    # SNR configuration
    snr_range: Tuple[float, float]  # SNR range in dB (min, max). If min=max, use fixed SNR
    
    # Timing offset configuration  
    timing_offset_range: Tuple[float, float]  # Timing offset range in seconds (min, max)
    
    # MMSE processing parameters
    mmse_block_size: int = 12  # Size of blocks for MMSE filtering
    
    @property
    def K(self) -> int:
        """Get number of cyclic shifts K based on ktc"""
        return 12 if self.ktc == 4 else 8
    
    @property 
    def num_users(self) -> int:
        """从cyclic_shifts推导用户数量"""
        return len(self.cyclic_shifts)
    
    @property
    def ports_per_user(self) -> List[int]:
        """从cyclic_shifts推导每个用户的端口数量"""
        return [len(user_shifts) for user_shifts in self.cyclic_shifts]
    
    @property
    def total_ports(self) -> int:
        """Calculate total number of ports across all users"""
        return sum(self.ports_per_user)
    
    def get_snr_db(self) -> float:
        """
        获取SNR值 (dB)
        
        如果snr_range的最小值和最大值相等，返回固定SNR。
        否则在范围内随机选择SNR。
        
        Returns:
            SNR值 (dB)
        """
        min_snr, max_snr = self.snr_range
        if min_snr == max_snr:
            return min_snr  # 固定SNR
        else:
            import random
            return random.uniform(min_snr, max_snr)  # 随机SNR
    
    def is_fixed_snr(self) -> bool:
        """
        检查是否使用固定SNR
        
        Returns:
            True if using fixed SNR, False if using random SNR
        """
        return self.snr_range[0] == self.snr_range[1]
    
    def get_locc(self) -> int:
        """
        Compute Locc based on user configuration
        
        In 3GPP, this would be calculated based on resource allocation.
        This is a simplified implementation.
        """
        if self.ktc == 4:
            if self.num_users == 1:
                return 1
            elif self.num_users == 2:
                return 4 if max(self.ports_per_user) <= 2 else 6
            else:
                return 6  # For more users
        else:  # ktc == 2
            if self.num_users == 1:
                return 1
            elif self.num_users == 2:
                return 2 if max(self.ports_per_user) <= 2 else 4
            else:
                return 4  # For more users
    
    def validate_config(self) -> bool:
        """
        Validate that the configuration is correct
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # 检查是否有用户
        if len(self.cyclic_shifts) == 0:
            raise ValueError("至少需要配置一个用户的循环移位")
        
        # 检查每个用户是否有端口
        for u, user_shifts in enumerate(self.cyclic_shifts):
            if len(user_shifts) == 0:
                raise ValueError(f"用户 {u} 必须至少有一个端口")
        
        # 检查循环移位值是否有效 (0 to K-1)
        for u, user_shifts in enumerate(self.cyclic_shifts):
            for shift in user_shifts:
                if shift < 0 or shift >= self.K:
                    raise ValueError(f"用户 {u} 的循环移位 {shift} 超出范围 [0, {self.K-1}]")
        
        # 检查SNR范围
        if self.snr_range[0] > self.snr_range[1]:
            raise ValueError(f"SNR范围无效: 最小值 {self.snr_range[0]} 大于最大值 {self.snr_range[1]}")
        
        # 检查时间偏移范围
        if self.timing_offset_range[0] > self.timing_offset_range[1]:
            raise ValueError(f"时间偏移范围无效: 最小值 {self.timing_offset_range[0]} 大于最大值 {self.timing_offset_range[1]}")
        
        # All checks passed
        return True
    
    def get_user_config(self, user_id: int) -> Dict:
        """
        获取指定用户的配置信息
        
        Args:
            user_id: 用户ID (0-based index)
            
        Returns:
            包含用户配置的字典
        """
        if user_id < 0 or user_id >= self.num_users:
            raise ValueError(f"User ID {user_id} is out of range [0, {self.num_users-1}]")
        
        return {
            'user_id': user_id,
            'num_ports': self.ports_per_user[user_id],
            'cyclic_shifts': self.cyclic_shifts[user_id],
            'sequence_length': self.seq_length,
            'sequence_type': 'zadoff_chu',  # Default sequence type
            'start_subcarrier': None,  # Will be determined by mapping logic
        }
    
    def get_timing_offset_seconds(self) -> float:
        """
        获取时间偏移值 (秒)
        
        如果timing_offset_range的最小值和最大值相等，返回固定偏移。
        否则在范围内随机选择偏移。
        
        Returns:
            时间偏移值 (秒)
        """
        min_offset, max_offset = self.timing_offset_range
        if min_offset == max_offset:
            return min_offset  # 固定偏移
        else:
            import random
            return random.uniform(min_offset, max_offset)  # 随机偏移
    
    def get_timing_offset_samples(self, sampling_rate: float) -> int:
        """
        获取时间偏移对应的采样点数
        
        Args:
            sampling_rate: 采样率 (Hz)
            
        Returns:
            时间偏移对应的采样点数 (整数)
        """
        timing_offset_seconds = self.get_timing_offset_seconds()
        return int(timing_offset_seconds * sampling_rate)
    
    def is_fixed_timing_offset(self) -> bool:
        """
        检查是否使用固定时间偏移
        
        Returns:
            True if using fixed timing offset, False if using random timing offset
        """
        return self.timing_offset_range[0] == self.timing_offset_range[1]
    
    def print_user_summary(self):
        """打印用户配置摘要"""
        print("👥 用户配置摘要")
        print("=" * 50)
        
        print("📋 基础参数:")
        print(f"   序列长度: {self.seq_length}")
        print(f"   Ktc: {self.ktc} (K={self.K})")
        print(f"   MMSE块大小: {self.mmse_block_size}")
        
        print(f"\n👥 用户配置:")
        print(f"   用户数: {self.num_users}")
        print(f"   总端口数: {self.total_ports}")
        for i in range(self.num_users):
            print(f"   用户{i}: {self.ports_per_user[i]}个端口, 循环移位: {self.cyclic_shifts[i]}")
        
        print(f"\n📊 信号质量参数:")
        if self.is_fixed_snr():
            print(f"   SNR: {self.snr_range[0]:.1f} dB (固定)")
        else:
            print(f"   SNR范围: {self.snr_range[0]:.1f} ~ {self.snr_range[1]:.1f} dB (随机)")
        
        if self.is_fixed_timing_offset():
            print(f"   时间偏移: {self.timing_offset_range[0]*1e9:.1f} ns (固定)")
        else:
            print(f"   时间偏移范围: {self.timing_offset_range[0]*1e9:.1f} ~ {self.timing_offset_range[1]*1e9:.1f} ns (随机)")
        
        print(f"\n✅ 配置验证: ", end="")
        try:
            self.validate_config()
            print("通过")
        except ValueError as e:
            print(f"失败 - {e}")
        

def create_example_config() -> SRSConfig:
    """
    Create an example SRS configuration as described in the requirements
    
    Returns:
        SRSConfig object with example parameters
    """
    # Example configuration: 1 user with 1 port
    return SRSConfig(
        seq_length=1200,  # Example value, can be changed
        ktc=4,  # K=12
        cyclic_shifts=[
            [0]  # User 0: 1个端口，循环移位为0
        ],
        snr_range=(30.0, 30.0),  # SNR range: 20-30 dB (random)
        timing_offset_range=(-130e-9, 130e-9),  # Timing offset: -130ns to 130ns (random)
        mmse_block_size=12  # Default block size for MMSE filtering
    )


def create_fixed_snr_config() -> SRSConfig:
    """
    创建固定SNR的示例配置
    
    Returns:
        SRSConfig object with fixed SNR and timing offset
    """
    return SRSConfig(
        seq_length=1200,
        ktc=4,  # K=12
        cyclic_shifts=[
            [0, 6],  # User 0: 2个端口
            [3, 9]   # User 1: 2个端口
        ],
        snr_range=(25.0, 25.0),  # Fixed SNR: 25 dB
        timing_offset_range=(0.0, 0.0),  # No timing offset
        mmse_block_size=12
    )


def create_multi_user_config() -> SRSConfig:
    """
    创建多用户配置示例
    
    Returns:
        SRSConfig object with multiple users
    """
    return SRSConfig(
        seq_length=1200,
        ktc=2,  # K=8
        cyclic_shifts=[
            [0],      # User 0: 1个端口
            [2, 5],   # User 1: 2个端口
            [7]       # User 2: 1个端口
        ],
        snr_range=(10.0, 40.0),  # Wide SNR range for training
        timing_offset_range=(-100e-9, 100e-9),  # Moderate timing offset range
        mmse_block_size=12
    )


def create_two_user_config() -> SRSConfig:
    """
    创建2用户配置（每用户1端口）
    
    Returns:
        SRSConfig object with 2 users, 1 port each
    """
    return SRSConfig(
        seq_length=1200,
        ktc=4,  # K=12
        cyclic_shifts=[
            [0],  # User 0: 1个端口
            [6]   # User 1: 1个端口
        ],
        snr_range=(20.0, 30.0),  # SNR range: 20-30 dB (random)
        timing_offset_range=(-130e-9, 130e-9),  # Timing offset: -130ns to 130ns (random)
        mmse_block_size=12
    )


if __name__ == "__main__":
    # 测试不同的用户配置
    print("🧪 测试用户配置")
    print("=" * 50)
    
    configs = [
        ("基础配置 (随机SNR和timing offset)", create_example_config()),
        ("固定SNR配置", create_fixed_snr_config()),
        ("多用户配置", create_multi_user_config()),
        ("两用户配置", create_two_user_config())
    ]
    
    for name, config in configs:
        print(f"\n📋 {name}")
        print("-" * 30)
        config.print_user_summary()
        
        # 测试几次随机值生成
        print(f"\n🎲 随机值测试:")
        for i in range(3):
            snr = config.get_snr_db()
            timing_offset_ns = config.get_timing_offset_seconds() * 1e9
            print(f"   测试{i+1}: SNR={snr:.1f}dB, Timing offset={timing_offset_ns:.1f}ns")
