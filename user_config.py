"""
用户级SRS配置

这个配置定义了用户相关的SRS参数：
- 用户数量和端口配置
- 循环移位配置
- 序列参数

# For system-level physical layer parameters (like sampling rate, carrier frequency, etc.), please refer to system_config.py
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
    - 所有参数均支持随机化（通过提供范围或列表选择）
    """
    
    # Basic parameters - now support ranges for randomization
    seq_length: List[int] = None  # List of possible sequence lengths (L)
    ktc_options: List[int] = None  # List of possible ktc values (ktc=4 -> K=12, ktc=2 -> K=8)
    
    # 🔧 主要配置：循环移位（其他参数从此推导）
    # List of possible cyclic shift configurations
    # Each configuration is a list of lists, representing multiple users and their port cyclic shifts
    cyclic_shifts_configs: List[List[List[int]]] = None  # Multiple possible cyclic shift configurations
    
    # SNR configuration
    snr_range: Tuple[float, float] = None  # SNR range in dB (min, max). If min=max, use fixed SNR
    
    # Timing offset configuration  
    timing_offset_range: Tuple[float, float] = None  # Timing offset range in seconds (min, max)
    
    # Channel model configuration - combined model and delay spread
    # Format: ["TDL-A-30", "TDL-C-300", ...] - model name followed by delay spread in ns
    channel_models: List[str] = None  # List of possible channel models with delay spreads in ns
    
    # MMSE processing parameters
    mmse_block_size: int = 12  # Size of blocks for MMSE filtering
    
    # Current active configuration (selected randomly for each sample)
    _current_seq_length: int = None
    _current_ktc: int = None
    _current_cyclic_shifts: List[List[int]] = None
    _current_channel_model: str = None
    
    def __post_init__(self):
        """Initialize with default values if not provided"""
        # Set default values for optional parameters
        if self.seq_length is None:
            self.seq_length = [816]  # Default to fixed sequence length
        
        if self.ktc_options is None:
            self.ktc_options = [4]  # Default to fixed ktc=4 (K=12)
        
        if self.cyclic_shifts_configs is None:
            # Default to a single configuration with one user and one port
            self.cyclic_shifts_configs = [[[0]]]
            
        if self.snr_range is None:
            self.snr_range = (20.0, 20.0)  # Default to fixed 20dB SNR
            
        if self.timing_offset_range is None:
            self.timing_offset_range = (0.0, 0.0)  # Default to no timing offset
            
        if self.channel_models is None:
            self.channel_models = ["TDL-A-30"]  # Default to TDL-A with 30ns delay spread
            
        # Initialize current configuration
        self.randomize_configuration()
    
    def randomize_configuration(self):
        """Randomly select a new configuration from the available options"""
        import random
        
        # Select random sequence length
        self._current_seq_length = random.choice(self.seq_length)
        
        # Select random ktc
        self._current_ktc = random.choice(self.ktc_options)
        
        # Calculate K based on selected ktc
        current_K = 12 if self._current_ktc == 4 else 8
        
        # Find compatible cyclic shift configurations
        compatible_configs = []
        for config in self.cyclic_shifts_configs:
            is_compatible = True
            for user_shifts in config:
                for shift in user_shifts:
                    if shift >= current_K:
                        is_compatible = False
                        break
                if not is_compatible:
                    break
            if is_compatible:
                compatible_configs.append(config)
        
        # If no compatible configs found, create a default one
        if not compatible_configs:
            # Create a simple default configuration compatible with current K
            max_users = len(self.cyclic_shifts_configs[0]) if self.cyclic_shifts_configs else 2
            default_config = []
            for user_id in range(max_users):
                # Assign non-overlapping shifts within the valid range
                base_shift = user_id % current_K
                default_config.append([base_shift])
            compatible_configs = [default_config]
        
        # Select random compatible configuration
        self._current_cyclic_shifts = random.choice(compatible_configs)
        
        # Select random channel model
        self._current_channel_model = random.choice(self.channel_models)
    
    @property
    def current_seq_length(self) -> int:
        """Get current sequence length"""
        return self._current_seq_length
    
    @property
    def current_ktc(self) -> int:
        """Get current ktc value"""
        return self._current_ktc
    
    @property
    def current_cyclic_shifts(self) -> List[List[int]]:
        """Get current cyclic shifts configuration"""
        return self._current_cyclic_shifts
    
    @property
    def current_channel_model(self) -> str:
        """Get current channel model with delay spread"""
        return self._current_channel_model
    
    def parse_channel_model(self) -> Tuple[str, float]:
        """
        Parse the current channel model string into model type and delay spread
        
        Returns:
            Tuple of (model_type: str, delay_spread: float in seconds)
        """
        # Split by "-" to get parts: e.g., "TDL-A-30" -> ["TDL", "A", "30"]
        parts = self._current_channel_model.split("-")
        
        # If format is "TDL-A-30", combine first two parts for model type
        if len(parts) == 3:
            model_type = f"{parts[0]}-{parts[1]}"
            delay_spread = float(parts[2]) * 1e-9  # Convert ns to seconds
        # If format is different, use best guess
        else:
            model_type = parts[0]
            delay_spread = 30e-9  # Default to 30ns
            
        return model_type, delay_spread
    
    @property
    def K(self) -> int:
        """Get number of cyclic shifts K based on current ktc"""
        return 12 if self._current_ktc == 4 else 8
    
    @property 
    def num_users(self) -> int:
        """从当前循环移位配置推导用户数量"""
        return len(self._current_cyclic_shifts)
    
    @property
    def ports_per_user(self) -> List[int]:
        """从当前循环移位配置推导每个用户的端口数量"""
        return [len(user_shifts) for user_shifts in self._current_cyclic_shifts]
    
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
        Compute Locc based on current user configuration
        
        In 3GPP, this would be calculated based on resource allocation.
        This is a simplified implementation.
        """
        if self._current_ktc == 4:
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
        # 检查序列长度列表
        if not self.seq_length or not all(isinstance(length, int) and length > 0 for length in self.seq_length):
            raise ValueError("序列长度必须是正整数列表")
            
        # 检查ktc选项列表
        if not self.ktc_options or not all(ktc in [2, 4] for ktc in self.ktc_options):
            raise ValueError("ktc选项必须是[2, 4]中的值")
            
        # 检查cyclic_shifts_configs格式和内容
        if not self.cyclic_shifts_configs:
            raise ValueError("至少需要配置一种循环移位配置")
            
        # 验证每个循环移位配置
        for config_idx, config in enumerate(self.cyclic_shifts_configs):
            # 检查是否有用户
            if not config:
                raise ValueError(f"配置 {config_idx} 至少需要一个用户的循环移位")
            
            # 检查每个用户是否有端口
            for u, user_shifts in enumerate(config):
                if not user_shifts:
                    raise ValueError(f"配置 {config_idx} 中的用户 {u} 必须至少有一个端口")
                
                # 此处暂时无法检查循环移位是否在K范围内，因为K取决于运行时选择的ktc
                # 在实际使用时会在randomize_configuration后进行检查
        
        # 检查SNR范围
        if self.snr_range[0] > self.snr_range[1]:
            raise ValueError(f"SNR范围无效: 最小值 {self.snr_range[0]} 大于最大值 {self.snr_range[1]}")
        
        # 检查时间偏移范围
        if self.timing_offset_range[0] > self.timing_offset_range[1]:
            raise ValueError(f"时间偏移范围无效: 最小值 {self.timing_offset_range[0]} 大于最大值 {self.timing_offset_range[1]}")
            
        # 检查信道模型格式
        for model in self.channel_models:
            parts = model.split("-")
            if len(parts) < 2:
                raise ValueError(f"信道模型格式无效: {model}，应为'TDL-A-30'格式")
            
            # 检查延迟扩散值是否为数字
            try:
                if len(parts) >= 3:
                    float(parts[2])  # 尝试将延迟扩散值转换为浮点数
            except ValueError:
                raise ValueError(f"信道模型延迟扩散值无效: {model}")
        
        # 检查当前活动配置
        self._validate_current_config()
        
        # All checks passed
        return True
        
    def _validate_current_config(self) -> bool:
        """
        验证当前活动的配置是否有效
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Skip if configuration is not initialized yet
        if (self._current_seq_length is None or self._current_ktc is None or 
            self._current_cyclic_shifts is None or self._current_channel_model is None):
            return True
            
        # Calculate current K value based on ktc
        current_K = 12 if self._current_ktc == 4 else 8
        
        # 检查循环移位值是否有效 (0 to K-1)
        for u, user_shifts in enumerate(self._current_cyclic_shifts):
            for shift in user_shifts:
                if shift < 0 or shift >= current_K:
                    raise ValueError(f"当前配置中用户 {u} 的循环移位 {shift} 超出范围 [0, {current_K-1}]")
        
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
            'cyclic_shifts': self._current_cyclic_shifts[user_id],
            'sequence_length': self._current_seq_length,
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
        
    def is_using_random_config(self) -> bool:
        """
        检查是否使用随机配置（任意参数支持随机）
        
        Returns:
            True if using any randomization, False if all parameters are fixed
        """
        return (
            len(self.seq_length) > 1 or 
            len(self.ktc_options) > 1 or 
            len(self.cyclic_shifts_configs) > 1 or 
            len(self.channel_models) > 1 or
            self.snr_range[0] != self.snr_range[1] or
            self.timing_offset_range[0] != self.timing_offset_range[1]
        )
    
    def print_user_summary(self):
        """打印用户配置摘要"""
        print("👥 用户配置摘要")
        print("=" * 50)
        
        print("📋 基础参数配置范围:")
        print(f"   序列长度选项: {self.seq_length}")
        print(f"   Ktc选项: {self.ktc_options}")
        print(f"   MMSE块大小: {self.mmse_block_size}")
        print(f"   信道模型选项: {self.channel_models}")
        
        print(f"\n📋 当前活动配置:")
        print(f"   序列长度: {self._current_seq_length}")
        print(f"   Ktc: {self._current_ktc} (K={self.K})")
        model_type, delay_spread = self.parse_channel_model()
        print(f"   信道模型: {model_type}, 延迟扩展: {delay_spread*1e9:.1f} ns")
        
        print(f"\n👥 当前用户配置:")
        print(f"   用户数: {self.num_users}")
        print(f"   总端口数: {self.total_ports}")
        for i in range(self.num_users):
            print(f"   用户{i}: {self.ports_per_user[i]}个端口, 循环移位: {self._current_cyclic_shifts[i]}")
        
        print(f"\n📊 信号质量参数:")
        if self.is_fixed_snr():
            print(f"   SNR: {self.snr_range[0]:.1f} dB (固定)")
        else:
            print(f"   SNR范围: {self.snr_range[0]:.1f} ~ {self.snr_range[1]:.1f} dB (随机)")
        
        if self.is_fixed_timing_offset():
            print(f"   时间偏移: {self.timing_offset_range[0]*1e9:.1f} ns (固定)")
        else:
            print(f"   时间偏移范围: {self.timing_offset_range[0]*1e9:.1f} ~ {self.timing_offset_range[1]*1e9:.1f} ns (随机)")
        
        print(f"\n🔄 随机化状态: {'启用' if self.is_using_random_config() else '禁用'}")
        
        print(f"\n✅ 配置验证: ", end="")
        try:
            self.validate_config()
            print("通过")
        except ValueError as e:
            print(f"失败 - {e}")
            
    def generate_new_sample_config(self):
        """
        为新样本生成随机配置
        当需要为每个训练/评估样本使用不同配置时调用此方法
        """
        self.randomize_configuration()
        return {
            'seq_length': self._current_seq_length,
            'ktc': self._current_ktc, 
            'cyclic_shifts': self._current_cyclic_shifts,
            'channel_model': self.parse_channel_model()[0],
            'delay_spread': self.parse_channel_model()[1],
            'snr_db': self.get_snr_db(),
            'timing_offset': self.get_timing_offset_seconds()
        }
        

def create_example_config() -> SRSConfig:
    """
    Create an example SRS configuration with randomized parameters
    
    Returns:
        SRSConfig object with randomized parameters
    """
    return SRSConfig(
        seq_length=list(range(12, 816+1, 12)),  # Create list from 12 to 816 with step 12
        ktc_options=[4],  # Support both K=8 and K=12
        cyclic_shifts_configs=[
            # Configuration 1: 2 users with 2 ports each (compatible with K=12)
            [
                [0, 6],  # User 0: 2 ports
                [3, 9]   # User 1: 2 ports
            ],
            # Configuration 2: 3 users with varying ports (compatible with K=8)
            [
                [0],      # User 0: 1 port
                [2, 5],   # User 1: 2 ports
                [7]       # User 2: 1 port
            ],
            # Configuration 3: 2 users with 2 ports each (compatible with K=8)
            [
                [0, 4],  # User 0: 2 ports
                [2, 6]   # User 1: 2 ports
            ]
        ],
        channel_models=["TDL-A-30", "TDL-B-100", "TDL-C-300"],  # Multiple channel models
        snr_range=(10.0, 30.0),  # SNR range: 10-30 dB (random)
        timing_offset_range=(-130e-9, 130e-9),  # Timing offset: -130ns to 130ns (random)
        mmse_block_size=12  # Default block size for MMSE filtering
    )


def create_fixed_snr_config() -> SRSConfig:
    """
    创建固定SNR的示例配置，但其他参数仍支持随机化
    
    Returns:
        SRSConfig object with fixed SNR but randomized other parameters
    """
    return SRSConfig(
        seq_length=[1200],  # Single sequence length option
        ktc_options=[4],  # Only K=12
        cyclic_shifts_configs=[
            # Only one configuration: 2 users with 2 ports each
            [
                [0, 6],  # User 0: 2 ports
                [3, 9]   # User 1: 2 ports
            ]
        ],
        channel_models=["TDL-A-30", "TDL-C-300"],  # Two channel model options
        snr_range=(25.0, 25.0),  # Fixed SNR: 25 dB
        timing_offset_range=(0.0, 0.0),  # No timing offset
        mmse_block_size=12
    )


def create_multi_user_config() -> SRSConfig:
    """
    创建多用户配置示例，所有参数都支持随机化
    
    Returns:
        SRSConfig object with multiple users and full randomization
    """
    return SRSConfig(
        seq_length=[720, 816, 1200, 1500],  # Multiple sequence length options
        ktc_options=[2, 4],  # Support both K=8 and K=12
        cyclic_shifts_configs=[
            # Configuration 1: 3 users (K=8)
            [
                [0],      # User 0: 1 port
                [2, 5],   # User 1: 2 ports
                [7]       # User 2: 1 port
            ],
            # Configuration 2: 4 users (K=12)
            [
                [0],      # User 0: 1 port
                [3],      # User 1: 1 port
                [6],      # User 2: 1 port
                [9]       # User 3: 1 port
            ],
            # Configuration 3: 2 users with multiple ports (compatible with both K=8 and K=12)
            [
                [0, 3, 6],  # User 0: 3 ports
                [1, 4, 7]   # User 1: 3 ports
            ]
        ],
        channel_models=["TDL-A-10", "TDL-A-30", "TDL-B-100", "TDL-C-300", "TDL-D-100", "TDL-E-30"],
        snr_range=(0.0, 40.0),  # Very wide SNR range for comprehensive training
        timing_offset_range=(-200e-9, 200e-9),  # Wide timing offset range
        mmse_block_size=12
    )


def create_two_user_config() -> SRSConfig:
    """
    创建2用户配置（每用户1端口）
    
    Returns:
        SRSConfig object with 2 users, 1 port each
    """
    return SRSConfig(
        seq_length=[1200],  # Fixed sequence length
        ktc_options=[4],    # Fixed K=12
        cyclic_shifts_configs=[
            # Only one configuration: 2 users with 1 port each
            [
                [0],  # User 0: 1 port
                [6]   # User 1: 1 port
            ]
        ],
        channel_models=["TDL-A-30"],  # Fixed channel model
        snr_range=(20.0, 30.0),  # SNR range: 20-30 dB (random)
        timing_offset_range=(-130e-9, 130e-9),  # Timing offset range (random)
        mmse_block_size=12
    )


if __name__ == "__main__":
    # 测试不同的用户配置
    print("🧪 测试用户配置")
    print("=" * 50)
    
    configs = [
        ("基础配置 (所有参数随机化)", create_example_config()),
        ("固定SNR配置 (其他参数随机)", create_fixed_snr_config()),
        ("全面随机多用户配置", create_multi_user_config()),
        ("简单两用户配置", create_two_user_config())
    ]
    
    for name, config in configs:
        print(f"\n📋 {name}")
        print("-" * 30)
        config.print_user_summary()
        
        # 测试几次完整随机配置生成
        print(f"\n🎲 随机样本配置测试:")
        for i in range(3):
            # 为每个样本重新随机化配置
            sample_config = config.generate_new_sample_config()
            
            # 打印主要参数
            print(f"\n   样本 {i+1}:")
            print(f"     序列长度: {sample_config['seq_length']}")
            print(f"     ktc: {sample_config['ktc']}")
            print(f"     信道模型: {sample_config['channel_model']}, 延迟扩展: {sample_config['delay_spread']*1e9:.1f}ns")
            print(f"     SNR: {sample_config['snr_db']:.1f}dB")
            print(f"     时间偏移: {sample_config['timing_offset']*1e9:.1f}ns")
            print(f"     用户配置:")
            for u, shifts in enumerate(sample_config['cyclic_shifts']):
                print(f"       用户{u}: 端口数={len(shifts)}, 循环移位={shifts}")
