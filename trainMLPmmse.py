import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


import sionna
SIONNA_AVAILABLE = True
print("SIONNA available - using professional 3GPP channel models")


from professional_channels import SIONNAChannelModel, SIONNAChannelGenerator, print_sionna_info
PROFESSIONAL_CHANNELS_AVAILABLE = True
print("Professional channel wrapper available")


import argparse
from typing import List, Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model_Traditional import SRSChannelEstimator
from model_AIpart import TrainableMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


class SRSTrainerModified:
    """
    Modified Trainer for SRS Channel Estimator that uses h_with_residual/phasor as input
    """
    def __init__(
        self,
        srs_estimator: SRSChannelEstimator = None,
        mmse_module: TrainableMMSEModule = None,
        config: SRSConfig = None,
        data_generator: SRSDataGenerator = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        use_tensorboard: bool = True,
        log_dir: str = "./logs",
        # Legacy parameters for backward compatibility
        srs_config: SRSConfig = None,
        save_dir: str = "./checkpoints_modified",
        use_trainable_mmse: bool = True,
        enable_plotting: bool = False,
        use_professional_channels: bool = True,
        use_sionna: bool = True
    ):
        """
        Initialize the trainer with support for distributed training
        
        Args:
            srs_estimator: Pre-initialized SRS estimator (for DDP support)
            mmse_module: Pre-initialized MMSE module (for DDP support)
            config: SRS configuration
            data_generator: Pre-initialized data generator
            device: Computation device
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            use_tensorboard: Whether to use TensorBoard logging
            log_dir: Directory for logs
            
            # Legacy parameters (backward compatibility)
            srs_config: Legacy parameter name for config
            save_dir: Directory for saving checkpoints
            use_trainable_mmse: Whether to use trainable MMSE
            enable_plotting: Whether to enable plotting
            use_professional_channels: Whether to use professional channels
            use_sionna: Whether to use SIONNA
        """
        # Handle legacy parameter names
        if config is None and srs_config is not None:
            config = srs_config
        
        if config is None:
            config = create_example_config()
        
        self.config = config
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.enable_plotting = enable_plotting
        
        # Initialize models if not provided (for backward compatibility)
        if srs_estimator is None or mmse_module is None:
            self._init_models_legacy(use_trainable_mmse)
        else:
            self.srs_estimator = srs_estimator
            self.mmse_module = mmse_module
        
        # Always initialize legacy attributes first to ensure all required attributes exist
        self._init_legacy_attributes(use_professional_channels, use_sionna)
            
        # Then handle the data generator
        if data_generator is not None:
            self.data_generator = data_generator
        
        # Initialize data structures for multi-generator support
        # 🔧 创建这些字典优先，确保其他方法可以向这些字典中添加项
        self.data_generators = {}  # 按SNR组织的数据生成器
        self.channel_models = {}   # 按信道类型和延时参数组织的信道模型
        self.per_ue_channels = {}  # 按UE ID组织的专用信道实例
        self.per_port_generators = {}  # 按端口ID组织的专用生成器实例
        
        # Initialize common channel configs (needed for all instances)
        self.common_channel_configs = [
            {'model': 'TDL-A', 'delay_spread': 30e-9},
            {'model': 'TDL-B', 'delay_spread': 100e-9},
            {'model': 'TDL-C', 'delay_spread': 300e-9},
        ]
        
        # Initialize training history (needed for all instances)
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        
        # Global step counter for logging
        self.global_step = 0
        
        # Initialize optimizer - ensure we have models first
        if hasattr(self, 'mmse_module') and self.mmse_module is not None:
            self.optimizer = optim.Adam(
                self.mmse_module.parameters(),
                lr=self.learning_rate,
                weight_decay=1e-5
            )
        else:
            self.optimizer = None  # Will be set later in _init_models_legacy
        
        # Initialize TensorBoard writer
        if self.use_tensorboard:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(self.log_dir)
        else:
            self.writer = None
        
        # Initialize checkpoint directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        print(f"✅ SRSTrainerModified initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - TensorBoard: {self.use_tensorboard}")
        print(f"   - Log directory: {self.log_dir}")
        print(f"   - Save directory: {self.save_dir}")
    
    def _init_models_legacy(self, use_trainable_mmse: bool):
        """Initialize models for backward compatibility"""
        # Create trainable MMSE module if needed
        if use_trainable_mmse:
            self.mmse_module = TrainableMMSEModule(
                seq_length=self.config.seq_length,
                mmse_block_size=self.config.mmse_block_size,
                use_complex_input=True
            ).to(self.device)
        else:
            self.mmse_module = None
            
        # Create models
        self.srs_estimator = SRSChannelEstimator(
            seq_length=self.config.seq_length,
            ktc=self.config.ktc,
            max_users=self.config.num_users,
            max_ports_per_user=max(self.config.ports_per_user),
            mmse_block_size=self.config.mmse_block_size,
            device=self.device,
            mmse_module=self.mmse_module if use_trainable_mmse else None  # 传入 MMSE 模块
        ).to(self.device)

        # Make sure all model parameters require gradients
        for name, param in self.srs_estimator.named_parameters():
            param.requires_grad = True
            print(f"Setting requires_grad=True for SRS estimator parameter: {name}, shape: {param.shape}")
            
        if self.mmse_module:
            for name, param in self.mmse_module.named_parameters():
                param.requires_grad = True
                print(f"Setting requires_grad=True for MMSE parameter: {name}, shape: {param.shape}")
        
        # Create optimizer
        model_params = []
        # 添加SRSChannelEstimator的参数
        for name, param in self.srs_estimator.named_parameters():
            if param.requires_grad:
                print(f"Adding trainable parameter: {name}, shape: {param.shape}")
                model_params.append(param)
        
        # 添加MMSE模块的参数
        if self.mmse_module:
            for name, param in self.mmse_module.named_parameters():
                if param.requires_grad:
                    print(f"Adding trainable MMSE parameter: {name}, shape: {param.shape}")
                    model_params.append(param)
        
        # 确保模型有可训练参数
        if len(model_params) == 0:
            raise ValueError("No trainable parameters found in the model!")
          
        # Print number of trainable parameters
        total_params = sum(p.numel() for p in model_params)
        print(f"总共有 {total_params} 个可训练参数")
        
        self.optimizer = optim.Adam(model_params, lr=0.01)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Create logs directory for TensorBoard
        legacy_log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(legacy_log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer (if not already initialized)
        if not hasattr(self, 'writer') or self.writer is None:
            legacy_log_dir = os.path.join(self.save_dir, 'logs')
            os.makedirs(legacy_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=legacy_log_dir)
        
        # Initialize training history (if not already initialized)
        if not hasattr(self, 'train_losses'):
            self.train_losses = []
            self.val_losses = []
            self.train_nmse = []
            self.val_nmse = []
        
        # Initialize global step counter (if not already initialized)
        if not hasattr(self, 'global_step'):
            self.global_step = 0
        
        # 🎯 完整实例化所有需要的组件
        print(f"\n🚀 开始完整实例化所有组件...")
        
        # Initialize legacy attributes if not already done
        if not hasattr(self, 'channel_params'):
            self._init_legacy_attributes(use_professional_channels=True, use_sionna=True)
        
        # 执行完整初始化
        try:
            self._initialize_all_instances()
        except Exception as e:
            print(f"⚠️ 实例初始化失败，继续使用基本配置: {e}")
            # 确保基本的数据生成器可用
            if not hasattr(self, 'data_generator') or self.data_generator is None:
                self._init_data_generator()
            
    def _init_legacy_attributes(self, use_professional_channels: bool, use_sionna: bool):
        """
        Initialize legacy attributes for backward compatibility
        
        This method initializes all the attributes that were previously
        initialized in the legacy constructor paths.
        """
        # Import required modules
        from system_config import create_default_system_config
        
        # Initialize system config
        self.system_config = create_default_system_config()
        
        # Initialize channel parameters
        self.channel_params = {
            'channel_model': 'TDL-C',  # Default channel model
            'delay_spread': self.system_config.delay_spread,
            'carrier_frequency': self.system_config.carrier_frequency,
            'device': self.device,
            'use_sionna': use_sionna and SIONNA_AVAILABLE,
            'use_professional_channels': use_professional_channels and PROFESSIONAL_CHANNELS_AVAILABLE
        }
        
        # Initialize signal generation parameters
        self.signal_gen_params = {
            'srs_config': self.config,
            'num_rx_antennas': self.system_config.num_rx_antennas,
            'sampling_rate': self.system_config.sampling_rate,
            'device': self.device,
            'enable_debug': False
        }
        
        # Initialize SRS config alias for backward compatibility
        self.srs_config = self.config
        
        # Initialize data generator if not provided
        if not hasattr(self, 'data_generator') or self.data_generator is None:
            self._init_data_generator()
    
    def _init_data_generator(self):
        """Initialize the data generator"""
        from data_generator_refactored import SRSDataGenerator
        
        # 首先检查是否有预初始化的数据生成器可以直接使用
        snr_key = "config_snr"
        if hasattr(self, 'data_generators') and snr_key in self.data_generators and self.data_generators[snr_key] is not None:
            self.data_generator = self.data_generators[snr_key]
            print(f"✅ 使用已初始化的数据生成器 (using_channel={self.data_generator.using_channel})")
            return
            
        # 检查是否有预初始化的信道模型可以使用
        channel_model = None
        if hasattr(self, 'channel_models') and self.channel_models:
            # 尝试获取与默认配置匹配的信道模型
            default_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
            if default_channel_key in self.channel_models and self.channel_models[default_channel_key] is not None:
                channel_model = self.channel_models[default_channel_key]
                print(f"✅ 使用预初始化的信道模型: {default_channel_key}")
            else:
                # 使用任何可用的信道模型
                available_models = [model for model in self.channel_models.values() if model is not None]
                if available_models:
                    channel_model = available_models[0]
                    found_key = [k for k, v in self.channel_models.items() if v == channel_model][0]
                    print(f"✅ 使用替代信道模型: {found_key}")
        
        # 如果没有预初始化的信道模型，则创建一个新的
        if channel_model is None and self.channel_params['use_professional_channels']:
            try:
                from professional_channels import SIONNAChannelModel
                
                channel_model = SIONNAChannelModel(
                    system_config=self.system_config,
                    model_type=self.channel_params['channel_model'],
                    num_rx_antennas=self.system_config.num_rx_antennas,
                    delay_spread=self.channel_params['delay_spread'],
                    device=self.device
                )
                print(f"✅ Created new SIONNA channel model: {self.channel_params['channel_model']}")
                
                # 缓存这个新创建的模型，如果channel_models字典存在
                if hasattr(self, 'channel_models'):
                    channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
                    self.channel_models[channel_key] = channel_model
                    print(f"✅ Cached new channel model: {channel_key}")
            except Exception as e:
                print(f"⚠️ Failed to create SIONNA channel model: {e}")
                channel_model = None
        
        # Create data generator
        self.data_generator = SRSDataGenerator(
            config=self.config,
            channel_model=channel_model,
            num_rx_antennas=self.system_config.num_rx_antennas,
            sampling_rate=self.system_config.sampling_rate,
            device=self.device
        )
        
        # 缓存这个新创建的数据生成器，如果data_generators字典存在
        if hasattr(self, 'data_generators'):
            self.data_generators[snr_key] = self.data_generator
            print(f"✅ Cached new data generator: {snr_key}")
        
        print(f"✅ Data generator initialized (using_channel={self.data_generator.using_channel})")

    def _initialize_all_instances(self):
        """
        完整初始化所有需要的实例
        
        🎯 性能优化策略：
        1. 预创建所有常用的信道模型实例
        2. 为每个SNR范围预创建数据生成器
        3. 为每个UE预创建专用信道实例（如果需要）
        4. 为每个port预创建信号生成器（如果需要）
        
        这样在训练时只需要查字典，不需要重复实例化
        """
        try:
            print(f"🚀 开始完整实例化...")
            
            # ========================================
            # 1. 初始化信道模型实例（按配置参数组织）
            # ========================================
            print(f"📡 初始化信道模型实例...")
            self._initialize_channel_models()
            
            # ========================================
            # 2. 初始化数据生成器实例（按SNR范围组织）
            # ========================================
            print(f"📊 初始化数据生成器实例...")
            self._initialize_data_generators()
            
            # ========================================
            # 3. 初始化per-UE专用实例（如果需要）
            # ========================================
            print(f"👥 初始化per-UE专用实例...")
            self._initialize_per_ue_instances()
            
            # ========================================
            # 4. 初始化per-port专用实例（如果需要）
            # ========================================
            print(f"📋 初始化per-port专用实例...")
            self._initialize_per_port_instances()
            
            print(f"✅ 所有实例初始化完成!")
            self._print_instance_summary()
            
        except Exception as e:
            print(f"❌ 实例初始化失败: {e}")
            raise RuntimeError(f"Failed to initialize instances: {e}")
    
    def _initialize_channel_models(self):
        """初始化所有常用的信道模型实例"""
        if not PROFESSIONAL_CHANNELS_AVAILABLE:
            print("⚠️  专业信道库不可用，跳过信道模型初始化")
            return
        
        # 确保common_channel_configs包含所有需要的信道配置
        if not hasattr(self, 'common_channel_configs') or not self.common_channel_configs:
            self.common_channel_configs = [
                {'model': 'TDL-A', 'delay_spread': 30e-9},
                {'model': 'TDL-B', 'delay_spread': 100e-9},
                {'model': 'TDL-C', 'delay_spread': 300e-9},
            ]
        else:
            # 确保TDL-A_30ns存在于配置中（避免警告）
            has_tdla_30ns = False
            for config in self.common_channel_configs:
                if config['model'] == 'TDL-A' and abs(config['delay_spread'] - 30e-9) < 1e-12:
                    has_tdla_30ns = True
                    break
                    
            if not has_tdla_30ns:
                print("   📌 添加TDL-A_30ns到配置中（避免警告）")
                self.common_channel_configs.insert(0, {'model': 'TDL-A', 'delay_spread': 30e-9})
        
        # 确保默认信道模型配置也包含在common_channel_configs中
        default_model = self.channel_params.get('channel_model', 'TDL-A')
        default_delay = self.channel_params.get('delay_spread', 30e-9)
        default_config_exists = False
        
        for config in self.common_channel_configs:
            if config['model'] == default_model and abs(config['delay_spread'] - default_delay) < 1e-12:
                default_config_exists = True
                break
                
        if not default_config_exists:
            print(f"   📌 添加默认信道模型 {default_model}_{default_delay*1e9:.0f}ns 到配置中")
            self.common_channel_configs.insert(0, {'model': default_model, 'delay_spread': default_delay})
            
        # 初始化所有信道模型实例
        from professional_channels import SIONNAChannelModel
        
        print(f"   🔄 正在初始化 {len(self.common_channel_configs)} 个信道模型...")
        
        # 确保至少有一个成功的信道模型
        successful_model = None
        
        for config in self.common_channel_configs:
            config_key = f"{config['model']}_{config['delay_spread']*1e9:.0f}ns"
            
            # 跳过已存在的模型
            if config_key in self.channel_models and self.channel_models[config_key] is not None:
                print(f"   ⏩ {config_key} 已存在，跳过初始化")
                successful_model = self.channel_models[config_key]  # 记录一个成功的模型
                continue
                
            try:
                print(f"   🔧 创建信道模型: {config_key}")
                channel_model = SIONNAChannelModel(
                    system_config=self.system_config,
                    model_type=config['model'],
                    num_rx_antennas=self.system_config.num_rx_antennas,
                    delay_spread=config['delay_spread'],
                    device=self.channel_params['device']
                )
                self.channel_models[config_key] = channel_model
                print(f"   ✅ {config_key} 创建成功")
                successful_model = channel_model  # 记录一个成功的模型
                
            except Exception as e:
                print(f"   ⚠️ {config_key} 创建失败: {e}, 将稍后重试")
        
        # 确保所有配置的信道模型都被成功创建
        if successful_model is not None:
            for config in self.common_channel_configs:
                config_key = f"{config['model']}_{config['delay_spread']*1e9:.0f}ns"
                
                # 如果某个信道模型创建失败，使用成功的模型替代
                if config_key not in self.channel_models or self.channel_models[config_key] is None:
                    print(f"   🔄 使用已成功创建的模型替代 {config_key}")
                    self.channel_models[config_key] = successful_model
                
        # 检查并提示初始化结果
        success_count = sum(1 for model in self.channel_models.values() if model is not None)
        print(f"   🔍 信道模型初始化结果: {success_count}/{len(self.common_channel_configs)} 个成功")
    
    def _initialize_data_generators(self):
        """初始化唯一的数据生成器（使用配置文件中的SNR范围）"""
        from data_generator_refactored import SRSDataGenerator
        
        # 获取默认信道模型
        default_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
        
        print(f"   🎯 查找默认信道模型: {default_channel_key}")
        print(f"   🎯 可用信道模型: {list(self.channel_models.keys())}")
        
        # 确保默认信道模型存在
        if default_channel_key not in self.channel_models or self.channel_models[default_channel_key] is None:
            print(f"   🔄 默认信道模型 {default_channel_key} 不存在，尝试创建")
            
            try:
                # 直接创建默认信道模型
                from professional_channels import SIONNAChannelModel
                default_channel_model = SIONNAChannelModel(
                    system_config=self.system_config,
                    model_type=self.channel_params['channel_model'],
                    num_rx_antennas=self.system_config.num_rx_antennas,
                    delay_spread=self.channel_params['delay_spread'],
                    device=self.channel_params['device']
                )
                self.channel_models[default_channel_key] = default_channel_model
                print(f"   ✅ 成功创建默认信道模型: {default_channel_key}")
            except Exception as e:
                print(f"   ⚠️ 创建默认信道模型失败: {e}")
                
                # 尝试使用任何可用的信道模型
                available_models = [model for k, model in self.channel_models.items() if model is not None]
                if available_models:
                    default_channel_model = available_models[0]
                    found_key = [k for k, v in self.channel_models.items() if v == default_channel_model][0]
                    print(f"   ✅ 使用已有信道模型: {found_key}")
                else:
                    # 如果没有可用模型，创建一个TDL-A模型作为备用
                    print(f"   🔄 创建TDL-A备用信道模型")
                    default_channel_model = SIONNAChannelModel(
                        system_config=self.system_config,
                        model_type="TDL-A",
                        num_rx_antennas=self.system_config.num_rx_antennas,
                        delay_spread=30e-9,
                        device=self.channel_params['device']
                    )
                    self.channel_models["TDL-A_30ns"] = default_channel_model
                    default_channel_key = "TDL-A_30ns"
        
        # 确保我们有一个有效的信道模型
        default_channel_model = self.channel_models[default_channel_key]
        print(f"   ✅ 使用信道模型: {default_channel_key}")
        
        # 只创建一个使用配置SNR范围的数据生成器
        config_snr_range = self.srs_config.snr_range
        snr_key = "config_snr"  # 使用固定的键名
        
        # 检查是否已经存在该数据生成器
        if snr_key in self.data_generators and self.data_generators[snr_key] is not None:
            print(f"   ⏩ 数据生成器 {snr_key} 已存在，跳过初始化")
            return
        
        try:
            print(f"   🔧 创建数据生成器: {snr_key} (SNR范围: {config_snr_range})")
            data_generator = SRSDataGenerator(
                config=self.signal_gen_params['srs_config'],
                channel_model=default_channel_model,
                num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
                sampling_rate=self.signal_gen_params['sampling_rate'],
                device=self.signal_gen_params['device']
            )
            self.data_generators[snr_key] = data_generator
            print(f"   ✅ {snr_key} 创建成功 (using_channel={data_generator.using_channel})")
            
        except Exception as e:
            print(f"   ❌ {snr_key} 创建失败: {e}")
            self.data_generators[snr_key] = None
            # 尝试创建不使用信道模型的数据生成器作为备选
            try:
                print(f"   🔄 尝试创建无信道的备选数据生成器")
                backup_generator = SRSDataGenerator(
                    config=self.signal_gen_params['srs_config'],
                    channel_model=None,  # 不使用信道模型
                    num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
                    sampling_rate=self.signal_gen_params['sampling_rate'],
                    device=self.signal_gen_params['device']
                )
                self.data_generators[snr_key] = backup_generator
                print(f"   ✅ 备选数据生成器创建成功 (using_channel=False)")
            except Exception as backup_err:
                print(f"   ❌ 备选数据生成器创建失败: {backup_err}")
                # 此时真的无法创建数据生成器，后续代码需要处理这种情况
    
    def _initialize_per_ue_instances(self):
        """为每个UE初始化专用实例（如果需要）"""
        # 当前设计使用统一的数据生成器，per-UE实例在信道内部处理
        # 如果将来需要per-UE的特殊处理，可以在这里添加
        
        # 🔧 添加配置验证，防止索引越界
        try:
            self.srs_config.validate_config()
        except Exception as e:
            raise RuntimeError(f"SRS配置验证失败: {e}")
        
        for user_id in range(self.srs_config.num_users):
            # 🔧 添加边界检查
            if user_id >= len(self.srs_config.ports_per_user):
                raise RuntimeError(f"用户{user_id}超出ports_per_user范围 (长度={len(self.srs_config.ports_per_user)})")
            if user_id >= len(self.srs_config.cyclic_shifts):
                raise RuntimeError(f"用户{user_id}超出cyclic_shifts范围 (长度={len(self.srs_config.cyclic_shifts)})")
                
            num_ports = self.srs_config.ports_per_user[user_id]
            print(f"   UE {user_id}: {num_ports} 端口")
            
            # 预留：可以为每个UE创建专用的处理实例
            self.per_ue_channels[user_id] = {
                'num_ports': num_ports,
                'cyclic_shifts': self.srs_config.cyclic_shifts[user_id],
                # 'dedicated_channel': None,  # 如果需要per-UE信道实例
                # 'dedicated_generator': None,  # 如果需要per-UE生成器
            }
    
    def _initialize_per_port_instances(self):
        """为每个port初始化专用实例（如果需要）"""
        # 当前设计使用统一的数据生成器，per-port实例在内部处理
        # 如果将来需要per-port的特殊处理，可以在这里添加
        
        for user_id in range(self.srs_config.num_users):
            # 🔧 再次边界检查，确保安全
            if user_id >= len(self.srs_config.ports_per_user):
                continue  # 跳过无效用户
                
            for port_id in range(self.srs_config.ports_per_user[user_id]):
                # 🔧 检查cyclic_shifts边界
                if (user_id >= len(self.srs_config.cyclic_shifts) or 
                    port_id >= len(self.srs_config.cyclic_shifts[user_id])):
                    print(f"   ⚠️  Port {user_id}:{port_id} 循环移位配置缺失，跳过")
                    continue
                    
                port_key = f"ue_{user_id}_port_{port_id}"
                cyclic_shift = self.srs_config.cyclic_shifts[user_id][port_id]
                
                print(f"   Port {port_key}: 循环移位 {cyclic_shift}")
                
                # 预留：可以为每个port创建专用的处理实例
                self.per_port_generators[port_key] = {
                    'user_id': user_id,
                    'port_id': port_id,
                    'cyclic_shift': cyclic_shift,
                    # 'dedicated_sequence_gen': None,  # 如果需要per-port序列生成器
                    # 'dedicated_mapper': None,  # 如果需要per-port映射器
                }
    
    def _print_instance_summary(self):
        """打印实例化总结"""
        print(f"\n� 实例化总结:")
        print(f"   信道模型: {len(self.channel_models)} 个")
        for key, model in self.channel_models.items():
            status = "✅" if model is not None else "❌"
            print(f"     {status} {key}")
        
        print(f"   数据生成器: {len(self.data_generators)} 个")
        for key, generator in self.data_generators.items():
            status = "✅" if generator is not None else "❌"
            print(f"     {status} {key}")
        
        print(f"   UE实例: {len(self.per_ue_channels)} 个")
        print(f"   Port实例: {len(self.per_port_generators)} 个")
    
    def get_data_generator(self, channel_config=None):
        """
        获取数据生成器实例
        
        🎯 优化设计：优先使用已初始化的数据生成器，避免重复创建
        
        Args:
            channel_config: 信道配置，格式：{'model': 'TDL-A', 'delay_spread': 300e-9}
            
        Returns:
            SRSDataGenerator实例
        """
        # 使用唯一的数据生成器键名
        snr_key = "config_snr"
        
        # 检查是否有预创建的生成器
        if snr_key in self.data_generators and self.data_generators[snr_key] is not None:
            return self.data_generators[snr_key]
        
        # 如果没有找到预创建的生成器，首先检查是否有与请求配置匹配的信道模型
        if channel_config is None:
            # 使用默认配置
            channel_config = self.common_channel_configs[0] if hasattr(self, 'common_channel_configs') and self.common_channel_configs else {
                'model': 'TDL-A',
                'delay_spread': self.system_config.delay_spread if hasattr(self, 'system_config') else 300e-9
            }

        # 构建信道模型的键名
        channel_key = f"{channel_config['model']}_{channel_config['delay_spread']*1e9:.0f}ns"
        
        # 检查信道模型是否存在
        if channel_key not in self.channel_models or self.channel_models[channel_key] is None:
            print(f"📢 请求的信道模型 {channel_key} 不存在，需要创建")
            # 确保信道模型被创建
            from professional_channels import SIONNAChannelModel
            try:
                channel_model = SIONNAChannelModel(
                    system_config=self.system_config,
                    model_type=channel_config['model'],
                    num_rx_antennas=self.system_config.num_rx_antennas,
                    delay_spread=channel_config['delay_spread'],
                    device=self.channel_params['device']
                )
                self.channel_models[channel_key] = channel_model
                print(f"✅ 成功创建信道模型: {channel_key}")
            except Exception as e:
                print(f"❗ 创建信道模型 {channel_key} 时出错: {e}")
                # 使用任何已存在的有效模型
                available_models = [model for k, model in self.channel_models.items() if model is not None]
                if not available_models:
                    # 如果没有可用的模型，强制创建一个TDL-A
                    print(f"📢 没有可用的信道模型，创建默认TDL-A模型")
                    try:
                        channel_model = SIONNAChannelModel(
                            system_config=self.system_config,
                            model_type="TDL-A",
                            num_rx_antennas=self.system_config.num_rx_antennas,
                            delay_spread=30e-9,
                            device=self.channel_params['device']
                        )
                        self.channel_models["TDL-A_30ns"] = channel_model
                        channel_key = "TDL-A_30ns"  # 更新当前使用的键名
                    except Exception as inner_e:
                        # 如果所有尝试都失败，打印错误但继续执行
                        print(f"⚠️ 创建默认TDL-A模型失败: {inner_e}")
                        print(f"📢 将创建无信道模型的数据生成器")
                        channel_model = None
                else:
                    # 使用第一个可用的模型
                    channel_model = available_models[0]
                    found_key = [k for k, v in self.channel_models.items() if v == channel_model][0]
                    print(f"✅ 使用已有的信道模型代替: {found_key}")
                    self.channel_models[channel_key] = channel_model  # 确保请求的键名也有对应的模型
        else:
            # 获取信道模型
            channel_model = self.channel_models[channel_key]
        
        # 创建新的数据生成器
        from data_generator_refactored import SRSDataGenerator
        generator = SRSDataGenerator(
            config=self.signal_gen_params['srs_config'],
            channel_model=channel_model,
            num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
            sampling_rate=self.signal_gen_params['sampling_rate'],
            device=self.signal_gen_params['device']
        )
        
        # 缓存新创建的生成器
        self.data_generators[snr_key] = generator
        print(f"✅ 创建并缓存新数据生成器: {snr_key} (using_channel={generator.using_channel})")
        
        return generator
    
    def get_channel_model(self, model_type="TDL-A", delay_spread=None):
        """
        获取信道模型实例
        
        🎯 优化设计：确保总是返回一个有效的信道模型，不会返回None或抛出异常
        
        Args:
            model_type: 信道模型类型
            delay_spread: 延迟扩展（如果为None则使用系统配置）
            
        Returns:
            SIONNAChannelModel实例
        """
        if delay_spread is None:
            delay_spread = self.system_config.delay_spread
        
        # 构建请求的信道模型键名
        channel_key = f"{model_type}_{delay_spread*1e9:.0f}ns"
        
        # 检查是否有匹配的预创建信道模型
        if channel_key in self.channel_models and self.channel_models[channel_key] is not None:
            return self.channel_models[channel_key]
        
        # 如果没有找到精确匹配的模型，尝试创建请求的模型
        print(f"📢 请求的信道模型 {channel_key} 不存在，尝试创建")
        
        try:
            from professional_channels import SIONNAChannelModel
            
            channel_model = SIONNAChannelModel(
                system_config=self.system_config,
                model_type=model_type,
                num_rx_antennas=self.system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.channel_params['device']
            )
            
            # 缓存新创建的信道模型
            self.channel_models[channel_key] = channel_model
            print(f"✅ 成功创建信道模型: {channel_key}")
            
            return channel_model
        except Exception as e:
            print(f"⚠️ 创建信道模型 {channel_key} 失败: {e}")
            
            # 尝试使用任何现有的模型
            available_models = [model for k, model in self.channel_models.items() if model is not None]
            if available_models:
                # 使用第一个可用的模型
                channel_model = available_models[0]
                found_key = [k for k, v in self.channel_models.items() if v == channel_model][0]
                print(f"✅ 使用现有信道模型: {found_key}")
                
                # 同时缓存这个模型到请求的键名下，以便下次直接使用
                self.channel_models[channel_key] = channel_model
                
                return channel_model
        
        # 如果所有尝试都失败，创建一个TDL-A备用模型
        print(f"📢 创建TDL-A_30ns备用信道模型")
        
        from professional_channels import SIONNAChannelModel
        backup_model = SIONNAChannelModel(
            system_config=self.system_config,
            model_type="TDL-A",
            num_rx_antennas=self.system_config.num_rx_antennas,
            delay_spread=30e-9,
            device=self.channel_params['device']
        )
        
        # 缓存备用模型到所有需要的键名
        backup_key = "TDL-A_30ns"
        self.channel_models[backup_key] = backup_model
        self.channel_models[channel_key] = backup_model  # 同时缓存到请求的键名
        
        print(f"✅ 使用备用信道模型: {backup_key}")
        return backup_model
            
            
    def generate_batch_with_dynamic_channel(self, batch_size: int, enable_debug: bool = False, channel_config=None):
        """
        动态生成批次数据，包含完整的信号生成、信道应用和LS估计流程
        
        🎯 优化设计：优先使用已初始化的数据生成器，避免重复创建
        
        Args:
            batch_size: 批次大小
            enable_debug: 是否启用调试
            channel_config: 信道配置，格式：{'model': 'TDL-A', 'delay_spread': 300e-9}
        """
        # 检查是否有预初始化的生成器
        snr_key = "config_snr"
        if snr_key in self.data_generators and self.data_generators[snr_key] is not None:
            data_generator = self.data_generators[snr_key]
            print(f"🔄 使用预初始化的数据生成器: {snr_key}") if enable_debug else None
        else:
            # 如果没有找到预初始化的生成器，则创建一个
            data_generator = self.get_data_generator(channel_config=channel_config)
            print(f"🆕 创建新的数据生成器") if enable_debug else None
        
        # 生成完整批次（包含ls_estimates, true_channels等）
        batch = data_generator.generate_batch(batch_size, enable_debug=enable_debug)
        
        return batch
    
    def train_epoch(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        """
        Train for one epoch - 完全批处理化版本
        
        Args:
            num_batches: Number of batches
            batch_size: Batch size
            
        Returns:
            Average loss and NMSE for the epoch
        """
        print("\n====== 开始训练epoch (批处理模式) ======")
        total_loss = 0
        total_nmse = 0
        total_sample_count = 0
        
        # Set models to training mode
        if self.mmse_module:
            self.mmse_module.train()
        
        for batch_idx in tqdm(range(num_batches), desc="训练中"):
            # Generate batch with dynamic channel - 使用已初始化的数据生成器
            with torch.no_grad():
                # 直接使用data_generators中的预创建实例，不指定特定信道配置
                batch = self.generate_batch_with_dynamic_channel(
                    batch_size, 
                    enable_debug=True
                )

            # Get batch data - 现在是列表形式
            ls_estimates = batch['ls_estimates']
            true_channels_dict = batch['true_channels']
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # 一次性处理整个批次的所有用户端口
            estimated_channels_dict = self.srs_estimator(
                ls_estimates=ls_estimates,
                user_config=self.srs_config,
                true_channels_dict=true_channels_dict # for debug purpose
            )
            
            # 批处理化计算损失和NMSE
            batch_loss, batch_nmse, batch_sample_count = self.compute_batch_loss_and_nmse(
                estimated_channels_dict, 
                true_channels_dict,
                is_training=True  # 训练模式
            )
            
            # 反向传播和优化
            if batch_loss.requires_grad:
                batch_loss.backward()
                
                # 梯度信息调试
                for name, param in self.srs_estimator.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"SRS参数 {name} 没有梯度")
                        elif param.grad.abs().mean().item() == 0:
                            print(f"SRS参数 {name} 的梯度全为零")
                        else:
                            grad_norm = param.grad.abs().mean().item()
                            if self.writer is not None:
                                self.writer.add_scalar(f'Gradients/SRS_{name}', grad_norm, self.global_step)
                
                # 执行梯度更新
                self.optimizer.step()
            else:
                print(f"警告：批次 {batch_idx} 的损失不需要梯度。跳过反向传播。")
            
            # Update totals
            with torch.no_grad():
                batch_loss_value = batch_loss.item()
                total_loss += batch_loss_value
                total_nmse += batch_nmse
                total_sample_count += batch_sample_count
                
                # 计算平均NMSE
                avg_batch_nmse = batch_nmse / batch_sample_count if batch_sample_count > 0 else 0
                  
                # Log batch-level metrics to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/batch', batch_loss_value, self.global_step)
                    self.writer.add_scalar('NMSE/batch', avg_batch_nmse, self.global_step)
                  
                # Print loss information to console for immediate feedback
                print(f"\n批次 [{batch_idx+1:03d}/{num_batches:03d}] - 损失: {batch_loss_value:.6f}, NMSE: {avg_batch_nmse:.2f} dB, 样本数: {batch_sample_count}")
                
                self.global_step += 1
        
        # Calculate averages
        avg_loss = total_loss / total_sample_count if total_sample_count > 0 else 0
        avg_nmse = total_nmse / total_sample_count if total_sample_count > 0 else 0
        
        return avg_loss, avg_nmse
        
    def validate(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        """
        Validate the model - 完全批处理化版本
        
        Args:
            num_batches: Number of batches
            batch_size: Batch size
            
        Returns:
            Average loss and NMSE for validation
        """
        print("\n====== 开始验证 (批处理模式) ======")
        total_loss = 0
        total_nmse = 0
        total_sample_count = 0
        self.srs_estimator.eval()
        if self.mmse_module:
            self.mmse_module.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="验证中"):
                # Generate batch with dynamic channel (使用配置文件中的SNR范围)
                batch = self.generate_batch_with_dynamic_channel(batch_size)
                
                # Get batch data - 现在是列表形式
                ls_estimates_dict = batch['ls_estimates']
                true_channels_dict = batch['true_channels']
                
                # 一次性处理整个批次的所有用户端口
                estimated_channels_dict = self.srs_estimator(
                    ls_estimates=ls_estimates_dict,
                    user_config=self.srs_config
                )
                
                # 批处理化计算损失和NMSE
                batch_loss, batch_nmse, batch_sample_count = self.compute_batch_loss_and_nmse(
                    estimated_channels_dict, 
                    true_channels_dict,
                    is_training=False  # 验证模式
                )
                
                # Update totals
                total_loss += batch_loss.item()
                total_nmse += batch_nmse
                total_sample_count += batch_sample_count
                
        # Calculate averages for the entire validation set
        avg_loss = total_loss / total_sample_count if total_sample_count > 0 else 0
        avg_nmse = total_nmse / total_sample_count if total_sample_count > 0 else 0
        
        return avg_loss, avg_nmse

    def train(self, num_epochs: int, num_batches: int, batch_size: int, 
              val_batches: int, val_every_n_epochs: int = 1, 
              save_every_n_epochs: int = 5) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs
        
        Args:
            num_epochs: Number of epochs to train for
            num_batches: Number of batches per epoch during training
            batch_size: Batch size for training
            val_batches: Number of batches for validation
            val_every_n_epochs: Validate every n epochs
            save_every_n_epochs: Save checkpoint every n epochs
            
        Returns:
            Dictionary with training history
        """
        print(f"\n====== 开始训练 ({num_epochs}轮) ======")
        
        best_val_nmse = float('inf')
        best_epoch = -1
        
        for epoch in range(num_epochs):
            print(f"\n====== Epoch {epoch+1}/{num_epochs} ======")
            
            # Train for one epoch
            train_loss, train_nmse = self.train_epoch(num_batches, batch_size)
            self.train_losses.append(train_loss)
            self.train_nmse.append(train_nmse)
            
            # Log training metrics to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('NMSE/train', train_nmse, epoch)
            
            # Validate if needed
            if (epoch + 1) % val_every_n_epochs == 0:
                val_loss, val_nmse = self.validate(val_batches, batch_size)
                self.val_losses.append(val_loss)
                self.val_nmse.append(val_nmse)
                
                # Log validation metrics to TensorBoard
                if self.writer is not None:
                    self.writer.add_scalar('Loss/val', val_loss, epoch)
                    self.writer.add_scalar('NMSE/val', val_nmse, epoch)
                
                # Check if this is the best model so far
                if val_nmse < best_val_nmse:
                    best_val_nmse = val_nmse
                    best_epoch = epoch
                    # Save best model
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"新的最佳模型! NMSE: {val_nmse:.6f} dB (epoch {epoch+1})")
                
                print(f"验证损失: {val_loss:.6f}, 验证NMSE: {val_nmse:.2f} dB")
            
            # Save checkpoint if needed
            if (epoch + 1) % save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
            
            # Print epoch summary
            print(f"轮次 {epoch+1} 训练损失: {train_loss:.6f}, 训练NMSE: {train_nmse:.2f} dB")
        
        print("\n====== 训练完成 ======")
        if best_epoch >= 0:
            print(f"最佳模型在epoch {best_epoch+1}, NMSE: {best_val_nmse:.2f} dB")
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_nmse': self.train_nmse,
            'val_nmse': self.val_nmse
        }
        
        return history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save a checkpoint of the model
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.srs_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_nmse': self.train_nmse,
            'val_nmse': self.val_nmse
        }
        
        if self.mmse_module:
            checkpoint['mmse_state_dict'] = self.mmse_module.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"保存检查点到: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.srs_estimator.load_state_dict(checkpoint['model_state_dict'])
        
        if self.mmse_module and 'mmse_state_dict' in checkpoint:
            self.mmse_module.load_state_dict(checkpoint['mmse_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'train_nmse' in checkpoint:
            self.train_nmse = checkpoint['train_nmse']
        if 'val_nmse' in checkpoint:
            self.val_nmse = checkpoint['val_nmse']
        
        epoch = checkpoint.get('epoch', -1)
        print(f"成功加载检查点，epoch {epoch+1}")
        
        return epoch

    def set_dynamic_training_params(self, channel_model=None, delay_spread=None):
        """
        动态调整训练参数
        
        允许在训练过程中调整信道模型等参数，
        实现课程学习 (Curriculum Learning) 等高级训练策略。
        
        注意：SNR范围统一从配置文件获取，不支持动态修改以避免不一致
        
        Args:
            channel_model: 新的信道模型，例如 "TDL-B"
            delay_spread: 新的延迟扩展，例如 500e-9
        """
        params_changed = False
                
        if channel_model is not None:
            old_model = self.channel_params['channel_model']
            self.channel_params['channel_model'] = channel_model
            print(f"🔧 更新信道模型: {old_model} -> {channel_model}")
            params_changed = True
            
        if delay_spread is not None:
            old_delay = self.channel_params['delay_spread']
            self.channel_params['delay_spread'] = delay_spread
            print(f"🔧 更新延迟扩展: {old_delay*1e9:.1f} ns -> {delay_spread*1e9:.1f} ns")
            params_changed = True
        
        # 如果信道参数变化，检查是否需要创建新的信道模型
        if params_changed:
            # 创建新的信道模型键名
            new_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
            
            # 检查该信道模型是否已存在
            if new_channel_key not in self.channel_models or self.channel_models[new_channel_key] is None:
                print(f"🔄 创建新的信道模型: {new_channel_key}")
                try:
                    from professional_channels import SIONNAChannelModel
                    new_channel_model = SIONNAChannelModel(
                        system_config=self.system_config,
                        model_type=self.channel_params['channel_model'],
                        num_rx_antennas=self.system_config.num_rx_antennas,
                        delay_spread=self.channel_params['delay_spread'],
                        device=self.channel_params['device']
                    )
                    self.channel_models[new_channel_key] = new_channel_model
                    print(f"✅ 新信道模型创建成功: {new_channel_key}")
                except Exception as e:
                    print(f"⚠️ 新信道模型创建失败: {e}，将使用现有模型")
                    # 使用任何可用的模型
                    available_models = [model for k, model in self.channel_models.items() if model is not None]
                    if available_models:
                        self.channel_models[new_channel_key] = available_models[0]
            
            # 重置数据生成器，以便使用新的信道配置
            if "config_snr" in self.data_generators:
                del self.data_generators["config_snr"]
                print(f"🔄 数据生成器已重置，将使用新的信道配置")
    
    def add_custom_channel_config(self, model_type, delay_spread=None):
        """
        添加新的信道配置并预创建对应的信道模型
        
        Args:
            model_type: 信道模型类型，例如 "TDL-D"
            delay_spread: 延迟扩展，例如 500e-9（如果为None则使用系统配置）
        """
        if delay_spread is None:
            delay_spread = self.system_config.delay_spread
        
        channel_key = f"{model_type}_{delay_spread*1e9:.0f}ns"
        
        if channel_key in self.channel_models and self.channel_models[channel_key] is not None:
            print(f"✅ 信道配置 {channel_key} 已存在")
            return self.channel_models[channel_key]
        
        print(f"🔄 添加新的信道配置: {channel_key}")
        
        if not PROFESSIONAL_CHANNELS_AVAILABLE:
            print(f"⚠️ 专业信道库不可用，无法创建信道模型")
            # 使用任何可用的模型
            available_models = [model for k, model in self.channel_models.items() if model is not None]
            if available_models:
                self.channel_models[channel_key] = available_models[0]
                print(f"✅ 使用现有信道模型替代")
                return self.channel_models[channel_key]
            return None
        
        try:
            from professional_channels import SIONNAChannelModel
            
            channel_model = SIONNAChannelModel(
                system_config=self.system_config,
                model_type=model_type,
                num_rx_antennas=self.system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.channel_params['device']
            )
            
            self.channel_models[channel_key] = channel_model
            print(f"✅ 信道模型 {channel_key} 创建成功")
            
            # 添加到common_channel_configs
            self.common_channel_configs.append({
                'model': model_type,
                'delay_spread': delay_spread
            })
            
            return channel_model
            
        except Exception as e:
            print(f"⚠️ 信道模型 {channel_key} 创建失败: {e}")
            # 使用任何可用的模型
            available_models = [model for k, model in self.channel_models.items() if model is not None]
            if available_models:
                self.channel_models[channel_key] = available_models[0]
                print(f"✅ 使用现有信道模型替代")
                return self.channel_models[channel_key]
            return None
    
    def get_current_params(self):
        """
        获取当前的动态参数设置和实例状态
        
        Returns:
            dict: 当前参数字典和实例状态
        """
        return {
            'signal_params': self.signal_gen_params,
            'channel_params': self.channel_params,
            'system_config': self.system_config,
            'instance_status': {
                'channel_models': list(self.channel_models.keys()),
                'data_generators': list(self.data_generators.keys()),
                'per_ue_channels': len(self.per_ue_channels),
                'per_port_generators': len(self.per_port_generators),
            },
            'instance_health': {
                'healthy_channel_models': sum(1 for m in self.channel_models.values() if m is not None),
                'healthy_data_generators': sum(1 for g in self.data_generators.values() if g is not None),
                'total_channel_models': len(self.channel_models),
                'total_data_generators': len(self.data_generators),
            }
        }
    
    def compute_batch_loss_and_nmse(self, 
                                    estimated_channels_list: List[Dict[Tuple[int, int], torch.Tensor]],
                                    true_channels_list: List[Dict[Tuple[int, int], torch.Tensor]],
                                    is_training: bool = True
                                    ) -> Tuple[torch.Tensor, float, int]:
        """
        批处理化的损失和NMSE计算 - 适配列表格式
        
        Args:
            estimated_channels_list: 估计信道列表，长度为batch_size，每个元素是Dict[(user_id, port_id), tensor]
                                   每个tensor形状: [num_rx_ant, seq_length]
            true_channels_list: 真实信道列表，长度为batch_size，每个元素是Dict[(user_id, port_id), tensor]
                               每个tensor形状: [num_rx_ant, seq_length]
            is_training: 是否在训练模式（验证时不检查梯度）
        
        Returns:
            Tuple[总损失标量, 总NMSE, 样本数量]
        """
        if is_training:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        total_nmse = 0.0
        sample_count = 0
        
        batch_size = len(estimated_channels_list)
        
        # 遍历每个batch样本
        for batch_idx in range(batch_size):
            est_dict = estimated_channels_list[batch_idx]    # Dict[(user_id, port_id), tensor]
            true_dict = true_channels_list[batch_idx]        # Dict[(user_id, port_id), tensor]
            
            # 遍历该样本的每个用户端口
            for user_port_key in est_dict.keys():
                if user_port_key not in true_dict:
                    continue
                    
                est_channels = est_dict[user_port_key]      # [num_rx_ant, seq_length]
                true_channels = true_dict[user_port_key]    # [num_rx_ant, seq_length]
                
                # 只在训练时检查梯度
                if is_training and not est_channels.requires_grad:
                    print(f"警告：估计的信道在batch {batch_idx} 用户端口 {user_port_key} 不需要梯度")
                    continue
                
                # 计算实部和虚部的MSE损失（对所有维度求平均）
                real_loss = torch.mean((torch.real(est_channels) - torch.real(true_channels))**2)
                imag_loss = torch.mean((torch.imag(est_channels) - torch.imag(true_channels))**2)
                
                # 累积损失
                channel_loss = real_loss + imag_loss
                total_loss = total_loss + channel_loss
                
                # 计算NMSE（仅用于监控，不需要梯度）
                with torch.no_grad():
                    num_rx_ant, seq_length = est_channels.shape
                    for ant_idx in range(num_rx_ant):
                        est_channel = est_channels[ant_idx, :]     # [seq_length]
                        true_channel = true_channels[ant_idx, :]   # [seq_length]
                        
                        nmse = calculate_nmse(true_channel, est_channel)
                        total_nmse += nmse
                        sample_count += 1
        
        return total_loss, total_nmse, sample_count
    

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SRS Channel Estimator with professional channel models")
    parser.add_argument('--test', action='store_true', help='Run standalone test')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--train_batches', type=int, default=50, help='Number of training batches per epoch')
    parser.add_argument('--val_batches', type=int, default=10, help='Number of validation batches')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every n epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every n epochs')
    parser.add_argument('--no_mmse', action='store_true', help='Disable trainable MMSE')
    parser.add_argument('--enable_plotting', action='store_true', help='Enable plotting')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_modified', help='Save directory')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Load checkpoint file')
    
    # Channel model arguments
    parser.add_argument('--channel_model', type=str, default='TDL-C', 
                       choices=['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'],
                       help='Channel model type')
    parser.add_argument('--use_sionna', action='store_true', default=True,
                       help='Use SIONNA professional channel models (default: True)')
    parser.add_argument('--use_custom_channels', action='store_true',
                       help='Force use of custom channel implementation')
    parser.add_argument('--delay_spread', type=float, default=300e-9,
                       help='Channel delay spread in seconds (default: use system config)')
    parser.add_argument('--carrier_frequency', type=float, default=3.5e9,
                       help='Carrier frequency in Hz')
    
    args = parser.parse_args()
    
    # Run test if requested
    if args.test:
        return test_standalone_trainer()
    
    # Override settings if custom is explicitly requested
    if args.use_custom_channels:
        args.use_sionna = False
    
    # Create configuration
    srs_config = create_example_config()
    
    # 如果delay_spread为None，使用系统配置的默认值
    if args.delay_spread is None:
        from system_config import create_default_system_config
        system_config = create_default_system_config()
        args.delay_spread = system_config.delay_spread
    
    # Print configuration summary
    print("\n" + "="*60)
    print("🔧 TRAINING CONFIGURATION")
    print("="*60)
    print(f"Channel Model: {args.channel_model}")
    print(f"Use SIONNA: {args.use_sionna and SIONNA_AVAILABLE}")
    print(f"Delay Spread: {args.delay_spread*1e9:.1f} ns (from {'system config' if args.delay_spread else 'command line'})")
    print(f"Carrier Frequency: {args.carrier_frequency/1e9:.1f} GHz")
    print(f"Trainable MMSE: {not args.no_mmse}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Create trainer
    trainer = SRSTrainerModified(
        srs_config=srs_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=args.save_dir,
        use_trainable_mmse=not args.no_mmse,
        enable_plotting=args.enable_plotting,
        use_professional_channels=True,
        use_sionna=args.use_sionna
    )
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.load_checkpoint:
        start_epoch = trainer.load_checkpoint(args.load_checkpoint) + 1
    
    # Train the model
    history = trainer.train(
        num_epochs=args.epochs,
        num_batches=args.train_batches,
        batch_size=args.batch_size,
        val_batches=args.val_batches,
        val_every_n_epochs=args.val_every,
        save_every_n_epochs=args.save_every
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    if history['val_loss']:
        plt.plot(range(0, len(history['val_loss']) * args.val_every, args.val_every), 
                 history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot NMSE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_nmse'], label='Train')
    if history['val_nmse']:
        plt.plot(range(0, len(history['val_nmse']) * args.val_every, args.val_every), 
                 history['val_nmse'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('NMSE (dB)')
    plt.title('Training and Validation NMSE')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(args.save_dir, 'training_history.png')
    plt.savefig(plot_path)
    print(f"保存训练历史图到: {plot_path}")
    
    plt.close()

def test_standalone_trainer():
    """Test function to verify standalone operation"""
    print("🧪 Testing standalone trainer...")
    
    # Create configuration
    config = create_example_config()
    
    # Create trainer
    trainer = SRSTrainerModified(
        config=config,
        device="cpu",  # Use CPU for testing
        batch_size=4,
        use_tensorboard=False,  # Disable for testing
        use_trainable_mmse=True,
        use_professional_channels=True,
        use_sionna=True
    )
    
    print("✅ Trainer created successfully")
    
    # Test batch generation
    try:
        batch = trainer.generate_batch_with_dynamic_channel(batch_size=2)
        print(f"✅ Batch generation successful: {list(batch.keys())}")
    except Exception as e:
        print(f"❌ Batch generation failed: {e}")
        return False
    
    # Test training for one mini-batch
    try:
        loss, nmse = trainer.train_epoch(num_batches=2, batch_size=2)
        print(f"✅ Training epoch successful: loss={loss:.6f}, nmse={nmse:.2f}")
    except Exception as e:
        print(f"❌ Training epoch failed: {e}")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == '__main__':
    print(f"PyTorch MKL-DNN available: {torch.backends.mkldnn.is_available()}")
    print(f"PyTorch using MKL-DNN: {torch.backends.mkldnn.enabled}")
    main()
