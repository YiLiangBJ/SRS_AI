import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Try to import professional channel libraries
try:
    import sionna
    SIONNA_AVAILABLE = True
    print("✅ SIONNA available - using professional 3GPP channel models")
except ImportError:
    SIONNA_AVAILABLE = False
    print("❌ SIONNA not available - using custom TDL implementation")
    print("   To install SIONNA (Intel网络): python -m pip install --proxy http://child-prc.intel.com:913 sionna tensorflow")
    print("   To install SIONNA (其他网络): python -m pip install sionna tensorflow")

# Import professional channel model wrapper
try:
    from professional_channels import SIONNAChannelModel, SIONNAChannelGenerator, print_sionna_info
    PROFESSIONAL_CHANNELS_AVAILABLE = True
    print("✅ Professional channel wrapper available")
except ImportError:
    PROFESSIONAL_CHANNELS_AVAILABLE = False
    print("❌ Professional channel wrapper not found")

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
        srs_config: SRSConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints_modified",
        use_trainable_mmse: bool = True,
        enable_plotting: bool = False,
        use_professional_channels: bool = True,
        use_sionna: bool = True
    ):
        """
        Initialize the trainer
        
        Args:
            srs_config: SRS configuration
            device: Computation device
            save_dir: Directory for saving checkpoints
            use_trainable_mmse: Whether to use trainable MMSE matrices
            enable_plotting: Whether to enable plotting during training
            use_professional_channels: Whether to use professional channel libraries
            use_sionna: Whether to use SIONNA (if available)
        """
        self.srs_config = srs_config
        self.device = device
        self.save_dir = save_dir
        self.use_trainable_mmse = use_trainable_mmse
        self.enable_plotting = enable_plotting
        self.use_professional_channels = use_professional_channels
        
        # 保存系统配置
        from system_config import create_default_system_config
        self.system_config = create_default_system_config()
        
        # 保存信号生成参数（运行时动态使用）
        self.signal_gen_params = {
            'srs_config': srs_config,
            'num_rx_antennas': self.system_config.num_rx_antennas,
            'sampling_rate': self.system_config.sampling_rate,
            'device': device
        }
        
        # 保存信道参数（运行时动态使用）
        self.channel_params = {
            'use_sionna': use_sionna,
            'channel_model': "TDL-A",
            'delay_spread': self.system_config.delay_spread,  # 🎯 使用系统配置的延迟扩展
            'carrier_frequency': 3.5e9,
            'device': device
        }

        # Create trainable MMSE module if needed
        if use_trainable_mmse:
            self.mmse_module = TrainableMMSEModule(
                seq_length=srs_config.seq_length,
                mmse_block_size=srs_config.mmse_block_size,
                use_complex_input=True
            ).to(device)
        else:
            self.mmse_module = None
            
        # Create models
        self.srs_estimator = SRSChannelEstimator(
            seq_length=srs_config.seq_length,
            ktc=srs_config.ktc,
            max_users=srs_config.num_users,
            max_ports_per_user=max(srs_config.ports_per_user),
            mmse_block_size=srs_config.mmse_block_size,
            device=device,
            mmse_module=self.mmse_module if use_trainable_mmse else None  # 传入 MMSE 模块
        ).to(device)

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
        os.makedirs(save_dir, exist_ok=True)
        
        # Create logs directory for TensorBoard
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
          
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        
        # Global step counter for logging
        self.global_step = 0
        
        # 🎯 完整实例化所有需要的组件
        print(f"\n🚀 开始完整实例化所有组件...")
        
        # 1. 创建信道模型字典（按不同参数组织）
        self.channel_models = {}
        
        # 2. 创建数据生成器字典（按不同SNR范围组织）
        self.data_generators = {}
        
        # 3. 创建per-UE信道实例字典（如果需要）
        self.per_ue_channels = {}
        
        # 4. 创建per-port信号生成器字典（如果需要）
        self.per_port_generators = {}
        
        # 5. 预定义常用的参数组合
        self.common_snr_ranges = [
            (-10, 40),  # 默认范围
            (-5, 30),   # 中等范围
            (0, 20),    # 高SNR范围
            (-15, 35),  # 扩展范围
        ]
        
        self.common_channel_configs = [
            {'model': 'TDL-A', 'delay_spread': self.system_config.delay_spread},
            {'model': 'TDL-B', 'delay_spread': self.system_config.delay_spread},
            {'model': 'TDL-C', 'delay_spread': self.system_config.delay_spread},
        ]
        
        # 执行完整初始化
        self._initialize_all_instances()
            
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
            
        for config in self.common_channel_configs:
            config_key = f"{config['model']}_{config['delay_spread']*1e9:.0f}ns"
            
            try:
                print(f"   创建信道模型: {config_key}")
                channel_model = SIONNAChannelModel(
                    system_config=self.system_config,
                    model_type=config['model'],
                    num_rx_antennas=self.system_config.num_rx_antennas,
                    delay_spread=config['delay_spread'],
                    device=self.channel_params['device']
                )
                self.channel_models[config_key] = channel_model
                print(f"   ✅ {config_key} 创建成功")
                
            except Exception as e:
                print(f"   ❌ {config_key} 创建失败: {e}")
                self.channel_models[config_key] = None
    
    def _initialize_data_generators(self):
        """为所有常用SNR范围初始化数据生成器"""
        from data_generator_refactored import SRSDataGenerator
        
        # 获取默认信道模型
        default_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
        default_channel_model = self.channel_models.get(default_channel_key)
        
        print(f"   🎯 查找默认信道模型: {default_channel_key}")
        print(f"   🎯 可用信道模型: {list(self.channel_models.keys())}")
        print(f"   🎯 默认信道模型: {'存在' if default_channel_model is not None else '不存在'}")
        
        for snr_range in self.common_snr_ranges:
            snr_key = f"snr_{snr_range[0]}_{snr_range[1]}"
            
            try:
                print(f"   创建数据生成器: {snr_key}")
                data_generator = SRSDataGenerator(
                    config=self.signal_gen_params['srs_config'],
                    channel_model=default_channel_model,  # 🎯 确保传入信道模型
                    num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
                    snr_range=snr_range,
                    sampling_rate=self.signal_gen_params['sampling_rate'],
                    device=self.signal_gen_params['device']
                )
                self.data_generators[snr_key] = data_generator
                print(f"   ✅ {snr_key} 创建成功 (using_channel={data_generator.using_channel})")
                
            except Exception as e:
                print(f"   ❌ {snr_key} 创建失败: {e}")
                self.data_generators[snr_key] = None
    
    def _initialize_per_ue_instances(self):
        """为每个UE初始化专用实例（如果需要）"""
        # 当前设计使用统一的数据生成器，per-UE实例在信道内部处理
        # 如果将来需要per-UE的特殊处理，可以在这里添加
        
        for user_id in range(self.srs_config.num_users):
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
            for port_id in range(self.srs_config.ports_per_user[user_id]):
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
    
    def get_data_generator(self, snr_range=(-10, 40), channel_config=None):
        """
        获取数据生成器实例
        
        🎯 智能选择：优先使用预创建的实例，必要时动态创建
        
        Args:
            snr_range: SNR范围
            channel_config: 信道配置，格式：{'model': 'TDL-A', 'delay_spread': 300e-9}
            
        Returns:
            SRSDataGenerator实例
        """
        # 生成查找键
        snr_key = f"snr_{snr_range[0]}_{snr_range[1]}"
        
        # 检查是否有预创建的生成器
        if snr_key in self.data_generators and self.data_generators[snr_key] is not None:
            generator = self.data_generators[snr_key]
            
            # 检查信道配置是否匹配（如果指定了）
            if channel_config is not None:
                # 这里可以添加信道配置匹配检查
                # 目前使用默认信道配置
                pass
            
            print(f"🎯 使用预创建的数据生成器: {snr_key} (using_channel={generator.using_channel})")
            return generator
        
        # 需要动态创建
        print(f"🔄 动态创建数据生成器: {snr_key}")
        
        # 获取信道模型
        if channel_config is not None:
            channel_key = f"{channel_config['model']}_{channel_config['delay_spread']*1e9:.0f}ns"
            channel_model = self.channel_models.get(channel_key)
            if channel_model is None:
                print(f"⚠️  信道模型 {channel_key} 不存在，使用默认")
                default_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
                channel_model = self.channel_models.get(default_key)
        else:
            default_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
            channel_model = self.channel_models.get(default_key)
        
        try:
            from data_generator_refactored import SRSDataGenerator
            generator = SRSDataGenerator(
                config=self.signal_gen_params['srs_config'],
                channel_model=channel_model,
                num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
                snr_range=snr_range,
                sampling_rate=self.signal_gen_params['sampling_rate'],
                device=self.signal_gen_params['device']
            )
            
            # 缓存新创建的生成器
            self.data_generators[snr_key] = generator
            print(f"✅ 动态生成器创建并缓存: {snr_key} (using_channel={generator.using_channel})")
            
            return generator
            
        except Exception as e:
            print(f"❌ 动态生成器创建失败: {e}")
            # 回退到任何可用的生成器
            for key, gen in self.data_generators.items():
                if gen is not None:
                    print(f"⚠️  回退使用: {key}")
                    return gen
            
            raise RuntimeError(f"无法获取数据生成器: {e}")
    
    def get_channel_model(self, model_type="TDL-A", delay_spread=None):
        """
        获取信道模型实例
        
        Args:
            model_type: 信道模型类型
            delay_spread: 延迟扩展（如果为None则使用系统配置）
            
        Returns:
            SIONNAChannelModel实例
        """
        if delay_spread is None:
            delay_spread = self.system_config.delay_spread
        
        channel_key = f"{model_type}_{delay_spread*1e9:.0f}ns"
        
        if channel_key in self.channel_models and self.channel_models[channel_key] is not None:
            print(f"🎯 使用预创建的信道模型: {channel_key}")
            return self.channel_models[channel_key]
        
        # 动态创建
        print(f"🔄 动态创建信道模型: {channel_key}")
        
        if not PROFESSIONAL_CHANNELS_AVAILABLE:
            print("❌ 专业信道库不可用")
            return None
        
        try:
            channel_model = SIONNAChannelModel(
                system_config=self.system_config,
                model_type=model_type,
                num_rx_antennas=self.system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.channel_params['device']
            )
            
            # 缓存新创建的信道模型
            self.channel_models[channel_key] = channel_model
            print(f"✅ 动态信道模型创建并缓存: {channel_key}")
            
            return channel_model
            
        except Exception as e:
            print(f"❌ 动态信道模型创建失败: {e}")
            return None
            
    def generate_batch_with_dynamic_channel(self, batch_size: int, enable_debug: bool = False, snr_range=(-10, 40), channel_config=None):
        """
        动态生成批次数据，包含完整的信号生成、信道应用和LS估计流程
        
        🎯 高效设计：使用预创建的数据生成器实例，通过字典查找避免重复创建
        这确保我们得到训练所需的所有数据：ls_estimates, true_channels, noise_powers等
        
        Args:
            batch_size: 批次大小
            enable_debug: 是否启用调试
            snr_range: SNR范围
            channel_config: 信道配置，格式：{'model': 'TDL-A', 'delay_spread': 300e-9}
        """
        # 🎯 从预创建的实例字典中获取数据生成器
        data_generator = self.get_data_generator(snr_range=snr_range, channel_config=channel_config)
        
        # 生成完整批次（包含ls_estimates, true_channels等）
        batch = data_generator.generate_batch(batch_size, enable_debug=enable_debug)
        
        if enable_debug:
            print(f"✅ 动态批次生成完成:")
            print(f"   批次大小: {batch_size}")
            print(f"   SNR范围: {snr_range}")
            if channel_config:
                print(f"   信道配置: {channel_config}")
            else:
                print(f"   信道模型: {self.channel_params['channel_model']}")
            print(f"   数据键: {list(batch.keys())}")
        
        return batch
    
    def train_epoch(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            num_batches: Number of batches
            batch_size: Batch size
            
        Returns:
            Average loss and NMSE for the epoch
        """
        print("\n====== 开始训练epoch ======")
        total_loss = 0
        total_nmse = 0
        
        # Set models to training mode
        # self.srs_estimator.train()
        if self.mmse_module:
            self.mmse_module.train()
        
        for batch_idx in tqdm(range(num_batches), desc="训练中"):
            # 🎯 动态SNR策略：可以根据训练进度调整SNR范围
            # 例如：课程学习，从高SNR开始，逐渐降低到低SNR
            current_snr_range = (-10, 40)  # 默认范围
            
            # 可选：实现课程学习策略
            # progress = (batch_idx + 1) / num_batches  # 训练进度 0-1
            # min_snr = -10 + (1 - progress) * 10  # 从0dB开始，逐渐降到-10dB
            # current_snr_range = (min_snr, 40)
            
            # Generate batch with dynamic channel and SNR
            with torch.no_grad():
                batch = self.generate_batch_with_dynamic_channel(
                    batch_size, 
                    enable_debug=True, 
                    snr_range=current_snr_range
                )

            # Get ls estimates and cyclic shifts
            ls_estimates = batch['ls_estimates']
            noise_powers = batch['noise_powers']
            true_channels = batch['true_channels']
            
            # Clear gradients
            self.optimizer.zero_grad()
              
            # Forward pass - 整个批次的累计损失
            batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            batch_nmse = 0.0
            
            for i in range(batch_size):                
                ls_estimate = ls_estimates[i]                
                noise_power = noise_powers[i].item()  # noise_power是标量，不需要梯度
                  
                # 创建一个新的损失累积器 (requires_grad=True)
                sample_total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                  
                # if self.mmse_module:
                # 运行MLP生成的MMSE滤波器，或者正常运行MMSE估计器
                channel_estimates = self.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=self.srs_config.cyclic_shifts,
                    noise_power=noise_power
                )
                # else:
                #     # 正常运行SRS估计器
                #     channel_estimates = self.srs_estimator(
                #         ls_estimate=ls_estimate,
                #         cyclic_shifts=self.srs_config.cyclic_shifts,
                #         noise_power=noise_power
                #     )
                
                # Calculate loss for each user/port
                idx = 0
                for u in range(self.srs_config.num_users):
                    for p in range(self.srs_config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            true_channel = true_channels[(u, p)][i]
                            est_channel = channel_estimates[idx]
                            
                            # 确保估计信道需要梯度
                            if not est_channel.requires_grad:
                                print(f"警告：估计的信道在批次 {batch_idx}，样本 {i}，用户 {u}，端口 {p} 不需要梯度")
                                continue
                            
                            # 计算实部和虚部的损失
                            real_loss = torch.mean((torch.real(est_channel) - torch.real(true_channel))**2)
                            imag_loss = torch.mean((torch.imag(est_channel) - torch.imag(true_channel))**2)
                              
                            # 此样本的损失
                            sample_loss = real_loss + imag_loss
                              
                            # 使用加法赋值更新样本总损失
                            sample_total_loss = sample_total_loss + sample_loss
                            
                            # Calculate NMSE
                            with torch.no_grad():  # NMSE只用于监控，不需要梯度
                                nmse = calculate_nmse(true_channel, est_channel)
                                batch_nmse += nmse  # 累加NMSE值，稍后在打印时将除以样本数
                            
                        idx += 1
                  
                # 将这个样本的总损失加到批次损失中
                batch_loss = batch_loss + sample_total_loss
            
            # 确保损失是一个需要梯度的标量 - 移到了样本循环外部
            if batch_loss.requires_grad:
                # Backward pass and optimize
                batch_loss.backward()
            
                # 打印所有参数的梯度信息，用于调试
                print("")  # 确保梯度信息打印在新行
                for name, param in self.srs_estimator.named_parameters():
                    if param.requires_grad:
                        if param.grad is None:
                            print(f"SRS参数 {name} 没有梯度")
                        elif param.grad.abs().mean().item() == 0:
                            print(f"SRS参数 {name} 的梯度全为零")
                        else:
                            grad_norm = param.grad.abs().mean().item()
                            print(f"SRS参数 {name} 的梯度范数: {grad_norm:.6f}")
                            self.writer.add_scalar(f'Gradients/SRS_{name}', grad_norm, self.global_step)
                
                # 执行梯度更新
                self.optimizer.step()            
            else:
                print(f"警告：批次 {batch_idx} 的损失不需要梯度。跳过反向传播。")
                print(f"批次损失的类型：{type(batch_loss)}")
                print(f"批次损失的requires_grad：{batch_loss.requires_grad}")            
            
            # Update totals
            with torch.no_grad():
                batch_loss_value = batch_loss.item()
                total_loss += batch_loss_value
                total_nmse += batch_nmse
                
                # 计算这个批次处理了多少个实际计算的样本数
                # 跟踪实际处理了多少个信道样本（每个用户可能有不同数量的端口）
                # 实际计算的样本数 = 批次大小 * 真实信道的总数（考虑到每个用户可能有不同的端口数）
                actual_channel_count = sum(1 for u in range(self.srs_config.num_users) 
                                         for p in range(self.srs_config.ports_per_user[u]) 
                                         if (u, p) in true_channels)
                num_samples_in_batch = batch_size * actual_channel_count
                avg_batch_nmse = batch_nmse / num_samples_in_batch
                  
                # Log batch-level metrics to TensorBoard
                self.writer.add_scalar('Loss/batch', batch_loss_value, self.global_step)
                self.writer.add_scalar('NMSE/batch', avg_batch_nmse, self.global_step)
                  
                # Print loss information to console for immediate feedback
                print(f"\n批次 [{batch_idx+1:03d}/{num_batches:03d}] - 损失: {batch_loss_value:.6f}, NMSE: {avg_batch_nmse:.2f} dB")
                
                self.global_step += 1
          
        # # Calculate averages
        # # 我们需要计算整个epoch处理的总样本数
        # # 每个批次处理的样本数 = 批次大小 * 实际信道数量
        # # 整个epoch处理的总样本数 = 批次数 * 每个批次处理的样本数
        # actual_channel_count = sum(1 for u in range(self.srs_config.num_users) 
        #                          for p in range(self.srs_config.ports_per_user[u]) 
        #                          if (u, p) in true_channels)  # 使用最后一个批次的数据
        num_channels = batch_size * actual_channel_count * num_batches
        avg_loss = total_loss / num_channels
        avg_nmse = total_nmse / num_channels
        
        return avg_loss, avg_nmse
        
    def validate(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        """
        Validate the model
        
        Args:
            num_batches: Number of batches
            batch_size: Batch size
            
        Returns:
            Average loss and NMSE for validation
        """
        print("\n====== 开始验证 ======")
        total_loss = 0
        total_nmse = 0
        
        self.srs_estimator.eval()
        if self.mmse_module:
            self.mmse_module.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="验证中"):
                # Generate batch with dynamic channel (验证时使用固定SNR范围)
                validation_snr_range = (-10, 40)  # 验证时使用标准SNR范围
                batch = self.generate_batch_with_dynamic_channel(
                    batch_size, 
                    snr_range=validation_snr_range
                )
                
                # Get ls estimates and cyclic shifts
                ls_estimates = batch['ls_estimates']
                noise_powers = batch['noise_powers']
                true_channels = batch['true_channels']
                
                # Forward pass
                batch_loss = 0
                batch_nmse = 0
                for i in range(batch_size):
                    ls_estimate = ls_estimates[i]
                    noise_power = noise_powers[i].item()
                    
                    # 验证阶段直接使用当前模型参数进行前向传播
                    # 不需要重新生成MMSE矩阵
                    channel_estimates = self.srs_estimator(
                        ls_estimate=ls_estimate,
                        cyclic_shifts=self.srs_config.cyclic_shifts,
                        noise_power=noise_power
                    )
                    
                    # Calculate loss for each user/port
                    idx = 0
                    for u in range(self.srs_config.num_users):
                        for p in range(self.srs_config.ports_per_user[u]):
                            if (u, p) in true_channels:
                                true_channel = true_channels[(u, p)][i]
                                est_channel = channel_estimates[idx]
                                
                                # Calculate loss
                                real_loss = torch.mean((torch.real(est_channel) - torch.real(true_channel))**2).item()
                                imag_loss = torch.mean((torch.imag(est_channel) - torch.imag(true_channel))**2).item()
                                sample_loss = real_loss + imag_loss
                                
                                batch_loss += sample_loss
                                
                                # Calculate NMSE
                                nmse = calculate_nmse(true_channel, est_channel)
                                batch_nmse += nmse
                                
                            idx += 1
                
                # Update totals
                total_loss += batch_loss
                total_nmse += batch_nmse
                
                # 计算这个批次处理了多少个实际计算的样本数
                actual_channel_count = sum(1 for u in range(self.srs_config.num_users) 
                                        for p in range(self.srs_config.ports_per_user[u]) 
                                        if (u, p) in true_channels)
                
        # Calculate averages for the entire validation set
        num_channels = batch_size * actual_channel_count * num_batches
        avg_loss = total_loss / num_channels
        avg_nmse = total_nmse / num_channels
        
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
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('NMSE/train', train_nmse, epoch)
            
            # Validate if needed
            if (epoch + 1) % val_every_n_epochs == 0:
                val_loss, val_nmse = self.validate(val_batches, batch_size)
                self.val_losses.append(val_loss)
                self.val_nmse.append(val_nmse)
                
                # Log validation metrics to TensorBoard
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

    def set_dynamic_training_params(self, snr_range=None, channel_model=None, delay_spread=None):
        """
        动态调整训练参数
        
        允许在训练过程中调整SNR范围、信道模型等参数，
        实现课程学习 (Curriculum Learning) 等高级训练策略。
        
        🎯 高效更新：利用预创建的实例，只在需要时动态创建新实例
        
        Args:
            snr_range: 新的SNR范围，例如 (-5, 30)
            channel_model: 新的信道模型，例如 "TDL-B"
            delay_spread: 新的延迟扩展，例如 500e-9
        """
        params_changed = False
        
        if snr_range is not None:
            print(f"🔧 更新SNR范围: {snr_range[0]}~{snr_range[1]} dB")
            # SNR范围变化时，检查是否有对应的预创建生成器
            snr_key = f"snr_{snr_range[0]}_{snr_range[1]}"
            if snr_key not in self.data_generators:
                print(f"   新SNR范围，将在下次使用时动态创建")
                
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
            new_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
            if new_channel_key not in self.channel_models:
                print(f"   新信道配置，将在下次使用时动态创建: {new_channel_key}")
    
    def add_custom_snr_range(self, snr_range):
        """
        添加新的SNR范围并预创建对应的数据生成器
        
        Args:
            snr_range: 新的SNR范围，例如 (-20, 50)
        """
        snr_key = f"snr_{snr_range[0]}_{snr_range[1]}"
        
        if snr_key in self.data_generators:
            print(f"⚠️  SNR范围 {snr_key} 已存在")
            return
        
        print(f"🔄 添加新的SNR范围: {snr_key}")
        
        try:
            # 获取默认信道模型
            default_channel_key = f"{self.channel_params['channel_model']}_{self.channel_params['delay_spread']*1e9:.0f}ns"
            default_channel_model = self.channel_models.get(default_channel_key)
            
            from data_generator_refactored import SRSDataGenerator
            data_generator = SRSDataGenerator(
                config=self.signal_gen_params['srs_config'],
                channel_model=default_channel_model,
                num_rx_antennas=self.signal_gen_params['num_rx_antennas'],
                snr_range=snr_range,
                sampling_rate=self.signal_gen_params['sampling_rate'],
                device=self.signal_gen_params['device']
            )
            
            self.data_generators[snr_key] = data_generator
            print(f"✅ 新SNR范围数据生成器创建成功: {snr_key}")
            
        except Exception as e:
            print(f"❌ 新SNR范围数据生成器创建失败: {e}")
            self.data_generators[snr_key] = None
    
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
        
        if channel_key in self.channel_models:
            print(f"⚠️  信道配置 {channel_key} 已存在")
            return
        
        print(f"🔄 添加新的信道配置: {channel_key}")
        
        if not PROFESSIONAL_CHANNELS_AVAILABLE:
            print("❌ 专业信道库不可用")
            return
        
        try:
            channel_model = SIONNAChannelModel(
                system_config=self.system_config,
                model_type=model_type,
                num_rx_antennas=self.system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.channel_params['device']
            )
            
            self.channel_models[channel_key] = channel_model
            print(f"✅ 新信道模型创建成功: {channel_key}")
            
        except Exception as e:
            print(f"❌ 新信道模型创建失败: {e}")
            self.channel_models[channel_key] = None
    
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
    

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SRS Channel Estimator with professional channel models")
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
    parser.add_argument('--channel_model', type=str, default='TDL-A', 
                       choices=['TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'],
                       help='Channel model type')
    parser.add_argument('--use_sionna', action='store_true', default=True,
                       help='Use SIONNA professional channel models (default: True)')
    parser.add_argument('--use_custom_channels', action='store_true',
                       help='Force use of custom channel implementation')
    parser.add_argument('--delay_spread', type=float, default=None,
                       help='Channel delay spread in seconds (default: use system config)')
    parser.add_argument('--carrier_frequency', type=float, default=3.5e9,
                       help='Carrier frequency in Hz')
    
    args = parser.parse_args()
    
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

if __name__ == '__main__':
    print(f"PyTorch MKL-DNN available: {torch.backends.mkldnn.is_available()}")
    print(f"PyTorch using MKL-DNN: {torch.backends.mkldnn.enabled}")
    main()
