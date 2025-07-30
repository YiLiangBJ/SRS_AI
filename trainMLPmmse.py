"""
Trainable MMSE Module and SRS Training Framework

This module provides MLP-based MMSE filtering for SRS channel estimation.
All computations are forced to run on CPU only.
"""

import os

# Force CPU-only execution - disable all CUDA/GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import torch.multiprocessing as mp
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
import json
from tqdm import tqdm
from user_config import SRSConfig

import sionna
SIONNA_AVAILABLE = True
print("SIONNA available - using professional 3GPP channel models")


from professional_channels import SIONNAChannelModel, SIONNAChannelGenerator, print_sionna_info
PROFESSIONAL_CHANNELS_AVAILABLE = True
print("Professional channel wrapper available")


from user_config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model_Traditional import SRSChannelEstimator
from model_AIpart import TrainableMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


class SRSTrainer:
    """
    Modified Trainer for SRS Channel Estimator that uses h_with_residual/phasor as input
    """
    def __init__(
        self,
        use_tensorboard: bool = True,
        log_dir: str = "./logs",
        save_dir: str = "./checkpoints_modified",
        device: str = "cpu",
        lr: float = 0.001
    ):
        self.use_tensorboard = use_tensorboard
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.device = device
        self.lr = lr

        # 每个 batch 实例化并随机化 SRSConfig
        from user_config import create_example_config
        srs_config = create_example_config()
        # 1. 实例化底层数据生成器
        from system_config import create_default_system_config
        system_config = create_default_system_config()
        from data_generator import BaseSRSDataGenerator
        base_generator = BaseSRSDataGenerator(
            srs_config=srs_config,
            system_config=system_config,
            num_rx_antennas=system_config.num_rx_antennas,
            sampling_rate=system_config.sampling_rate,
            device=device
        )

        # 3. 实例化 SRSDataGenerator
        self.data_generator = SRSDataGenerator(
            base_generator=base_generator,
        )

        # 4. 实例化 MMSE 模块
        self.mmse_module = TrainableMMSEModule().to(device)

        # 5. 实例化 SRSChannelEstimator
        self.srs_estimator = SRSChannelEstimator(
            mmse_module=self.mmse_module,
            device=device
        ).to(device)

        # 6.
        self.optimizer = optim.Adam(
            self.mmse_module.parameters(),
            lr=self.lr
        )

        # 7. 目录和 TensorBoard 初始化
        os.makedirs(self.save_dir, exist_ok=True)
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None

        # 8. 初始化训练历史
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        self.global_step = 0
    
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
    
    def train_epoch(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        print("\n====== Starting training epoch (batch processing mode) ======")
        total_loss = 0
        total_nmse = 0
        total_sample_count = 0
        self.mmse_module.train()
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            # 每个 batch 实例化并随机化 SRSConfig
            self.srs_config.randomize_configuration()
            # 解析信道模型参数
            model_type, delay_spread = self.srs_config.parse_channel_model()
            # 实例化 SIONNAChannelModel
            from professional_channels import SIONNAChannelModel
            channel_model = SIONNAChannelModel(
                system_config=self.system_config,
                model_type=model_type,
                num_rx_antennas=self.system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.device
            )
            # 每个 batch 实例化 BaseSRSDataGenerator，确保用最新 srs_config
            from data_generator import BaseSRSDataGenerator, SRSDataGenerator
            base_generator = BaseSRSDataGenerator(
                srs_config=self.srs_config,
                system_config=self.system_config,
                num_rx_antennas=self.system_config.num_rx_antennas,
                sampling_rate=self.system_config.sampling_rate,
                device=self.device
            )
            # 实例化 SRSDataGenerator 并注入信道模型
            data_generator = SRSDataGenerator(base_generator=base_generator)
            data_generator.channel_model = channel_model
            self.data_generator = data_generator
            # Generate batch with dynamic channel
            with torch.no_grad():
                ls_estimates_tensor, true_channel_tensor = self.data_generator.generate_batch(batch_size)

            # Clear gradients
            self.optimizer.zero_grad()

            # Model forward: tensor batch input, output shape (batch_size, num_user_ports, ...)
            estimated_channels = self.srs_estimator(
                ls_estimates=ls_estimates_tensor,
                srs_config=self.srs_config
            )

            # Vectorized loss and NMSE calculation
            # Assume compute_batch_loss_and_nmse supports tensor input
            batch_loss, batch_nmse, batch_sample_count = self.compute_batch_loss_and_nmse(
                estimated_channels,
                true_channel_tensor,
                is_training=True
            )

            # Backpropagation and optimization
            if batch_loss.requires_grad:
                batch_loss.backward()
                # Gradient info
                for name, param in self.srs_estimator.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        grad_norm = param.grad.abs().mean().item()
                        if self.writer is not None:
                            self.writer.add_scalar(f'Gradients/SRS_{name}', grad_norm, self.global_step)
                self.optimizer.step()
            else:
                print(f"Warning: Batch {batch_idx} loss does not require gradients. Skipping backpropagation.")

            # Update totals
            with torch.no_grad():
                batch_loss_value = batch_loss.item()
                total_loss += batch_loss_value
                total_nmse += batch_nmse
                total_sample_count += batch_sample_count
                avg_batch_nmse = batch_nmse / batch_sample_count if batch_sample_count > 0 else 0
                if self.writer is not None:
                    self.writer.add_scalar('Loss/batch', batch_loss_value, self.global_step)
                    self.writer.add_scalar('NMSE/batch', avg_batch_nmse, self.global_step)
                print(f"\nBatch [{batch_idx+1:03d}/{num_batches:03d}] - Loss: {batch_loss_value:.6f}, NMSE: {avg_batch_nmse:.2f} dB, Samples: {batch_sample_count}")
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
        print("\n====== Starting validation (batch processing mode) ======")
        total_loss = 0
        total_nmse = 0
        total_sample_count = 0
        self.srs_estimator.eval()
        if self.mmse_module:
            self.mmse_module.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Validating"):
                # Generate batch with dynamic channel (using SNR range from configuration file)
                batch = self.generate_batch_with_dynamic_channel(batch_size)
                
                # Get batch data - now in list format
                ls_estimates_dict = batch['ls_estimates']
                true_channels_dict = batch['true_channels']
                
                # Process all user ports in the entire batch at once
                estimated_channels_dict = self.srs_estimator(
                    ls_estimates=ls_estimates_dict,
                    user_config=self.srs_config
                )
                
                # Batch processing computation of loss and NMSE
                batch_loss, batch_nmse, batch_sample_count = self.compute_batch_loss_and_nmse(
                    estimated_channels_dict, 
                    true_channels_dict,
                    is_training=False  # Validation mode
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
        print(f"\n====== Starting training ({num_epochs} epochs) ======")
        
        best_val_nmse = float('inf')
        best_epoch = -1
        
        from user_config import create_example_config
        self.srs_config = create_example_config()
        from system_config import create_default_system_config
        self.system_config = create_default_system_config()

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
                    print(f"New best model! NMSE: {val_nmse:.6f} dB (epoch {epoch+1})")
                
                print(f"Validation loss: {val_loss:.6f}, Validation NMSE: {val_nmse:.2f} dB")
            
            # Save checkpoint if needed
            if (epoch + 1) % save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
            
            # Print epoch summary
            print(f"Epoch {epoch+1} training loss: {train_loss:.6f}, training NMSE: {train_nmse:.2f} dB")
        
        print("\n====== Training completed ======")
        if best_epoch >= 0:
            print(f"Best model at epoch {best_epoch+1}, NMSE: {best_val_nmse:.2f} dB")
        
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
        print(f"Saved checkpoint to: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model to: {best_path}")

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
        print(f"Successfully loaded checkpoint, epoch {epoch+1}")
        
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
                                    estimated_channels: torch.Tensor,
                                    true_channels: torch.Tensor,
                                    is_training: bool = True
                                    ) -> Tuple[torch.Tensor, float, int]:
        """
        全tensor化版本，输入 shape: [batch_size, num_rx_ant, seq_length] 或更多 batch 维度
        """
        # loss: 对所有 batch、天线、序列点求平均
        real_loss = torch.mean((torch.real(estimated_channels) - torch.real(true_channels)) ** 2)
        imag_loss = torch.mean((torch.imag(estimated_channels) - torch.imag(true_channels)) ** 2)
        total_loss = real_loss + imag_loss
        # NMSE: 对所有样本、天线分别计算，再取平均
        with torch.no_grad():
            # flatten batch维度，保留天线和seq_length
            batch_dims = estimated_channels.shape[:-2]
            num_rx_ant = estimated_channels.shape[-2]
            seq_length = estimated_channels.shape[-1]
            est_flat = estimated_channels.reshape(-1, num_rx_ant, seq_length)
            true_flat = true_channels.reshape(-1, num_rx_ant, seq_length)
            # nmse: [batch*num_rx_ant]
            mse = torch.mean(torch.abs(est_flat - true_flat) ** 2, dim=-1)  # [batch*num_rx_ant, num_rx_ant]
            power = torch.mean(torch.abs(true_flat) ** 2, dim=-1)           # [batch*num_rx_ant, num_rx_ant]
            nmse = 10 * torch.log10((mse / (power + 1e-8)).mean()).item()
            sample_count = est_flat.numel() // seq_length
        return total_loss, nmse, sample_count
    

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SRS Channel Estimator with professional channel models")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--train_batches', type=int, default=100, help='Number of training batches per epoch')
    parser.add_argument('--val_batches', type=int, default=10, help='Number of validation batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every n epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every n epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_modified', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory for TensorBoard')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Load checkpoint file')
    parser.add_argument('--num_threads', type=int, default=os.cpu_count(), help='Number of CPU threads for PyTorch (default: all cores)')
    # Device argument
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for training (cpu or cuda, default: cpu)')
    args = parser.parse_args()

    # 设置PyTorch多进程启动方式（推荐spawn，避免fork导致的死锁和内存问题）
    mp.set_start_method('spawn', force=True)

    # 设置PyTorch线程数（并行计算核心数）
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)
    print(f"PyTorch CPU线程数设置为: {args.num_threads}")
    
    # Force SIONNA availability check - no fallback
    if not SIONNA_AVAILABLE:
        raise RuntimeError("SIONNA is required but not available. Please install SIONNA to proceed.")
    
    if not PROFESSIONAL_CHANNELS_AVAILABLE:
        raise RuntimeError("Professional channel wrapper is required but not available.")
    
    # Enforce strict device requirements
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was specified but no CUDA-capable GPU is available. Please specify 'cpu' if CPU execution is intended.")
        
        # Enable CUDA by clearing the CPU-only environment variables
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        if 'CUDA_LAUNCH_BLOCKING' in os.environ:
            del os.environ['CUDA_LAUNCH_BLOCKING']
        if 'XLA_FLAGS' in os.environ:
            del os.environ['XLA_FLAGS']
        print(f"🎯 CUDA enabled: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        # Ensure CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("🔒 CPU-only mode enabled")
    
    # Create trainer
    trainer = SRSTrainer(
        use_tensorboard=True,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        device=args.device,
        lr=args.lr
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
    
if __name__ == '__main__':
    print(f"PyTorch MKL-DNN available: {torch.backends.mkldnn.is_available()}")
    print(f"PyTorch using MKL-DNN: {torch.backends.mkldnn.enabled}")
    main()
