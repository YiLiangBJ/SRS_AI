import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict, Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model_unified import SRSChannelEstimatorUnified
from model_unified_mmse import UnifiedMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


class SRSTrainerUnified:
    """
    Trainer for SRS Channel Estimator with unified MMSE approach
    """
    def __init__(
        self,
        config: SRSConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints_unified",
        use_trainable_mmse: bool = True,
        enable_plotting: bool = False
    ):
        """
        Initialize the trainer with unified MMSE
        
        Args:
            config: SRS configuration
            device: Computation device
            save_dir: Directory for saving checkpoints
            use_trainable_mmse: Whether to use trainable MMSE matrices
            enable_plotting: Whether to enable plotting during training
        """
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.use_trainable_mmse = use_trainable_mmse
        self.enable_plotting = enable_plotting
        
        # Create data generator
        self.data_gen = SRSDataGenerator(config, device=device)
        
        # Create models - using the unified version
        self.srs_estimator = SRSChannelEstimatorUnified(
            seq_length=config.seq_length,
            ktc=config.ktc,
            max_users=config.num_users,
            max_ports_per_user=max(config.ports_per_user),
            mmse_block_size=config.mmse_block_size,
            device=device
        ).to(device)
          
        # Create trainable MMSE module if needed - using the unified version
        if use_trainable_mmse:
            self.mmse_module = UnifiedMMSEModule(
                seq_length=config.seq_length,
                mmse_block_size=config.mmse_block_size,
                use_complex_input=True
            ).to(device)
        else:
            self.mmse_module = None
            
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
        
        self.optimizer = optim.Adam(model_params, lr=0.001)
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Create logs directory for TensorBoard
        self.log_dir = os.path.join(save_dir, 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
          
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard 日志保存到: {self.log_dir}")
        print(f"启动 TensorBoard: tensorboard --logdir={self.save_dir}/logs")
          
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        
        # 添加模型参数配置到TensorBoard
        self.writer.add_text('Configuration/Model', f"序列长度: {config.seq_length}, 用户数: {config.num_users}, MMSE块大小: {config.mmse_block_size}")
        self.writer.add_text('Configuration/Training', f"设备: {device}, 统一MMSE架构: {use_trainable_mmse}")
        ports_text = ", ".join([f"用户{i}: {p}个端口" for i, p in enumerate(config.ports_per_user)])
        self.writer.add_text('Configuration/Users', f"端口配置: {ports_text}")
        
        # Global step counter for logging
        self.global_step = 0
        
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
        if self.mmse_module:
            self.mmse_module.train()
        
        for batch_idx in tqdm(range(num_batches), desc="训练中"):
            # Generate batch
            batch = self.data_gen.generate_batch(batch_size)
            
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
                
                # 处理信道估计
                # 注意: 在统一版本中，MMSE矩阵是在forward过程中由h_with_residual/phasor生成的
                
                # 处理SRS估计器
                channel_estimates = self.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=self.config.cyclic_shifts,
                    noise_power=noise_power
                )
                
                # 在此处，如果使用trainable MMSE，为每个用户/端口的h_with_residual/phasor生成MMSE矩阵
                # 当SRSChannelEstimatorUnified处理每个用户的信道时，它会更新current_channel_input
                if self.mmse_module:
                    # 获取最新的h_with_residual/phasor
                    channel_input = self.srs_estimator.current_channel_input
                    if channel_input is not None:
                        # 使用统一的MMSE模块生成C和R矩阵
                        C, R = self.mmse_module(channel_input)
                        # 设置估计器的MMSE矩阵
                        self.srs_estimator.set_mmse_matrices(C=C, R=R)
                
                # Calculate loss for each user/port
                idx = 0
                for u in range(self.config.num_users):
                    for p in range(self.config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            true_channel = true_channels[(u, p)][i]
                            est_channel = channel_estimates[idx]
                              
                            # 确保估计信道需要梯度
                            if not est_channel.requires_grad:
                                print(f"警告：估计的信道在批次 {batch_idx}，样本 {i}，用户 {u}，端口 {p} 不需要梯度")
                                continue
                                
                            # 将估计信道转换回频域进行比较 (从K个时域抽头转换为L个频域点)
                            est_channel_freq = torch.fft.fft(est_channel, n=self.config.seq_length)
                            
                            # 计算实部和虚部的损失
                            real_loss = torch.mean((torch.real(est_channel_freq) - torch.real(true_channel))**2)
                            imag_loss = torch.mean((torch.imag(est_channel_freq) - torch.imag(true_channel))**2)
                              
                            # 此样本的损失
                            sample_loss = real_loss + imag_loss
                              
                            # 使用加法赋值更新样本总损失
                            sample_total_loss = sample_total_loss + sample_loss
                              
                            # Calculate NMSE
                            with torch.no_grad():  # NMSE只用于监控，不需要梯度
                                nmse = calculate_nmse(true_channel, est_channel_freq)
                                batch_nmse += nmse  # 累加NMSE值，稍后在打印时将除以样本数
                            
                            idx += 1
                  
                # 将这个样本的总损失加到批次损失中
                batch_loss = batch_loss + sample_total_loss
            
            # 确保损失是一个需要梯度的标量 - 移到了样本循环外部
            if batch_loss.requires_grad:
                # Backward pass and optimize
                batch_loss.backward()
            
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
                actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                         for p in range(self.config.ports_per_user[u]) 
                                         if (u, p) in true_channels)
                num_samples_in_batch = batch_size * actual_channel_count
                avg_batch_nmse = batch_nmse / num_samples_in_batch
                  
                # Log batch-level metrics to TensorBoard
                self.writer.add_scalar('Loss/batch', batch_loss_value, self.global_step)
                self.writer.add_scalar('NMSE/batch', avg_batch_nmse, self.global_step)
                
                # 按训练进度记录损失和NMSE（以批次为单位）
                self.writer.add_scalar('Training/batch_progress', (batch_idx + 1) / num_batches * 100, self.global_step)
                
                # 记录每个用户端口的平均NMSE
                idx = 0
                for u in range(self.config.num_users):
                    for p in range(self.config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            # 这里通过索引模拟计算每个用户/端口的NMSE
                            # 在实际场景中，您可能需要分别计算每个用户/端口的NMSE
                            self.writer.add_scalar(f'PerUser/user_{u}_port_{p}_nmse', avg_batch_nmse, self.global_step)
                            idx += 1
                  
                # Print loss information to console for immediate feedback
                print(f"\n批次 [{batch_idx+1:03d}/{num_batches:03d}] - 损失: {batch_loss_value:.6f}, NMSE: {avg_batch_nmse:.2f} dB")
                
                self.global_step += 1
          
        # Calculate averages
        # 我们需要计算整个epoch处理的总样本数
        actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                 for p in range(self.config.ports_per_user[u]) 
                                 if (u, p) in true_channels)  # 使用最后一个批次的数据
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
            for batch_idx in tqdm(range(num_batches), desc="验证中"):
                # Generate batch
                batch = self.data_gen.generate_batch(batch_size)
                
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
                    
                    # 处理SRS估计器
                    channel_estimates = self.srs_estimator(
                        ls_estimate=ls_estimate,
                        cyclic_shifts=self.config.cyclic_shifts,
                        noise_power=noise_power
                    )
                    
                    # 如果使用trainable MMSE，为h_with_residual/phasor生成MMSE矩阵
                    if self.mmse_module:
                        channel_input = self.srs_estimator.current_channel_input
                        if channel_input is not None:
                            C, R = self.mmse_module(channel_input)
                            self.srs_estimator.set_mmse_matrices(C=C, R=R)
                    
                    # Calculate loss for each user/port
                    idx = 0
                    for u in range(self.config.num_users):
                        for p in range(self.config.ports_per_user[u]):
                            if (u, p) in true_channels:
                                true_channel = true_channels[(u, p)][i]
                                est_channel = channel_estimates[idx]
                                
                                # 将估计信道转换回频域进行比较
                                est_channel_freq = torch.fft.fft(est_channel, n=self.config.seq_length)
                                
                                # Calculate loss
                                real_loss = torch.mean((torch.real(est_channel_freq) - torch.real(true_channel))**2).item()
                                imag_loss = torch.mean((torch.imag(est_channel_freq) - torch.imag(true_channel))**2).item()
                                sample_loss = real_loss + imag_loss
                                
                                batch_loss += sample_loss
                                
                                # Calculate NMSE
                                nmse = calculate_nmse(true_channel, est_channel_freq)
                                batch_nmse += nmse
                                
                            idx += 1
                
                # Update totals
                total_loss += batch_loss
                total_nmse += batch_nmse
                
                # 计算这个批次处理了多少个实际计算的样本数
                actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                         for p in range(self.config.ports_per_user[u]) 
                                         if (u, p) in true_channels)
                num_samples_in_batch = batch_size * actual_channel_count
                avg_batch_nmse = batch_nmse / num_samples_in_batch
                
                # Log validation batch metrics
                self.writer.add_scalar('Validation/batch_loss', batch_loss / num_samples_in_batch, self.global_step)
                self.writer.add_scalar('Validation/batch_nmse', avg_batch_nmse, self.global_step)
                
        # Calculate averages for the entire validation set
        actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                for p in range(self.config.ports_per_user[u]) 
                                if (u, p) in true_channels)
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
        print(f"批次大小: {batch_size}, 每轮批次数: {num_batches}")
        print(f"验证批次数: {val_batches}, 每 {val_every_n_epochs} 轮验证一次")
        print(f"每 {save_every_n_epochs} 轮保存一次检查点")
        
        best_val_nmse = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n====== Epoch {epoch+1}/{num_epochs} ======")
            
            # Train for one epoch
            train_loss, train_nmse = self.train_epoch(num_batches, batch_size)
            self.train_losses.append(train_loss)
            self.train_nmse.append(train_nmse)
            
            # Log epoch metrics
            self.writer.add_scalar('Training/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('Training/epoch_nmse', train_nmse, epoch)
            self.writer.add_scalar('Training/epoch_progress', (epoch + 1) / num_epochs * 100, epoch)
            
            # Validate if needed
            if (epoch + 1) % val_every_n_epochs == 0:
                val_loss, val_nmse = self.validate(val_batches, batch_size)
                self.val_losses.append(val_loss)
                self.val_nmse.append(val_nmse)
                
                # Log validation metrics
                self.writer.add_scalar('Validation/epoch_loss', val_loss, epoch)
                self.writer.add_scalar('Validation/epoch_nmse', val_nmse, epoch)
                
                # Check if this is the best model
                if val_nmse < best_val_nmse:
                    best_val_nmse = val_nmse
                    self._save_checkpoint(epoch, is_best=True)
                    print(f"新的最佳模型! NMSE: {val_nmse:.2f} dB")
                
                print(f"验证损失: {val_loss:.6f}, 验证NMSE: {val_nmse:.2f} dB")
            
            # Save checkpoint if needed
            if (epoch + 1) % save_every_n_epochs == 0:
                self._save_checkpoint(epoch)
            
            # Print epoch summary
            print(f"轮次 {epoch+1} 结束 - 训练损失: {train_loss:.6f}, 训练NMSE: {train_nmse:.2f} dB")
        
        print("\n====== 训练完成 ======")
        
        # Return training history
        history = {
            'train_loss': self.train_losses,
            'train_nmse': self.train_nmse,
            'val_loss': self.val_losses,
            'val_nmse': self.val_nmse
        }
        
        return history
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save a checkpoint of the model
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        # Create state dict
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.srs_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses,
            'train_nmse': self.train_nmse,
            'val_loss': self.val_losses,
            'val_nmse': self.val_nmse
        }
        
        if self.mmse_module:
            checkpoint['mmse_module_state_dict'] = self.mmse_module.state_dict()
        
        # Save regular checkpoint
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        print(f"保存检查点到: {path}")
        
        # If this is the best model, save a copy
        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型到: {best_path}")

    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint
        
        Args:
            path: Path to the checkpoint file
            
        Returns:
            The epoch number of the loaded checkpoint
        """
        print(f"加载检查点: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.srs_estimator.load_state_dict(checkpoint['model_state_dict'])
        
        if self.mmse_module and 'mmse_module_state_dict' in checkpoint:
            self.mmse_module.load_state_dict(checkpoint['mmse_module_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_loss' in checkpoint:
            self.train_losses = checkpoint['train_loss']
        if 'train_nmse' in checkpoint:
            self.train_nmse = checkpoint['train_nmse']
        if 'val_loss' in checkpoint:
            self.val_losses = checkpoint['val_loss']
        if 'val_nmse' in checkpoint:
            self.val_nmse = checkpoint['val_nmse']
            
        epoch = checkpoint.get('epoch', 0)
        print(f"成功加载检查点，轮次: {epoch}")
        
        return epoch

def main():
    """Main function to run the training process"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train SRS Channel Estimator with unified MMSE approach')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--train_batches', type=int, default=100, help='Number of batches per epoch during training')
    parser.add_argument('--val_batches', type=int, default=20, help='Number of batches for validation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every n epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every n epochs')
    parser.add_argument('--no_trainable_mmse', action='store_true', help='Disable trainable MMSE matrices')
    parser.add_argument('--enable_plotting', action='store_true', help='Enable plotting during training')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_unified', help='Directory for saving checkpoints')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_example_config()
    
    # Create trainer
    trainer = SRSTrainerUnified(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=args.save_dir,
        use_trainable_mmse=not args.no_trainable_mmse,
        enable_plotting=args.enable_plotting
    )
    
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
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    if history['val_loss']:
        plt.plot(range(0, len(history['val_loss']) * args.val_every, args.val_every), 
                history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
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
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    print(f"训练历史保存到: {os.path.join(args.save_dir, 'training_history.png')}")
    
if __name__ == "__main__":
    main()
