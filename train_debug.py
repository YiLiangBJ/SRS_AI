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
from model import SRSChannelEstimator
from model_cholesky import TrainableMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


class SRSTrainer:
    """
    Trainer for SRS Channel Estimator
    """
    def __init__(
        self,
        config: SRSConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints",
        use_trainable_mmse: bool = True,
        enable_plotting: bool = False
    ):
        """
        Initialize the trainer
        
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
        
        # Create models
        self.srs_estimator = SRSChannelEstimator(
            seq_length=config.seq_length,
            ktc=config.ktc,
            max_users=config.num_users,
            max_ports_per_user=max(config.ports_per_user),
            mmse_block_size=config.mmse_block_size,
            device=device
        ).to(device)
          # Create trainable MMSE module if needed
        if use_trainable_mmse:
            self.mmse_module = TrainableMMSEModule(
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
          # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
        
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
        # self.srs_estimator.train()
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
                
                # Use trainable MMSE module if available
                if self.mmse_module:
                    # 直接使用ls_estimate作为输入，保留复数信息
                    C, R = self.mmse_module(ls_estimate)
                    
                    # Set MMSE matrices in estimator
                    self.srs_estimator.set_mmse_matrices(C=C, R=R)
                
                # Process through SRS estimator
                channel_estimates = self.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=self.config.cyclic_shifts,
                    noise_power=noise_power
                )
                
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
                        elif param.grad.abs().sum().item() == 0:
                            print(f"SRS参数 {name} 的梯度全为零")
                        else:
                            grad_norm = param.grad.abs().sum().item()
                            print(f"SRS参数 {name} 的梯度范数: {grad_norm:.6f}")
                            self.writer.add_scalar(f'Gradients/SRS_{name}', grad_norm, self.global_step)
                  # 打印MMSE模块的梯度信息
                if self.mmse_module:
                    # 确保梯度信息打印在新行，避免与进度条同行
                    print("")
                    for name, param in self.mmse_module.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                print(f"MMSE参数 {name} 没有梯度")
                            elif param.grad.abs().sum().item() == 0:
                                print(f"MMSE参数 {name} 的梯度全为零")
                            else:
                                grad_norm = param.grad.abs().sum().item()
                                print(f"MMSE参数 {name} 的梯度范数: {grad_norm:.6f}")
                    
                    # 记录每个MMSE参数的梯度范数到TensorBoard
                    for name, param in self.mmse_module.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            grad_norm = param.grad.abs().sum().item()
                            self.writer.add_scalar(f'Gradients/{name}', grad_norm, self.global_step)
                
                # 执行梯度更新
                self.optimizer.step()            
            else:
                print(f"警告：批次 {batch_idx} 的损失不需要梯度。跳过反向传播。")
                print(f"批次损失的类型：{type(batch_loss)}")
                print(f"批次损失的requires_grad：{batch_loss.requires_grad}")            # Update totals
            with torch.no_grad():
                batch_loss_value = batch_loss.item()
                total_loss += batch_loss_value
                total_nmse += batch_nmse
                
                # 计算这个批次处理了多少个实际计算的样本数
                # 跟踪实际处理了多少个信道样本（每个用户可能有不同数量的端口）
                # 实际计算的样本数 = 批次大小 * 真实信道的总数（考虑到每个用户可能有不同的端口数）
                actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                         for p in range(self.config.ports_per_user[u]) 
                                         if (u, p) in true_channels)
                num_samples_in_batch = batch_size * actual_channel_count
                avg_batch_nmse = batch_nmse / num_samples_in_batch
                  # Log batch-level metrics to TensorBoard
                self.writer.add_scalar('Loss/batch', batch_loss_value, self.global_step)
                self.writer.add_scalar('NMSE/batch', avg_batch_nmse, self.global_step)
                  # Print loss information to console for immediate feedback
                print(f"\n批次 [{batch_idx+1:03d}/{num_batches:03d}] - 损失: {batch_loss_value:.6f}, NMSE: {avg_batch_nmse:.2f} dB")
                
                self.global_step += 1
          # Calculate averages
        # 我们需要计算整个epoch处理的总样本数
        # 每个批次处理的样本数 = 批次大小 * 实际信道数量
        # 整个epoch处理的总样本数 = 批次数 * 每个批次处理的样本数
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
            for _ in tqdm(range(num_batches), desc="验证中"):
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
                      # Use trainable MMSE module if available
                    if self.mmse_module:
                        # 直接使用ls_estimate作为输入，保留复数信息
                        C, R = self.mmse_module(ls_estimate)
                        
                        # Set MMSE matrices in estimator
                        self.srs_estimator.set_mmse_matrices(C=C, R=R)
                    
                    # Process through SRS estimator
                    channel_estimates = self.srs_estimator(
                        ls_estimate=ls_estimate,
                        cyclic_shifts=self.config.cyclic_shifts,
                        noise_power=noise_power
                    )
                    
                    # Calculate loss for each user/port
                    idx = 0
                    for u in range(self.config.num_users):
                        for p in range(self.config.ports_per_user[u]):
                            if (u, p) in true_channels:
                                true_channel = true_channels[(u, p)][i]
                                est_channel = channel_estimates[idx]
                                
                                # Calculate loss
                                real_loss = torch.mean((torch.real(est_channel) - torch.real(true_channel))**2)
                                imag_loss = torch.mean((torch.imag(est_channel) - torch.imag(true_channel))**2)
                                sample_loss = real_loss + imag_loss
                                
                                batch_loss += sample_loss.item()                                # Calculate NMSE
                                nmse = calculate_nmse(true_channel, est_channel)
                                batch_nmse += nmse  # 累加NMSE值，稍后在打印时将除以样本数
                                
                            idx += 1                # Update totals
                total_loss += batch_loss
                total_nmse += batch_nmse
                
                # 计算这个批次处理了多少个实际计算的样本数
                # 跟踪实际处理了多少个信道样本（每个用户可能有不同数量的端口）
                # 实际计算的样本数 = 批次大小 * 真实信道的总数（考虑到每个用户可能有不同的端口数）
                actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                         for p in range(self.config.ports_per_user[u]) 
                                         if (u, p) in true_channels)
                num_samples_in_batch = batch_size * actual_channel_count
                avg_batch_nmse = batch_nmse / num_samples_in_batch                # Log validation batch metrics
                self.writer.add_scalar('Validation/batch_loss', batch_loss, self.global_step)
                self.writer.add_scalar('Validation/batch_nmse', avg_batch_nmse, self.global_step)
                
                # Print validation loss information to console
                print(f"\n验证批次 - 损失: {batch_loss:.6f}, NMSE: {avg_batch_nmse:.2f} dB")
          # Calculate averages
        # 我们需要计算整个验证集处理的总样本数
        # 每个批次处理的样本数 = 批次大小 * 实际信道数量
        # 整个验证集处理的总样本数 = 批次数 * 每个批次处理的样本数
        actual_channel_count = sum(1 for u in range(self.config.num_users) 
                                 for p in range(self.config.ports_per_user[u]) 
                                 if (u, p) in true_channels)  # 使用最后一个批次的数据
        num_channels = batch_size * actual_channel_count * num_batches
        avg_loss = total_loss / num_channels
        avg_nmse = total_nmse / num_channels
        
        return avg_loss, avg_nmse
    
    def train(self, num_epochs: int, train_batches: int, val_batches: int, batch_size: int) -> None:
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs
            train_batches: Number of training batches per epoch
            val_batches: Number of validation batches per epoch
            batch_size: Batch size
        """
        for epoch in range(num_epochs):
            print(f"训练轮次 {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_nmse = self.train_epoch(train_batches, batch_size)
            self.train_losses.append(train_loss)
            self.train_nmse.append(train_nmse)
            
            # Validate
            val_loss, val_nmse = self.validate(val_batches, batch_size)
            self.val_losses.append(val_loss)
            self.val_nmse.append(val_nmse)
            print(f"\n======== 轮次 {epoch+1}/{num_epochs} 完成 ========")
            print(f"训练损失: {train_loss:.6f}, 训练NMSE: {train_nmse:.2f} dB")
            print(f"验证损失: {val_loss:.6f}, 验证NMSE: {val_nmse:.2f} dB")
            print("=====================================")
            
            # Log metrics to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('NMSE/train', train_nmse, epoch)
            self.writer.add_scalar('NMSE/val', val_nmse, epoch)
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Plot training progress only if enabled
            if self.enable_plotting:
                self.plot_training_progress()
    
    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save a checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'config': self.config,
            'srs_estimator_state_dict': self.srs_estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_nmse': self.train_nmse,
            'val_nmse': self.val_nmse,
        }
        
        if self.mmse_module:
            checkpoint['mmse_module_state_dict'] = self.mmse_module.state_dict()
        
        torch.save(checkpoint, os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt"))
        
        # Save best model
        if len(self.val_losses) == 1 or val_loss < min(self.val_losses[:-1]):
            torch.save(checkpoint, os.path.join(self.save_dir, "best_model.pt"))
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        checkpoint = torch.load(checkpoint_path)
        
        # Load model state
        self.srs_estimator.load_state_dict(checkpoint['srs_estimator_state_dict'])
        
        if 'mmse_module_state_dict' in checkpoint and self.mmse_module:
            self.mmse_module.load_state_dict(checkpoint['mmse_module_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.train_nmse = checkpoint['train_nmse']
        self.val_nmse = checkpoint['val_nmse']
        
        print(f"加载了第 {checkpoint['epoch']} 轮的检查点")
    
    def plot_training_progress(self) -> None:
        """Plot training progress"""
        # Skip if plotting is disabled
        if not self.enable_plotting:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, 'b-', label='训练损失')
        plt.plot(self.val_losses, 'r-', label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        # NMSE plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_nmse, 'b-', label='训练NMSE')
        plt.plot(self.val_nmse, 'r-', label='验证NMSE')
        plt.xlabel('轮次')
        plt.ylabel('NMSE (dB)')
        plt.title('训练和验证NMSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'))
        plt.close()


def main():
    # 创建一个空的命名空间对象，用于存放参数
    args = argparse.Namespace()
    
    # 直接设置参数，不需要通过命令行指定
    args.epochs = 50              # 训练轮数
    args.train_batches = 100      # 每轮训练的批次数 
    args.val_batches = 20         # 每轮验证的批次数
    args.batch_size = 16          # 批次大小
    args.save_dir = './checkpoints'  # 保存检查点的目录
    args.use_trainable_mmse = True   # 使用可训练的MMSE矩阵
    args.load_checkpoint = None      # 不加载现有检查点
    args.enable_plotting = False     # 禁用绘图
    
    print("使用的设备：", "cuda" if torch.cuda.is_available() else "cpu")
    print("TensorBoard日志路径：", os.path.join(args.save_dir, 'logs'))
    print("可以使用以下命令启动TensorBoard：")
    print("tensorboard --logdir=./checkpoints/logs")
    
    # Create example config
    config = create_example_config()
    
    # Create trainer with plotting disabled by default
    trainer = SRSTrainer(
        config=config,
        save_dir=args.save_dir,
        use_trainable_mmse=args.use_trainable_mmse,
        enable_plotting=args.enable_plotting
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Train
    trainer.train(
        num_epochs=args.epochs,
        train_batches=args.train_batches,
        val_batches=args.val_batches,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
