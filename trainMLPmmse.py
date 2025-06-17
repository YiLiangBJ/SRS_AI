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


class SRSTrainerModified:
    """
    Modified Trainer for SRS Channel Estimator that uses h_with_residual/phasor as input
    """
    def __init__(
        self,
        config: SRSConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str = "./checkpoints_modified",
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
          
        # Create trainable MMSE module if needed
        if use_trainable_mmse:
            self.mmse_module = TrainableMMSEModule(
                seq_length=config.seq_length,
                mmse_block_size=config.mmse_block_size,
                use_complex_input=True
            ).to(device)
        else:
            self.mmse_module = None
            
        # Create models
        self.srs_estimator = SRSChannelEstimator(
            seq_length=config.seq_length,
            ktc=config.ktc,
            max_users=config.num_users,
            max_ports_per_user=max(config.ports_per_user),
            mmse_block_size=config.mmse_block_size,
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
            with torch.no_grad():
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
                  
                # if self.mmse_module:
                # 运行MLP生成的MMSE滤波器，或者正常运行MMSE估计器
                channel_estimates = self.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=self.config.cyclic_shifts,
                    noise_power=noise_power
                )
                # else:
                #     # 正常运行SRS估计器
                #     channel_estimates = self.srs_estimator(
                #         ls_estimate=ls_estimate,
                #         cyclic_shifts=self.config.cyclic_shifts,
                #         noise_power=noise_power
                #     )
                
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
          
        # # Calculate averages
        # # 我们需要计算整个epoch处理的总样本数
        # # 每个批次处理的样本数 = 批次大小 * 实际信道数量
        # # 整个epoch处理的总样本数 = 批次数 * 每个批次处理的样本数
        # actual_channel_count = sum(1 for u in range(self.config.num_users) 
        #                          for p in range(self.config.ports_per_user[u]) 
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
                        # Use trainable MMSE module with h_with_residual/phasor
                    if self.mmse_module:
                        # 首先运行SRS估计器来生成h_with_residual/phasor
                        channel_estimates_initial = self.srs_estimator(
                            ls_estimate=ls_estimate,
                            cyclic_shifts=self.config.cyclic_shifts,
                            noise_power=noise_power
                        )
                        
                        # 对所有用户/端口的h_with_residual/phasor进行处理
                        h_inputs = []
                        user_port_pairs = []
                        
                        # 检查是否有h_with_residual/phasors字典可用
                        if hasattr(self.srs_estimator, 'current_h_with_residual_phasors') and self.srs_estimator.current_h_with_residual_phasors:
                            for (u, p), h_input in self.srs_estimator.current_h_with_residual_phasors.items():
                                h_inputs.append(h_input)
                                user_port_pairs.append((u, p))
                                
                            # 对每个用户/端口分别生成MMSE矩阵
                            all_C = {}
                            all_R = {}
                            
                            for idx, (u, p) in enumerate(user_port_pairs):
                                # 使用当前用户/端口的h_with_residual/phasor作为输入
                                C, R = self.mmse_module(h_inputs[idx])
                                all_C[(u, p)] = C
                                all_R[(u, p)] = R
                            
                            # 使用最后一个MMSE矩阵设置估计器
                            if user_port_pairs:
                                last_u, last_p = user_port_pairs[-1]
                                self.srs_estimator.set_mmse_matrices(C=all_C[(last_u, last_p)], R=all_R[(last_u, last_p)])
                            
                            # 重新运行估计器，使用新生成的MMSE矩阵
                            channel_estimates = self.srs_estimator(
                                ls_estimate=ls_estimate,
                                cyclic_shifts=self.config.cyclic_shifts,
                                noise_power=noise_power
                            )
                        else:
                            # 如果没有可用的字典，则回退到使用单个h_with_residual_phasor
                            if self.srs_estimator.current_h_with_residual_phasor is not None:
                                C, R = self.mmse_module(self.srs_estimator.current_h_with_residual_phasor)
                                self.srs_estimator.set_mmse_matrices(C=C, R=R)
                                channel_estimates = self.srs_estimator(
                                    ls_estimate=ls_estimate,
                                    cyclic_shifts=self.config.cyclic_shifts,
                                    noise_power=noise_power
                                )
                            else:
                                channel_estimates = channel_estimates_initial
                    else:
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


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SRS Channel Estimator with h_with_residual/phasor as input")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--train_batches', type=int, default=100, help='Number of training batches per epoch')
    parser.add_argument('--val_batches', type=int, default=20, help='Number of validation batches')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every n epochs')
    parser.add_argument('--save_every', type=int, default=5, help='Save checkpoint every n epochs')
    parser.add_argument('--no_mmse', action='store_true', help='Disable trainable MMSE')
    parser.add_argument('--enable_plotting', action='store_true', help='Enable plotting')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_modified', help='Save directory')
    parser.add_argument('--load_checkpoint', type=str, default='', help='Load checkpoint file')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_example_config()
    
    # Create trainer
    trainer = SRSTrainerModified(
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir=args.save_dir,
        use_trainable_mmse=not args.no_mmse,
        enable_plotting=args.enable_plotting
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
    main()
