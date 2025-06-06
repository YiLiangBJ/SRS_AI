import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict, Optional, Tuple

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model import SRSChannelEstimator, TrainableMMSEModule
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
        use_trainable_mmse: bool = True
    ):
        """
        Initialize the trainer
        
        Args:
            config: SRS configuration
            device: Computation device
            save_dir: Directory for saving checkpoints
            use_trainable_mmse: Whether to use trainable MMSE matrices
        """
        self.config = config
        self.device = device
        self.save_dir = save_dir
        self.use_trainable_mmse = use_trainable_mmse
        
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
                seq_length=config.seq_length
            ).to(device)
        else:
            self.mmse_module = None
        
        # Create optimizer
        self.parameters = list(self.srs_estimator.parameters())
        if self.mmse_module:
            self.parameters += list(self.mmse_module.parameters())
        
        self.optimizer = optim.Adam(self.parameters, lr=0.001)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_nmse = []
        self.val_nmse = []
    
    def train_epoch(self, num_batches: int, batch_size: int) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            num_batches: Number of batches
            batch_size: Batch size
            
        Returns:
            Average loss and NMSE for the epoch
        """
        total_loss = 0
        total_nmse = 0
        
        self.srs_estimator.train()
        if self.mmse_module:
            self.mmse_module.train()
        
        for _ in tqdm(range(num_batches), desc="Training"):
            # Generate batch
            batch = self.data_gen.generate_batch(batch_size)
            
            # Get ls estimates and cyclic shifts
            ls_estimates = batch['ls_estimates']
            noise_powers = batch['noise_powers']
            true_channels = batch['true_channels']
            
            # Clear gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            batch_loss = 0
            batch_nmse = 0
            
            for i in range(batch_size):
                ls_estimate = ls_estimates[i]
                noise_power = noise_powers[i].item()
                
                # Use trainable MMSE module if available
                if self.mmse_module:
                    # Extract channel statistics from ls_estimate
                    channel_stats = torch.abs(ls_estimate)  # Use magnitude as channel statistics
                    
                    # Get trainable C and R matrices
                    C, R = self.mmse_module(channel_stats, torch.tensor([noise_power], device=self.device))
                    
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
                            sample_loss = self.loss_fn(
                                torch.real(est_channel),
                                torch.real(true_channel)
                            ) + self.loss_fn(
                                torch.imag(est_channel),
                                torch.imag(true_channel)
                            )
                            
                            batch_loss += sample_loss
                            
                            # Calculate NMSE
                            nmse = calculate_nmse(true_channel, est_channel)
                            batch_nmse += nmse
                            
                        idx += 1
            
            # Average loss and NMSE over batch
            batch_loss /= (batch_size * sum(self.config.ports_per_user))
            batch_nmse /= (batch_size * sum(self.config.ports_per_user))
            
            # Backward pass and optimize
            batch_loss.backward()
            self.optimizer.step()
            
            # Update totals
            total_loss += batch_loss.item()
            total_nmse += batch_nmse
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_nmse = total_nmse / num_batches
        
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
        total_loss = 0
        total_nmse = 0
        
        self.srs_estimator.eval()
        if self.mmse_module:
            self.mmse_module.eval()
        
        with torch.no_grad():
            for _ in tqdm(range(num_batches), desc="Validating"):
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
                        # Extract channel statistics from ls_estimate
                        channel_stats = torch.abs(ls_estimate)  # Use magnitude as channel statistics
                        
                        # Get trainable C and R matrices
                        C, R = self.mmse_module(channel_stats, torch.tensor([noise_power], device=self.device))
                        
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
                                sample_loss = self.loss_fn(
                                    torch.real(est_channel),
                                    torch.real(true_channel)
                                ) + self.loss_fn(
                                    torch.imag(est_channel),
                                    torch.imag(true_channel)
                                )
                                
                                batch_loss += sample_loss
                                
                                # Calculate NMSE
                                nmse = calculate_nmse(true_channel, est_channel)
                                batch_nmse += nmse
                                
                            idx += 1
                
                # Average loss and NMSE over batch
                batch_loss /= (batch_size * sum(self.config.ports_per_user))
                batch_nmse /= (batch_size * sum(self.config.ports_per_user))
                
                # Update totals
                total_loss += batch_loss.item()
                total_nmse += batch_nmse
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_nmse = total_nmse / num_batches
        
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
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss, train_nmse = self.train_epoch(train_batches, batch_size)
            self.train_losses.append(train_loss)
            self.train_nmse.append(train_nmse)
            
            # Validate
            val_loss, val_nmse = self.validate(val_batches, batch_size)
            self.val_losses.append(val_loss)
            self.val_nmse.append(val_nmse)
            
            print(f"Train Loss: {train_loss:.6f}, Train NMSE: {train_nmse:.2f} dB")
            print(f"Val Loss: {val_loss:.6f}, Val NMSE: {val_nmse:.2f} dB")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_loss)
            
            # Plot training progress
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
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def plot_training_progress(self) -> None:
        """Plot training progress"""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, 'b-', label='Train Loss')
        plt.plot(self.val_losses, 'r-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # NMSE plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_nmse, 'b-', label='Train NMSE')
        plt.plot(self.val_nmse, 'r-', label='Val NMSE')
        plt.xlabel('Epoch')
        plt.ylabel('NMSE (dB)')
        plt.title('Training and Validation NMSE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'))
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SRS Channel Estimator')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--train_batches', type=int, default=100, help='Number of training batches per epoch')
    parser.add_argument('--val_batches', type=int, default=20, help='Number of validation batches per epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--use_trainable_mmse', action='store_true', help='Use trainable MMSE matrices')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='Path to checkpoint to load')
    args = parser.parse_args()
    
    # Create example config
    config = create_example_config()
    
    # Create trainer
    trainer = SRSTrainer(
        config=config,
        save_dir=args.save_dir,
        use_trainable_mmse=args.use_trainable_mmse
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
