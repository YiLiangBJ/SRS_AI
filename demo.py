import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Optional, Tuple

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model import SRSChannelEstimator, TrainableMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


def demo_srs_channel_estimation(
    config: Optional[SRSConfig] = None,
    checkpoint_path: Optional[str] = None,
    snr_db: float = 10.0,
    use_trainable_mmse: bool = False
) -> None:
    """
    Demonstrate SRS channel estimation
    
    Args:
        config: SRS configuration (if None, use default configuration)
        checkpoint_path: Path to trained model checkpoint (if None, use default model)
        snr_db: SNR value in dB for the demonstration
        use_trainable_mmse: Whether to use trainable MMSE matrices
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use default config if none provided
    if config is None:
        config = create_example_config()
    
    # Create data generator with fixed SNR
    data_gen = SRSDataGenerator(
        config,
        snr_range=(snr_db, snr_db),  # Fixed SNR
        device=device
    )
    
    # Create SRS channel estimator
    srs_estimator = SRSChannelEstimator(
        seq_length=config.seq_length,
        ktc=config.ktc,
        max_users=config.num_users,
        max_ports_per_user=max(config.ports_per_user),
        device=device
    ).to(device)
    
    # Create MMSE module if needed
    mmse_module = None
    if use_trainable_mmse:
        mmse_module = TrainableMMSEModule(
            seq_length=config.seq_length
        ).to(device)
    
    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        srs_estimator.load_state_dict(checkpoint['srs_estimator_state_dict'])
        
        if 'mmse_module_state_dict' in checkpoint and mmse_module:
            mmse_module.load_state_dict(checkpoint['mmse_module_state_dict'])
    
    # Set models to evaluation mode
    srs_estimator.eval()
    if mmse_module:
        mmse_module.eval()
    
    # Generate sample
    print("Generating sample data...")
    sample = data_gen.generate_sample()
    ls_estimate = sample['ls_estimate']
    noise_power = sample['noise_power']
    true_channels = sample['true_channels']
    
    print(f"Sample SNR: {snr_db} dB")
    print(f"Estimated noise power: {noise_power:.6f}")
    
    # Print cyclic shift configuration
    print("\nCyclic Shift Configuration:")
    for u in range(config.num_users):
        for p in range(config.ports_per_user[u]):
            shift = config.cyclic_shifts[u][p]
            print(f"User {u}, Port {p}: Cyclic Shift = {shift}")
    
    # Process through model
    print("\nPerforming channel estimation...")
    with torch.no_grad():
        # Use trainable MMSE module if available
        if mmse_module:
            print("Using trainable MMSE module")
            # Extract channel statistics from ls_estimate
            channel_stats = torch.abs(ls_estimate)
            
            # Get trainable C and R matrices
            C, R = mmse_module(channel_stats, torch.tensor([noise_power], device=device))
            
            # Set MMSE matrices in estimator
            srs_estimator.set_mmse_matrices(C=C, R=R)
        else:
            print("Using traditional MMSE approach")
        
        # Process through SRS estimator
        channel_estimates = srs_estimator(
            ls_estimate=ls_estimate,
            cyclic_shifts=config.cyclic_shifts,
            noise_power=noise_power
        )
    
    # Analyze and visualize results
    print("\nResults:")
    idx = 0
    for u in range(config.num_users):
        for p in range(config.ports_per_user[u]):
            # Find true channel
            true_channel = None
            for user, port, channel in true_channels:
                if user == u and port == p:
                    true_channel = channel
                    break
            
            if true_channel is not None:
                # Get estimated channel
                est_channel = channel_estimates[idx]
                
                # Calculate NMSE
                nmse = calculate_nmse(true_channel, est_channel)
                print(f"User {u}, Port {p}: NMSE = {nmse:.2f} dB")
                
                # Visualize
                plt.figure(figsize=(12, 10))
                
                # Plot magnitude
                plt.subplot(2, 1, 1)
                plt.plot(torch.abs(true_channel).cpu().numpy(), 'b-', label='True Channel')
                plt.plot(torch.abs(est_channel).cpu().numpy(), 'r--', label='Estimated Channel')
                plt.title(f"User {u}, Port {p} - Magnitude (NMSE: {nmse:.2f} dB)")
                plt.legend()
                plt.grid(True)
                
                # Plot phase
                plt.subplot(2, 1, 2)
                plt.plot(torch.angle(true_channel).cpu().numpy(), 'b-', label='True Channel')
                plt.plot(torch.angle(est_channel).cpu().numpy(), 'r--', label='Estimated Channel')
                plt.title(f"User {u}, Port {p} - Phase")
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.show()
            
            idx += 1
    
    print("\nDemonstration completed.")


def custom_config_demo():
    """Demo with custom configuration"""
    # Create a custom SRS configuration
    config = SRSConfig(
        seq_length=1200,
        ktc=4,  # K=12
        num_users=3,
        ports_per_user=[2, 2, 1],  # 2 ports for first two users, 1 port for third
        cyclic_shifts=[
            [0, 6],    # User 0's port shifts
            [3, 9],    # User 1's port shifts
            [1]        # User 2's port shift
        ]
    )
    
    # Run demo with custom config
    demo_srs_channel_estimation(
        config=config,
        snr_db=15.0,
        use_trainable_mmse=False  # Use traditional MMSE
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo SRS Channel Estimation')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--snr', type=float, default=10.0, help='SNR in dB')
    parser.add_argument('--use_trainable_mmse', action='store_true', help='Use trainable MMSE matrices')
    parser.add_argument('--custom_config', action='store_true', help='Use custom SRS configuration')
    args = parser.parse_args()
    
    if args.custom_config:
        custom_config_demo()
    else:
        # Run demo with default config
        demo_srs_channel_estimation(
            checkpoint_path=args.checkpoint,
            snr_db=args.snr,
            use_trainable_mmse=args.use_trainable_mmse
        )


if __name__ == "__main__":
    main()
