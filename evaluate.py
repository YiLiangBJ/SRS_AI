import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model import SRSChannelEstimator, TrainableMMSEModule
from utils import calculate_nmse, visualize_channel_estimate


def evaluate_model(
    checkpoint_path: str,
    num_samples: int = 100,
    snr_values: List[float] = None,
    save_dir: str = './results'
) -> None:
    """
    Evaluate a trained model
    
    Args:
        checkpoint_path: Path to the model checkpoint
        num_samples: Number of samples to evaluate
        snr_values: List of SNR values to evaluate at (if None, use random SNRs)
        save_dir: Directory for saving results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
      # Create models
    srs_estimator = SRSChannelEstimator(
        seq_length=config.seq_length,
        ktc=config.ktc,
        max_users=config.num_users,
        max_ports_per_user=max(config.ports_per_user),
        mmse_block_size=config.mmse_block_size,
        device=device
    ).to(device)
    
    # Load model state
    srs_estimator.load_state_dict(checkpoint['srs_estimator_state_dict'])
    
    # Create MMSE module if it exists in checkpoint
    mmse_module = None
    if 'mmse_module_state_dict' in checkpoint:
        mmse_module = TrainableMMSEModule(
            seq_length=config.seq_length
        ).to(device)
        mmse_module.load_state_dict(checkpoint['mmse_module_state_dict'])
    
    # Set models to evaluation mode
    srs_estimator.eval()
    if mmse_module:
        mmse_module.eval()
    
    # Create data generator
    if snr_values is None:
        # Use default SNR range
        data_gen = SRSDataGenerator(config, device=device)
    else:
        # Fix SNR to specified values
        data_gen = SRSDataGenerator(
            config,
            snr_range=(min(snr_values), max(snr_values)),  # Will be overridden
            device=device
        )
    
    # Evaluate with fixed SNRs
    if snr_values:
        # Dictionary to store results per SNR
        results_per_snr = {snr: [] for snr in snr_values}
        
        # For each SNR
        for snr_db in tqdm(snr_values, desc="Evaluating SNRs"):
            # Evaluate multiple samples at this SNR
            for _ in range(num_samples):
                # Generate sample with fixed SNR
                batch = data_gen.generate_batch(1)
                
                # Override SNR in batch
                signal_power = torch.mean(torch.abs(batch['ls_estimates'][0]) ** 2).item()
                noise_power = signal_power / (10 ** (snr_db / 10))
                
                # Regenerate noisy signal with fixed SNR
                ls_estimate_clean = batch['ls_estimates'][0] / (1 + noise_power)  # Approximate clean signal
                noise_real = torch.randn_like(torch.real(ls_estimate_clean)) * np.sqrt(noise_power / 2)
                noise_imag = torch.randn_like(torch.imag(ls_estimate_clean)) * np.sqrt(noise_power / 2)
                noise = torch.complex(noise_real, noise_imag)
                ls_estimate = ls_estimate_clean + noise
                
                # Process through model
                with torch.no_grad():
                    # Use trainable MMSE module if available
                    if mmse_module:
                        # Extract channel statistics from ls_estimate
                        channel_stats = torch.abs(ls_estimate)
                        
                        # Get trainable C and R matrices
                        C, R = mmse_module(channel_stats, torch.tensor([noise_power], device=device))
                        
                        # Set MMSE matrices in estimator
                        srs_estimator.set_mmse_matrices(C=C, R=R)
                    
                    # Process through SRS estimator
                    channel_estimates = srs_estimator(
                        ls_estimate=ls_estimate,
                        cyclic_shifts=config.cyclic_shifts,
                        noise_power=noise_power
                    )
                
                # Calculate NMSE for each user/port
                true_channels = batch['true_channels']
                idx = 0
                for u in range(config.num_users):
                    for p in range(config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            true_channel = true_channels[(u, p)][0]
                            est_channel = channel_estimates[idx]
                            
                            # Calculate NMSE
                            nmse = calculate_nmse(true_channel, est_channel)
                            
                            # Store result
                            results_per_snr[snr_db].append((u, p, nmse))
                        
                        idx += 1
        
        # Calculate average NMSE per SNR
        avg_nmse_per_snr = {}
        for snr_db in snr_values:
            # Overall average
            avg_nmse = np.mean([result[2] for result in results_per_snr[snr_db]])
            avg_nmse_per_snr[snr_db] = avg_nmse
            
            # Per user/port average
            for u in range(config.num_users):
                for p in range(config.ports_per_user[u]):
                    user_port_results = [r[2] for r in results_per_snr[snr_db] if r[0] == u and r[1] == p]
                    if user_port_results:
                        print(f"SNR {snr_db} dB, User {u}, Port {p}: Average NMSE = {np.mean(user_port_results):.2f} dB")
        
        # Plot NMSE vs SNR
        plt.figure(figsize=(10, 6))
        snrs = sorted(avg_nmse_per_snr.keys())
        nmses = [avg_nmse_per_snr[snr] for snr in snrs]
        plt.plot(snrs, nmses, 'bo-')
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('NMSE vs SNR')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'nmse_vs_snr.png'))
        
        # Print overall results
        print("\nOverall Results:")
        for snr_db in snrs:
            print(f"SNR {snr_db} dB: Average NMSE = {avg_nmse_per_snr[snr_db]:.2f} dB")
    
    # Generate visualizations for a few samples
    else:
        for i in range(3):  # Generate 3 sample visualizations
            # Generate sample
            sample = data_gen.generate_sample()
            ls_estimate = sample['ls_estimate']
            noise_power = sample['noise_power']
            true_channels = sample['true_channels']
            
            # Process through model
            with torch.no_grad():
                # Use trainable MMSE module if available
                if mmse_module:
                    # Extract channel statistics from ls_estimate
                    channel_stats = torch.abs(ls_estimate)
                    
                    # Get trainable C and R matrices
                    C, R = mmse_module(channel_stats, torch.tensor([noise_power], device=device))
                    
                    # Set MMSE matrices in estimator
                    srs_estimator.set_mmse_matrices(C=C, R=R)
                
                # Process through SRS estimator
                channel_estimates = srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=config.cyclic_shifts,
                    noise_power=noise_power
                )
            
            # Visualize results for each user/port
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
                        
                        # Visualize
                        plt.figure(figsize=(12, 10))
                        
                        # Plot magnitude
                        plt.subplot(2, 1, 1)
                        plt.plot(torch.abs(true_channel).cpu().numpy(), 'b-', label='True Channel')
                        plt.plot(torch.abs(est_channel).cpu().numpy(), 'r--', label='Estimated Channel')
                        plt.title(f"Sample {i+1}, User {u}, Port {p} - Magnitude (NMSE: {nmse:.2f} dB)")
                        plt.legend()
                        plt.grid(True)
                        
                        # Plot phase
                        plt.subplot(2, 1, 2)
                        plt.plot(torch.angle(true_channel).cpu().numpy(), 'b-', label='True Channel')
                        plt.plot(torch.angle(est_channel).cpu().numpy(), 'r--', label='Estimated Channel')
                        plt.title(f"Sample {i+1}, User {u}, Port {p} - Phase")
                        plt.legend()
                        plt.grid(True)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(save_dir, f"sample_{i+1}_user_{u}_port_{p}.png"))
                        plt.close()
                        
                        print(f"Sample {i+1}, User {u}, Port {p}: NMSE = {nmse:.2f} dB")
                    
                    idx += 1


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate SRS Channel Estimator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate')
    parser.add_argument('--snr_min', type=float, default=-5, help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=30, help='Maximum SNR in dB')
    parser.add_argument('--snr_step', type=float, default=5, help='SNR step size in dB')
    parser.add_argument('--save_dir', type=str, default='./results', help='Directory for saving results')
    args = parser.parse_args()
    
    # Generate SNR values
    snr_values = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step).tolist()
    
    # Evaluate model
    evaluate_model(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        snr_values=snr_values,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
