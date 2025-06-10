import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np

from config import create_example_config
from data_generator import SRSDataGenerator
from model import SRSChannelEstimator
from model_cholesky import TrainableMMSEModule
from utils import calculate_nmse


def test_gradient_flow():
    """
    Test gradient flow through MMSE module when using h_with_residual/phasor as input
    """
    print("Testing gradient flow through MMSE module using h_with_residual/phasor...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create config and models
    config = create_example_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create data generator, estimator, and MMSE module
    data_gen = SRSDataGenerator(config, device=device)
    
    srs_estimator = SRSChannelEstimator(
        seq_length=config.seq_length,
        ktc=config.ktc,
        max_users=config.num_users,
        max_ports_per_user=max(config.ports_per_user),
        mmse_block_size=config.mmse_block_size,
        device=device
    ).to(device)
    
    mmse_module = TrainableMMSEModule(
        seq_length=config.seq_length,
        mmse_block_size=config.mmse_block_size,
        use_complex_input=True
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(list(srs_estimator.parameters()) + list(mmse_module.parameters()), lr=0.001)
    
    # Generate a single sample for testing
    batch = data_gen.generate_batch(1)
    ls_estimate = batch['ls_estimates'][0]
    noise_power = batch['noise_powers'][0].item()
    true_channels = batch['true_channels']
    
    # Track gradients for different network parts
    mmse_grads_ls = []     # Using ls_estimate as input
    mmse_grads_h = []      # Using h_with_residual/phasor as input
    
    # Number of iterations to test
    n_iters = 5
    
    print("\nTesting with ls_estimate as input...")
    for i in range(n_iters):
        # Clear gradients
        optimizer.zero_grad()
        
        # Use ls_estimate as input to MMSE module
        C, R = mmse_module(ls_estimate)
        srs_estimator.set_mmse_matrices(C=C, R=R)
        
        # Process through SRS estimator
        channel_estimates = srs_estimator(
            ls_estimate=ls_estimate,
            cyclic_shifts=config.cyclic_shifts,
            noise_power=noise_power
        )
        
        # Calculate loss
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        idx = 0
        for u in range(config.num_users):
            for p in range(config.ports_per_user[u]):
                if (u, p) in true_channels:
                    true_channel = true_channels[(u, p)][0]
                    est_channel = channel_estimates[idx]
                    
                    # Calculate MSE loss
                    real_loss = torch.mean((torch.real(est_channel) - torch.real(true_channel))**2)
                    imag_loss = torch.mean((torch.imag(est_channel) - torch.imag(true_channel))**2)
                    loss = real_loss + imag_loss
                    
                    total_loss = total_loss + loss
                idx += 1
        
        # Backward pass
        total_loss.backward()
        
        # Record MMSE gradients
        avg_grad_norm = 0
        param_count = 0
        for name, param in mmse_module.named_parameters():
            if param.grad is not None:
                avg_grad_norm += param.grad.abs().mean().item()
                param_count += 1
                
        if param_count > 0:
            avg_grad_norm /= param_count
            mmse_grads_ls.append(avg_grad_norm)
            print(f"Iteration {i+1}, Average MMSE gradient norm (ls_estimate): {avg_grad_norm:.6f}")
    
    print("\nTesting with h_with_residual/phasor as input...")
    for i in range(n_iters):
        # Clear gradients
        optimizer.zero_grad()
        
        # First pass to get h_with_residual/phasor
        _ = srs_estimator(
            ls_estimate=ls_estimate,
            cyclic_shifts=config.cyclic_shifts,
            noise_power=noise_power
        )
        
        # Get h_with_residual/phasor
        if srs_estimator.current_h_with_residual_phasor is not None:
            # Use h_with_residual/phasor as input to MMSE module
            C, R = mmse_module(srs_estimator.current_h_with_residual_phasor)
            srs_estimator.set_mmse_matrices(C=C, R=R)
            
            # Second pass with updated MMSE matrices
            channel_estimates = srs_estimator(
                ls_estimate=ls_estimate,
                cyclic_shifts=config.cyclic_shifts,
                noise_power=noise_power
            )
            
            # Calculate loss
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            idx = 0
            for u in range(config.num_users):
                for p in range(config.ports_per_user[u]):
                    if (u, p) in true_channels:
                        true_channel = true_channels[(u, p)][0]
                        est_channel = channel_estimates[idx]
                        
                        # Calculate MSE loss
                        real_loss = torch.mean((torch.real(est_channel) - torch.real(true_channel))**2)
                        imag_loss = torch.mean((torch.imag(est_channel) - torch.imag(true_channel))**2)
                        loss = real_loss + imag_loss
                        
                        total_loss = total_loss + loss
                    idx += 1
            
            # Backward pass
            total_loss.backward()
            
            # Record MMSE gradients
            avg_grad_norm = 0
            param_count = 0
            for name, param in mmse_module.named_parameters():
                if param.grad is not None:
                    avg_grad_norm += param.grad.abs().mean().item()
                    param_count += 1
                    
            if param_count > 0:
                avg_grad_norm /= param_count
                mmse_grads_h.append(avg_grad_norm)
                print(f"Iteration {i+1}, Average MMSE gradient norm (h_with_residual): {avg_grad_norm:.6f}")
        else:
            print("Warning: h_with_residual/phasor not available")
    
    # Plot gradient norms
    plt.figure(figsize=(10, 5))
    x = np.arange(1, n_iters + 1)
    
    if mmse_grads_ls:
        plt.plot(x, mmse_grads_ls, 'b-o', label='Using ls_estimate')
    if mmse_grads_h:
        plt.plot(x, mmse_grads_h, 'r-s', label='Using h_with_residual/phasor')
    
    plt.title('MMSE Module Gradient Norms')
    plt.xlabel('Iteration')
    plt.ylabel('Average Gradient Norm')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png')
    print("Gradient flow comparison saved to 'gradient_flow_comparison.png'")


if __name__ == "__main__":
    test_gradient_flow()
