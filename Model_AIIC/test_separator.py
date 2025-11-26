"""
Test script for Channel Separator models using existing SRS data generator

This script integrates the new channel separator models with the existing
SRS data generation framework.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC.channel_separator import ResidualRefinementSeparator
from data_generator import BaseSRSDataGenerator
from user_config import SRSConfig, create_example_config
from system_config import create_default_system_config
from utils import calculate_nmse


def generate_training_data(
    srs_config: SRSConfig = None,
    batch_size: int = 32,
    snr_db: float = 20.0,
    seq_len: int = 12,
    num_ports: int = 4
):
    """
    Generate training data with fixed dimensions (simplified version)
    
    Returns:
        y: (B, L) received signal in time domain
        h_targets: (B, P, L) shifted channel targets
        pos_values: list of port positions
        h_true: (B, P, L) original channels
    """
    # Fixed port positions for 4 ports
    pos_values = [0, 2, 6, 8]
    
    # Generate simple random channels (frequency domain)
    # In practice, these should come from channel models
    h_freq = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64) * 0.3
    
    # Frequency domain: y_freq = sum(M_p * c_p)
    # where M_p is the phase rotation matrix
    y_freq = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    
    for p_idx, pos in enumerate(pos_values):
        # Phase rotation in frequency domain
        n = torch.arange(seq_len, dtype=torch.float32)
        phase = torch.exp(1j * 2 * np.pi * pos * n / seq_len)
        
        # Apply phase rotation and accumulate
        y_freq += h_freq[:, p_idx] * phase
    
    # Convert to time domain
    y_time = torch.fft.ifft(y_freq, dim=-1)
    
    # Add noise
    noise_power = 10 ** (-snr_db / 10)
    noise = (torch.randn_like(y_time.real) + 1j * torch.randn_like(y_time.imag)) * np.sqrt(noise_power / 2)
    y_time = y_time + noise
    
    # Create targets (shifted channels in time domain)
    h_time = torch.fft.ifft(h_freq, dim=-1)
    h_targets = torch.stack([
        torch.roll(h_time[:, p_idx], shifts=pos, dims=-1)
        for p_idx, pos in enumerate(pos_values)
    ], dim=1)
    
    return y_time, h_targets, pos_values, h_time


def test_model(num_epochs=10, num_stages=3, share_weights=False):
    """
    Test Residual Refinement Channel Separator
    """
    print("="*80)
    print(f"Testing Residual Refinement Channel Separator")
    print("="*80)
    
    # Create configuration
    srs_config = create_example_config()
    
    # IMPORTANT: Force seq_len = 12 for our discussion
    seq_len = 12
    num_ports = 4  # Fixed for our test case
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of ports: {num_ports}")
    print(f"  Num stages: {num_stages}")
    print(f"  Users: {srs_config.num_users}")
    
    # Create model
    model = ResidualRefinementSeparator(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=64,
        num_stages=num_stages,
        share_weights_across_stages=share_weights
    )
    
    print(f"  Share weights: {share_weights}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    losses = []
    
    for epoch in range(num_epochs):
        # Generate batch
        y, h_targets, pos_values, h_true = generate_training_data(
            batch_size=32, snr_db=20.0, seq_len=seq_len, num_ports=num_ports
        )
        
        # Forward pass
        optimizer.zero_grad()
        h_pred = model(y)
        
        # Loss: MSE between predicted and target (shifted)
        # Complex MSE = MSE(real) + MSE(imag)
        loss = torch.nn.functional.mse_loss(h_pred.real, h_targets.real) + \
               torch.nn.functional.mse_loss(h_pred.imag, h_targets.imag)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    # Evaluation
    print(f"\n{'='*80}")
    print("Evaluation")
    print(f"{'='*80}")
    
    model.eval()
    with torch.no_grad():
        # Generate test batch
        y_test, h_targets_test, pos_values, h_true = generate_training_data(
            batch_size=100, snr_db=20.0, seq_len=seq_len, num_ports=num_ports
        )
        
        # Predict
        h_pred = model(y_test)
        
        # Get unshifted channels
        h_unshifted = model.get_unshifted_channels(h_pred, pos_values)
        
        # Calculate NMSE
        nmse_shifted = calculate_nmse(h_pred, h_targets_test)
        nmse_unshifted = calculate_nmse(h_unshifted, h_true)
        
        # Check reconstruction
        y_recon = h_pred.sum(dim=1)
        recon_error = (y_test - y_recon).abs().mean()
        
        print(f"  NMSE (shifted targets): {nmse_shifted:.4f} dB")
        print(f"  NMSE (unshifted channels): {nmse_unshifted:.4f} dB")
        print(f"  Reconstruction error: {recon_error:.6f}")
        print(f"  sum(h_pred) ≈ y: {'✓' if recon_error < 0.01 else '✗'}")
    
    return model, losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--stages', type=int, default=3,
                       help='Number of refinement stages')
    parser.add_argument('--share', action='store_true',
                       help='Share weights across stages for same port')
    
    args = parser.parse_args()
    
    # Test model
    model, losses = test_model(
        num_epochs=args.epochs,
        num_stages=args.stages,
        share_weights=args.share
    )
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}")
