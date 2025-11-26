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
    snr_db = 20.0,  # Can be scalar or list [snr0, snr1, snr2, snr3]
    seq_len: int = 12,
    num_ports: int = 4
):
    """
    Generate training data with SNR control
    
    Args:
        snr_db: SNR in dB. Scalar for all ports, or list for per-port SNR
    
    Returns:
        y: (B, L) received signal with noise
        h_targets: (B, P, L) shifted channel targets (adjusted for SNR)
        pos_values: list of port positions
        h_true: (B, P, L) original channels (adjusted for SNR)
    """
    # Fixed port positions for 4 ports
    pos_values = [0, 2, 6, 8]
    
    # Generate base channels
    h_base = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64)
    
    # Generate noise with unit power
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()
    
    # Adjust signal power based on SNR
    if isinstance(snr_db, (list, tuple)):
        # Different SNR for each port
        h_true = torch.zeros_like(h_base)
        for i in range(num_ports):
            signal_power = 10 ** (snr_db[i] / 10)
            h_true[:, i] = h_base[:, i] * signal_power.sqrt()
    else:
        # Same SNR for all ports
        signal_power = 10 ** (snr_db / 10)
        h_true = h_base * signal_power.sqrt()
    
    # Create mixed signal with shifted channels
    y_clean = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    h_targets = []
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_true[:, i], shifts=pos, dims=-1)
        y_clean += shifted
        h_targets.append(shifted)
    h_targets = torch.stack(h_targets, dim=1)
    
    # Add noise
    y = y_clean + noise
    
    return y, h_targets, pos_values, h_true


def test_model(num_epochs=10, num_stages=3, share_weights=False, normalize_energy=True):
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
        share_weights_across_stages=share_weights,
        normalize_energy=normalize_energy
    )
    
    print(f"  Share weights: {share_weights}")
    print(f"  Normalize energy: {normalize_energy}")
    
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
        
        # Calculate NMSE (逐点比较)
        mse_shifted = (h_pred - h_targets_test).abs().pow(2).mean()
        signal_power_shifted = h_targets_test.abs().pow(2).mean()
        nmse_shifted = 10 * torch.log10(mse_shifted / (signal_power_shifted + 1e-10))
        
        mse_unshifted = (h_unshifted - h_true).abs().pow(2).mean()
        signal_power_unshifted = h_true.abs().pow(2).mean()
        nmse_unshifted = 10 * torch.log10(mse_unshifted / (signal_power_unshifted + 1e-10))
        
        # Check reconstruction (optional)
        y_recon = h_pred.sum(dim=1)
        recon_mse = (y_test - y_recon).abs().pow(2).mean()
        
        print(f"  NMSE (shifted targets):   {nmse_shifted:.2f} dB")
        print(f"  NMSE (unshifted channels): {nmse_unshifted:.2f} dB")
        print(f"  Reconstruction MSE:        {recon_mse:.6f}")
    
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
    parser.add_argument('--no-normalize', action='store_true',
                       help='Disable energy normalization')
    
    args = parser.parse_args()
    
    # Test model
    model, losses = test_model(
        num_epochs=args.epochs,
        num_stages=args.stages,
        share_weights=args.share,
        normalize_energy=not args.no_normalize
    )
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}")
