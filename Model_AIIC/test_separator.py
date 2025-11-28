"""
Test script for Channel Separator models using existing SRS data generator

This script integrates the new channel separator models with the existing
SRS data generation framework.
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model_AIIC.channel_separator import ResidualRefinementSeparator
from Model_AIIC.channel_models import TDLChannelGenerator, SimpleRayleighChannel
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
    
    # Try to use TDL channel generator first
    use_tdl = False
    try:
        channel_gen = TDLChannelGenerator(
            tdl_model='A',
            delay_spread=30e-9,
            scs=30e3,
            Ktc=4,
            seq_len=seq_len,
            add_timing_offset=True,
            normalize_energy=False  # We'll normalize ourselves
        )
        h_base = channel_gen.generate(batch_size=batch_size, num_ports=num_ports)
        use_tdl = True
        print(f"Using TDL-A channel model")
    except Exception as e:
        print(f"TDL channel generation failed ({e}), using Rayleigh fallback")
        # Fallback: simple random channels
        h_base = torch.randn(batch_size, num_ports, seq_len, dtype=torch.complex64)
    
    # Generate noise with unit power
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()
    
    # Adjust signal power based on SNR
    if isinstance(snr_db, (list, tuple)):
        # Different SNR for each port
        h_true = torch.zeros_like(h_base)
        for i in range(num_ports):
            signal_power = torch.tensor(10 ** (snr_db[i] / 10))
            h_true[:, i] = h_base[:, i] * signal_power.sqrt()
    else:
        # Same SNR for all ports
        signal_power = torch.tensor(10 ** (snr_db / 10))
        h_true = h_base * signal_power.sqrt()
    
    # Create mixed signal with shifted channels
    y_clean = torch.zeros(batch_size, seq_len, dtype=torch.complex64)
    h_targets = []
    for i, pos in enumerate(pos_values):
        shifted = torch.roll(h_true[:, i], shifts=-pos, dims=-1)
        y_clean += shifted
        h_targets.append(shifted)
    h_targets = torch.stack(h_targets, dim=1)
    
    # Add noise
    y = y_clean + noise
    
    return y, h_targets, pos_values, h_true


def test_model(num_batches=100, batch_size=32, num_stages=3, snr_db=20.0, share_weights=False):
    """
    Test Residual Refinement Channel Separator with online training
    """
    print("="*80)
    print(f"Residual Refinement Channel Separator - Online Training")
    print("="*80)
    
    # Configuration
    seq_len = 12
    num_ports = 4
    
    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of ports: {num_ports}")
    print(f"  Num stages: {num_stages}")
    print(f"  Share weights: {share_weights}")
    print(f"  SNR: {snr_db} dB")
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {num_batches}")
    
    # Create model
    model = ResidualRefinementSeparator(
        seq_len=seq_len,
        num_ports=num_ports,
        hidden_dim=64,
        num_stages=num_stages,
        share_weights_across_stages=share_weights,
        normalize_energy=True
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Online Training
    print(f"\nOnline Training...")
    model.train()
    losses = []
    
    for batch_idx in range(num_batches):
        # Generate batch on-the-fly
        y, h_targets, pos_values, h_true = generate_training_data(
            batch_size=batch_size, 
            snr_db=snr_db, 
            seq_len=seq_len, 
            num_ports=num_ports
        )
        
        # Forward
        optimizer.zero_grad()
        h_pred = model(y)
        
        # Loss: NMSE on shifted targets
        mse = (h_pred - h_targets).abs().pow(2).mean()
        signal_power = h_targets.abs().pow(2).mean()
        nmse = mse / (signal_power + 1e-10)
        loss = nmse  # Can also use: 10 * torch.log10(nmse) for dB scale
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Print progress
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.6f}")
    
    # Evaluation
    print(f"\n{'='*80}")
    print("Evaluation on Test Batch")
    print(f"{'='*80}")
    
    model.eval()
    with torch.no_grad():
        # Generate test batch
        y_test, h_targets_test, pos_values, h_true_test = generate_training_data(
            batch_size=200, 
            snr_db=snr_db, 
            seq_len=seq_len, 
            num_ports=num_ports
        )
        
        # Predict
        h_pred = model(y_test)
        
        # Get unshifted channels
        h_unshifted = model.get_unshifted_channels(h_pred, pos_values)
        
        # NMSE on shifted targets
        mse_shifted = (h_pred - h_targets_test).abs().pow(2).mean()
        signal_power = h_targets_test.abs().pow(2).mean()
        nmse_shifted = 10 * torch.log10(mse_shifted / (signal_power + 1e-10))
        
        # NMSE on unshifted channels
        mse_unshifted = (h_unshifted - h_true_test).abs().pow(2).mean()
        signal_power_true = h_true_test.abs().pow(2).mean()
        nmse_unshifted = 10 * torch.log10(mse_unshifted / (signal_power_true + 1e-10))
        
        # Port-wise NMSE
        port_nmse = []
        for p in range(num_ports):
            mse_p = (h_pred[:, p] - h_targets_test[:, p]).abs().pow(2).mean()
            power_p = h_targets_test[:, p].abs().pow(2).mean()
            nmse_p = 10 * torch.log10(mse_p / (power_p + 1e-10))
            port_nmse.append(nmse_p.item())
        
        print(f"  NMSE (shifted):   {nmse_shifted:.2f} dB")
        print(f"  NMSE (unshifted): {nmse_unshifted:.2f} dB")
        print(f"  Port-wise NMSE:   {[f'{x:.2f}' for x in port_nmse]} dB")
        print(f"  Final loss:       {losses[-1]:.6f}")
        print(f"  Avg loss (last 10): {sum(losses[-10:])/10:.6f}")
    
    return model, losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=10000,
                       help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--stages', type=int, default=3,
                       help='Number of refinement stages')
    parser.add_argument('--snr', type=float, default=20.0,
                       help='SNR in dB')
    parser.add_argument('--share', action='store_true',
                       help='Share weights across stages for same port')
    
    args = parser.parse_args()
    
    # Test model
    model, losses = test_model(
        num_batches=args.batches,
        batch_size=args.batch_size,
        num_stages=args.stages,
        snr_db=args.snr,
        share_weights=args.share
    )
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}")
