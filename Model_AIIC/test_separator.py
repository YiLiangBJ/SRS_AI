"""
Test script for Channel Separator models using existing SRS data generator

This script integrates the new channel separator models with the existing
SRS data generation framework.
"""

import sys
import os

# CPU optimization: Set thread count BEFORE importing torch/numpy
# Detect if running under numactl or in multi-NUMA system
try:
    # Get CPUs available to this process (respects numactl, cgroup, etc.)
    available_cpus = len(os.sched_getaffinity(0))
except AttributeError:
    # Fallback for systems without sched_getaffinity
    available_cpus = os.cpu_count()

# Use physical cores only, not hyperthreads
# For 2-socket SPR-EE: 224 logical cores = 112 physical cores
# Best practice: use physical cores only for CPU-intensive workloads
num_physical_cores = available_cpus // 2  # Assume 2-way SMT (hyperthreading)

# Use physical cores, limited to reasonable number
num_threads = min(num_physical_cores, 56)  # Cap at 56 for single NUMA node

# Override with environment variable if set
if 'OMP_NUM_THREADS' in os.environ:
    num_threads = int(os.environ['OMP_NUM_THREADS'])
    print(f"📌 Using OMP_NUM_THREADS from environment: {num_threads}")
else:
    # Set all threading environment variables
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

# Intel MKL optimizations for SPR
os.environ['KMP_BLOCKTIME'] = '0'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['KMP_HW_SUBSET'] = '1t'  # Use 1 thread per physical core (disable hyperthreading)

print(f"🚀 CPU Optimization:")
print(f"   Available CPUs: {available_cpus}")
print(f"   Physical cores: {num_physical_cores}")
print(f"   Using threads: {num_threads}")
print(f"   NUMA nodes: Run with 'numactl --hardware' to check")

import torch
import torch.nn.functional as F
import numpy as np

# Set PyTorch inter-op threads globally (can only be set once)
torch.set_num_interop_threads(1)

# No TensorFlow needed - using custom NumPy TDL
# print(f"✅ Using custom TDL channel (pure NumPy, no GIL limitation)")

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
    snr_db = 20.0,  # Can be: scalar, list [snr0, snr1, snr2, snr3], or tuple (min, max) for random
    seq_len: int = 12,
    num_ports: int = 4,
    tdl_config: str = 'A-30'  # Format: 'MODEL-DELAY_NS' e.g., 'A-30', 'B-100', 'C-300'
):
    """
    Generate training data with TDL channel and SNR control
    
    Args:
        snr_db: SNR in dB. 
                - Scalar: same SNR for all ports
                - List [snr0, snr1, snr2, snr3]: fixed SNR per port
                - Tuple (min, max): random SNR uniformly sampled per sample
        tdl_config: TDL configuration string 'MODEL-DELAY_NS'
                    e.g., 'A-30' = TDL-A with 30ns delay spread
                          'B-100' = TDL-B with 100ns delay spread
                    Or list of configs for random selection per sample
    
    Returns:
        y: (B, L) received signal with noise
        h_targets: (B, P, L) shifted channel targets (adjusted for SNR)
        pos_values: list of port positions
        h_true: (B, P, L) original channels (adjusted for SNR)
    """
    # Fixed port positions for 4 ports
    pos_values = [0, 3, 6, 9]
    
    # Parse TDL configuration
    # Support: single config string or list for random selection
    if isinstance(tdl_config, list):
        # Random selection per sample (each sample in batch gets random config)
        tdl_configs = tdl_config
        tdl_config = np.random.choice(tdl_configs)
    
    # Parse: 'A-30' -> model='A', delay_spread=30e-9
    parts = tdl_config.split('-')
    tdl_model = parts[0].upper()
    delay_ns = float(parts[1]) if len(parts) > 1 else 30
    delay_spread = delay_ns * 1e-9
    
    # Use custom TDL channel (pure NumPy, no GIL, independent fading)
    from Model_AIIC.tdl_channel import TDLChannel
    
    # Create TDL channel model
    tdl = TDLChannel(
        model=tdl_model,
        delay_spread=delay_spread,
        carrier_frequency=3.5e9,
        normalize=True
    )
    
    # Generate all channels at once: batch_size x num_ports
    # Each sample has independent random phases (no time correlation)
    scs = 30e3
    Ktc = 4
    sampling_rate = scs * Ktc * seq_len
    
    h_base = tdl.generate_batch_parallel(
        batch_size=batch_size,
        num_ports=num_ports,
        seq_len=seq_len,
        sampling_rate=sampling_rate,
        return_torch=True
    )
    
    # Normalize per port
    for p in range(num_ports):
        power = h_base[:, p].abs().pow(2).mean()
        if power > 0:
            h_base[:, p] = h_base[:, p] / power.sqrt()
    
    # Generate noise with unit power
    noise = (torch.randn(batch_size, seq_len) + 1j * torch.randn(batch_size, seq_len))
    noise = noise / noise.abs().pow(2).mean().sqrt()
    
    # Adjust signal power based on SNR
    if isinstance(snr_db, tuple) and len(snr_db) == 2:
        # Random SNR: (min, max) - uniform per sample
        snr_min, snr_max = snr_db
        h_true = torch.zeros_like(h_base)
        for b in range(batch_size):
            # Random SNR for this sample
            sample_snr = np.random.uniform(snr_min, snr_max)
            signal_power = torch.tensor(10 ** (sample_snr / 10))
            h_true[b] = h_base[b] * signal_power.sqrt()
    elif isinstance(snr_db, list):
        # Different SNR for each port (fixed)
        h_true = torch.zeros_like(h_base)
        for i in range(num_ports):
            signal_power = torch.tensor(10 ** (snr_db[i] / 10))
            h_true[:, i] = h_base[:, i] * signal_power.sqrt()
    else:
        # Same SNR for all ports (scalar)
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


def test_model(
    num_batches=100, 
    batch_size=32, 
    num_stages=3, 
    snr_db=20.0,  # Can be scalar, list, or tuple (min, max)
    share_weights=False,
    tdl_configs='A-30',  # Can be string or list of strings
    early_stop_loss=None,  # Stop if loss below this value
    validation_interval=100,  # Validate every N batches
    patience=5,  # Number of validation checks that must pass early stop threshold
    save_dir=None,  # Directory to save model and metrics
    exp_name=None  # Experiment name for this run
):
    """
    Test Residual Refinement Channel Separator with online training using TDL channels
    
    Args:
        num_batches: Maximum number of training batches
        batch_size: Batch size
        num_stages: Number of refinement stages
        snr_db: SNR configuration (scalar, list, or (min, max) tuple)
        share_weights: Whether to share weights across stages
        tdl_configs: TDL configuration(s). String like 'A-30' or list like ['A-30', 'B-100', 'C-300']
        early_stop_loss: If not None, stop training when loss < this value (with patience)
        validation_interval: How often to run validation
        patience: Number of consecutive validation passes needed for early stopping
        save_dir: Directory to save model and metrics (None = don't save)
        exp_name: Experiment name (used for subfolder naming)
    """
    # Get thread count from environment (set at module import)
    num_threads = int(os.environ.get('OMP_NUM_THREADS', 56))
    
    # Set PyTorch threading (only if not already set)
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    # Note: set_num_interop_threads can only be called once, so we skip it in loop
    
    print("="*80)
    print(f"Residual Refinement Channel Separator - Online Training")
    print("="*80)
    print(f"🔧 Thread Configuration:")
    print(f"   Available CPUs: {len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()}")
    print(f"   PyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"   PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    print(f"   OMP threads: {os.environ.get('OMP_NUM_THREADS', 'default')}")
    print(f"   Using custom NumPy TDL channel (no GIL limitation)")
    
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
    print(f"  Max training batches: {num_batches}")
    print(f"  TDL configs: {tdl_configs}")
    if early_stop_loss is not None:
        print(f"  Early stop loss: {early_stop_loss:.6f} (patience: {patience})")
        print(f"  Validation interval: {validation_interval}")
    
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
    print(f"💡 Tip: Use larger batch_size (e.g., 2048-4096) for better CPU utilization")
    model.train()
    losses = []
    val_losses = []
    
    import time
    start_time = time.time()
    data_gen_time = 0
    forward_time = 0
    backward_time = 0
    
    # Early stopping variables
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    for batch_idx in range(num_batches):
        # Generate batch on-the-fly using TDL
        t0 = time.time()
        y, h_targets, pos_values, h_true = generate_training_data(
            batch_size=batch_size, 
            snr_db=snr_db, 
            seq_len=seq_len, 
            num_ports=num_ports,
            tdl_config=tdl_configs
        )
        data_gen_time += time.time() - t0
        
        # Forward
        t0 = time.time()
        optimizer.zero_grad()
        h_pred = model(y)
        forward_time += time.time() - t0
        
        # Loss: NMSE on shifted targets
        mse = (h_pred - h_targets).abs().pow(2).mean()
        signal_power = h_targets.abs().pow(2).mean()
        nmse = mse / (signal_power + 1e-10)
        loss = nmse  # Can also use: 10 * torch.log10(nmse) for dB scale
        
        # Backward
        t0 = time.time()
        loss.backward()
        optimizer.step()
        backward_time += time.time() - t0
        
        losses.append(loss.item())
        
        # Print progress with throughput and timing breakdown
        if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * batch_size / elapsed if elapsed > 0 else 0
            
            # Calculate percentage of time spent in each phase
            total_time = data_gen_time + forward_time + backward_time
            if total_time > 0:
                data_pct = 100 * data_gen_time / total_time
                fwd_pct = 100 * forward_time / total_time
                bwd_pct = 100 * backward_time / total_time
                timing_info = f"[Data:{data_pct:.0f}% Fwd:{fwd_pct:.0f}% Bwd:{bwd_pct:.0f}%]"
            else:
                timing_info = ""
            
            print(f"  Batch {batch_idx+1}/{num_batches}, Loss: {loss.item():.6f}, "
                  f"Throughput: {samples_per_sec:.0f} samples/s {timing_info}")
        
        # Validation and early stopping check
        if early_stop_loss is not None and (batch_idx + 1) % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                # Run validation on multiple batches for stability
                val_loss_sum = 0
                val_batches = 5
                for _ in range(val_batches):
                    y_val, h_val, _, _ = generate_training_data(
                        batch_size=batch_size,
                        snr_db=snr_db,
                        seq_len=seq_len,
                        num_ports=num_ports,
                        tdl_config=tdl_configs
                    )
                    h_pred_val = model(y_val)
                    mse_val = (h_pred_val - h_val).abs().pow(2).mean()
                    signal_power_val = h_val.abs().pow(2).mean()
                    nmse_val = mse_val / (signal_power_val + 1e-10)
                    val_loss_sum += nmse_val.item()
                
                avg_val_loss = val_loss_sum / val_batches
                val_losses.append(avg_val_loss)
                
                print(f"  → Validation Loss: {avg_val_loss:.6f}")
                
                # Check early stopping condition
                if avg_val_loss < early_stop_loss:
                    early_stop_counter += 1
                    print(f"  → Early stop progress: {early_stop_counter}/{patience}")
                    
                    if early_stop_counter >= patience:
                        print(f"\n✓ Early stopping triggered! Val loss {avg_val_loss:.6f} < {early_stop_loss:.6f}")
                        print(f"  Stopped at batch {batch_idx+1}/{num_batches}")
                        break
                else:
                    early_stop_counter = 0  # Reset counter
                
                # Track best validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
            
            model.train()  # Back to training mode
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("Final Evaluation on Test Batch")
    print(f"{'='*80}")
    
    model.eval()
    final_train_loss = losses[-1] if losses else None
    best_val_loss = min(val_losses) if val_losses else None
    
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
        print(f"  Final train loss: {final_train_loss:.6f}")
        if best_val_loss:
            print(f"  Best val loss:    {best_val_loss:.6f}")
        print(f"  Avg loss (last 10): {sum(losses[-10:])/10:.6f}")
    
    # Save model and metrics if save_dir is provided
    if save_dir is not None:
        from pathlib import Path
        import json
        
        # Create save directory
        save_path = Path(save_dir)
        if exp_name:
            save_path = save_path / exp_name
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"Saving Model and Metrics")
        print(f"{'='*80}")
        print(f"Save directory: {save_path}")
        
        # Save PyTorch model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'seq_len': seq_len,
                'num_ports': num_ports,
                'hidden_dim': 64,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'normalize_energy': True
            },
            'hyperparameters': {
                'num_stages': num_stages,
                'share_weights': share_weights,
                'batch_size': batch_size,
                'snr_db': str(snr_db),
                'tdl_configs': tdl_configs
            },
            'losses': losses,
            'val_losses': val_losses,
            'final_train_loss': final_train_loss,
            'best_val_loss': best_val_loss,
            'test_nmse_shifted': nmse_shifted.item(),
            'test_nmse_unshifted': nmse_unshifted.item()
        }, save_path / 'model.pth')
        print(f"  ✓ Saved PyTorch model: {save_path / 'model.pth'}")
        
        # Save ONNX
        try:
            # Create dummy input for ONNX export
            dummy_input = torch.randn(1, seq_len, dtype=torch.complex64)
            
            # Export (note: ONNX doesn't support complex64 natively)
            # We'll export with float32 input/output for compatibility
            model_cpu = model.cpu()
            torch.onnx.export(
                model_cpu,
                dummy_input,
                save_path / 'model.onnx',
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            print(f"  ✓ Exported ONNX: {save_path / 'model.onnx'}")
        except Exception as e:
            print(f"  ✗ ONNX export failed: {e}")
        
        # Save metrics JSON
        metrics = {
            'hyperparameters': {
                'num_stages': num_stages,
                'share_weights': share_weights,
                'batch_size': batch_size,
                'max_batches': num_batches,
                'snr_db': str(snr_db),
                'tdl_configs': tdl_configs,
                'early_stop_loss': early_stop_loss,
                'validation_interval': validation_interval,
                'patience': patience
            },
            'results': {
                'final_train_loss': final_train_loss,
                'best_val_loss': best_val_loss,
                'test_nmse_shifted_db': nmse_shifted.item(),
                'test_nmse_unshifted_db': nmse_unshifted.item(),
                'port_wise_nmse_db': port_nmse,
                'num_epochs': len(losses),
                'stopped_early': len(losses) < num_batches
            },
            'timestamp': __import__('datetime').datetime.now().isoformat()
        }
        
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  ✓ Saved metrics: {save_path / 'metrics.json'}")
        
        # Save loss curves
        np.save(save_path / 'train_losses.npy', np.array(losses))
        if val_losses:
            np.save(save_path / 'val_losses.npy', np.array(val_losses))
        print(f"  ✓ Saved loss curves")
        
        print(f"{'='*80}\n")
    
    return model, losses


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=10000,
                       help='Number of training batches')
    parser.add_argument('--batch_size', type=int, default=2048,
                       help='Batch size (larger=better CPU utilization, e.g., 2048-4096 for 56 cores)')
    parser.add_argument('--stages', type=str, default='3',
                       help='Number of refinement stages. Single: "3", Multiple: "2,3,4"')
    parser.add_argument('--snr', type=str, default='20.0',
                       help='SNR in dB. Can be: scalar (e.g., "20"), range (e.g., "10,30"), or list (e.g., "[15,18,20,22]")')
    parser.add_argument('--tdl', type=str, default='A-30',
                       help='TDL config(s). Single: "A-30", Multiple: "A-30,B-100,C-300"')
    parser.add_argument('--share_weights', type=str, default='False',
                       help='Share weights across stages. Single: "True", Multiple: "True,False"')
    parser.add_argument('--early_stop', type=float, default=None,
                       help='Early stop threshold for loss')
    parser.add_argument('--val_interval', type=int, default=100,
                       help='Validation interval (batches)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save models and metrics (None = don\'t save)')
    
    args = parser.parse_args()
    
    # Parse SNR argument
    snr_str = args.snr.strip()
    if ',' in snr_str and not snr_str.startswith('['):
        # Range: "10,30" -> (10, 30)
        parts = [float(x.strip()) for x in snr_str.split(',')]
        snr_db = tuple(parts)
    elif snr_str.startswith('['):
        # List: "[15,18,20,22]" -> [15, 18, 20, 22]
        snr_db = eval(snr_str)
    else:
        # Scalar: "20" -> 20.0
        snr_db = float(snr_str)
    
    # Parse TDL configs
    tdl_configs = args.tdl.strip()
    if ',' in tdl_configs:
        # Multiple configs: "A-30,B-100,C-300" -> ['A-30', 'B-100', 'C-300']
        tdl_configs = [x.strip() for x in tdl_configs.split(',')]
    
    # Parse hyperparameter lists
    stages_list = [int(x.strip()) for x in args.stages.split(',')]
    share_weights_list = [x.strip().lower() == 'true' for x in args.share_weights.split(',')]
    
    # Generate all hyperparameter combinations
    from itertools import product
    hyperparameter_combinations = list(product(stages_list, share_weights_list))
    
    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Configuration")
    print(f"{'='*80}")
    print(f"Total combinations: {len(hyperparameter_combinations)}")
    print(f"  stages: {stages_list}")
    print(f"  share_weights: {share_weights_list}")
    print(f"Common settings:")
    print(f"  SNR: {snr_db}")
    print(f"  TDL: {tdl_configs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max batches: {args.batches}")
    if args.early_stop:
        print(f"  Early stop: {args.early_stop}")
    if args.save_dir:
        print(f"  Save directory: {args.save_dir}")
    print(f"{'='*80}\n")
    
    # Run all combinations
    results = []
    for idx, (num_stages, share_weights) in enumerate(hyperparameter_combinations):
        exp_name = f"stages={num_stages}_share={share_weights}"
        
        print(f"\n{'#'*80}")
        print(f"# Experiment {idx+1}/{len(hyperparameter_combinations)}: {exp_name}")
        print(f"{'#'*80}\n")
        
        try:
            model, losses = test_model(
                num_batches=args.batches,
                batch_size=args.batch_size,
                num_stages=num_stages,
                snr_db=snr_db,
                share_weights=share_weights,
                tdl_configs=tdl_configs,
                early_stop_loss=args.early_stop,
                validation_interval=args.val_interval,
                patience=args.patience,
                save_dir=args.save_dir,
                exp_name=exp_name
            )
            
            results.append({
                'experiment': exp_name,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'final_loss': losses[-1] if losses else None,
                'min_loss': min(losses) if losses else None,
                'num_epochs': len(losses),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"\n✗ Experiment {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'experiment': exp_name,
                'num_stages': num_stages,
                'share_weights': share_weights,
                'status': 'failed',
                'error': str(e)
            })
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"All Experiments Complete!")
    print(f"{'='*80}")
    print(f"Results Summary:")
    print(f"{'-'*80}")
    
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        # Sort by min_loss
        sorted_results = sorted(successful_results, key=lambda x: x['min_loss'])
        
        print(f"\n{'Rank':<6} {'Experiment':<30} {'Min Loss':<12} {'Final Loss':<12} {'Epochs':<8}")
        print(f"{'-'*80}")
        for i, result in enumerate(sorted_results):
            print(f"{i+1:<6} {result['experiment']:<30} "
                  f"{result['min_loss']:<12.6f} {result['final_loss']:<12.6f} {result['num_epochs']:<8}")
        
        print(f"\nBest configuration: {sorted_results[0]['experiment']}")
        print(f"  Min Loss: {sorted_results[0]['min_loss']:.6f}")
    
    failed_results = [r for r in results if r['status'] == 'failed']
    if failed_results:
        print(f"\nFailed experiments: {len(failed_results)}")
        for result in failed_results:
            print(f"  - {result['experiment']}: {result['error']}")
    
    # Save summary if save_dir provided
    if args.save_dir:
        from pathlib import Path
        import json
        
        summary_path = Path(args.save_dir) / 'search_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'hyperparameters': {
                    'stages': stages_list,
                    'share_weights': share_weights_list,
                    'snr_db': str(snr_db),
                    'tdl_configs': tdl_configs,
                    'batch_size': args.batch_size,
                    'max_batches': args.batches
                },
                'results': results
            }, f, indent=2)
        print(f"\n✓ Summary saved to: {summary_path}")
    
    print(f"\n{'='*80}")
    print("✓ Test completed successfully!")
    print(f"{'='*80}")
