import torch
import matplotlib.pyplot as plt
import os
import argparse
import time
from tqdm import tqdm
import numpy as np

from config import create_example_config
from train_debug import SRSTrainer
from train_debug_h_with_residual import SRSTrainerModified


def run_benchmark(num_batches: int = 10, batch_size: int = 32, use_cuda: bool = True):
    """
    Run a benchmark comparison between the original approach and the h_with_residual approach
    
    Args:
        num_batches: Number of batches to process
        batch_size: Batch size for testing
        use_cuda: Whether to use CUDA
    """
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    print(f"Running benchmark on {device}")
    
    # Create configuration
    config = create_example_config()
    
    # Create trainers
    trainer_original = SRSTrainer(
        config=config, 
        device=device,
        save_dir="./benchmark_original",
        use_trainable_mmse=True,
        enable_plotting=False
    )
    
    trainer_modified = SRSTrainerModified(
        config=config, 
        device=device,
        save_dir="./benchmark_modified",
        use_trainable_mmse=True,
        enable_plotting=False
    )
    
    # Benchmark results
    original_times = []
    modified_times = []
    original_nmse = []
    modified_nmse = []
    
    # Run benchmark
    print("\nRunning original approach benchmark...")
    for _ in tqdm(range(num_batches)):
        # Generate batch
        batch = trainer_original.data_gen.generate_batch(batch_size)
        ls_estimates = batch['ls_estimates']
        noise_powers = batch['noise_powers']
        true_channels = batch['true_channels']
        
        # Measure time for original approach
        start_time = time.time()
        batch_nmse = 0
        
        with torch.no_grad():  # No need for gradients in benchmarking
            for i in range(batch_size):
                ls_estimate = ls_estimates[i]
                noise_power = noise_powers[i].item()
                
                # Use trainable MMSE module
                C, R = trainer_original.mmse_module(ls_estimate)
                trainer_original.srs_estimator.set_mmse_matrices(C=C, R=R)
                
                # Process through SRS estimator
                channel_estimates = trainer_original.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=config.cyclic_shifts,
                    noise_power=noise_power
                )
                
                # Calculate NMSE for each user/port
                idx = 0
                for u in range(config.num_users):
                    for p in range(config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            true_channel = true_channels[(u, p)][i]
                            est_channel = channel_estimates[idx]
                            
                            # Calculate NMSE
                            from utils import calculate_nmse
                            nmse = calculate_nmse(true_channel, est_channel)
                            batch_nmse += nmse
                        idx += 1
        
        original_times.append(time.time() - start_time)
        original_nmse.append(batch_nmse / (batch_size * sum(1 for u in range(config.num_users) 
                                                     for p in range(config.ports_per_user[u]) 
                                                     if (u, p) in true_channels)))
    
    print("\nRunning modified approach benchmark (h_with_residual/phasor)...")
    for _ in tqdm(range(num_batches)):
        # Generate batch (reuse the same batch for fair comparison)
        batch = trainer_modified.data_gen.generate_batch(batch_size)
        ls_estimates = batch['ls_estimates']
        noise_powers = batch['noise_powers']
        true_channels = batch['true_channels']
        
        # Measure time for modified approach
        start_time = time.time()
        batch_nmse = 0
        
        with torch.no_grad():  # No need for gradients in benchmarking
            for i in range(batch_size):
                ls_estimate = ls_estimates[i]
                noise_power = noise_powers[i].item()
                
                # First run SRS estimator to generate h_with_residual/phasor
                channel_estimates_initial = trainer_modified.srs_estimator(
                    ls_estimate=ls_estimate,
                    cyclic_shifts=config.cyclic_shifts,
                    noise_power=noise_power
                )
                
                # Check if h_with_residual/phasor is available
                if trainer_modified.srs_estimator.current_h_with_residual_phasor is not None:
                    # Use h_with_residual/phasor as input to generate MMSE matrices
                    C, R = trainer_modified.mmse_module(trainer_modified.srs_estimator.current_h_with_residual_phasor)
                    
                    # Set MMSE matrices
                    trainer_modified.srs_estimator.set_mmse_matrices(C=C, R=R)
                    
                    # Run estimator again with new matrices
                    channel_estimates = trainer_modified.srs_estimator(
                        ls_estimate=ls_estimate,
                        cyclic_shifts=config.cyclic_shifts,
                        noise_power=noise_power
                    )
                else:
                    channel_estimates = channel_estimates_initial
                
                # Calculate NMSE for each user/port
                idx = 0
                for u in range(config.num_users):
                    for p in range(config.ports_per_user[u]):
                        if (u, p) in true_channels:
                            true_channel = true_channels[(u, p)][i]
                            est_channel = channel_estimates[idx]
                            
                            # Calculate NMSE
                            from utils import calculate_nmse
                            nmse = calculate_nmse(true_channel, est_channel)
                            batch_nmse += nmse
                        idx += 1
        
        modified_times.append(time.time() - start_time)
        modified_nmse.append(batch_nmse / (batch_size * sum(1 for u in range(config.num_users) 
                                                     for p in range(config.ports_per_user[u]) 
                                                     if (u, p) in true_channels)))
    
    # Calculate statistics
    avg_original_time = np.mean(original_times)
    avg_modified_time = np.mean(modified_times)
    avg_original_nmse = np.mean(original_nmse)
    avg_modified_nmse = np.mean(modified_nmse)
    
    # Print results
    print("\n===== Benchmark Results =====")
    print(f"Original Approach:")
    print(f"  - Average Processing Time: {avg_original_time:.6f} sec per batch")
    print(f"  - Average NMSE: {avg_original_nmse:.2f} dB")
    print(f"Modified Approach (h_with_residual/phasor):")
    print(f"  - Average Processing Time: {avg_modified_time:.6f} sec per batch")
    print(f"  - Average NMSE: {avg_modified_nmse:.2f} dB")
    print(f"Time Increase: {(avg_modified_time / avg_original_time - 1) * 100:.2f}%")
    print(f"NMSE Improvement: {avg_original_nmse - avg_modified_nmse:.2f} dB")
    
    # Create plots
    plt.figure(figsize=(14, 6))
    
    # Plot processing times
    plt.subplot(1, 2, 1)
    plt.bar(['Original', 'Modified'], [avg_original_time, avg_modified_time])
    plt.title('Average Processing Time (sec)')
    plt.ylabel('Time (seconds)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Plot NMSE
    plt.subplot(1, 2, 2)
    plt.bar(['Original', 'Modified'], [avg_original_nmse, avg_modified_nmse])
    plt.title('Average NMSE (dB)')
    plt.ylabel('NMSE (dB)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("Benchmark results saved to 'benchmark_results.png'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark original vs h_with_residual approach")
    parser.add_argument('--batches', type=int, default=10, help='Number of batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    run_benchmark(
        num_batches=args.batches,
        batch_size=args.batch_size,
        use_cuda=not args.cpu
    )
