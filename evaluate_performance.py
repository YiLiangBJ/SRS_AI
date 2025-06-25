import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Optional, Tuple, Literal
import pandas as pd
from tqdm import tqdm

from config import SRSConfig, create_example_config
from data_generator import SRSDataGenerator
from model_Traditional import SRSChannelEstimator
from model_AIpart import TrainableMMSEModule
from utils import calculate_nmse


class SRSEvaluator:
    """
    Class for evaluating SRS Channel Estimator performance across different
    SNR values and channel models
    """
    def __init__(
        self,
        config: SRSConfig,
        checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_trainable_mmse: bool = True
    ):
        """
        Initialize evaluator
        
        Args:
            config: SRS configuration
            checkpoint_path: Path to trained model checkpoint
            device: Computation device
            use_trainable_mmse: Whether to use trainable MMSE
        """
        self.config = config
        self.device = device
        
        # Create MMSE module if needed
        if use_trainable_mmse:
            self.mmse_module = TrainableMMSEModule(
                seq_length=config.seq_length,
                mmse_block_size=config.mmse_block_size,
                use_complex_input=True
            ).to(device)
        else:
            self.mmse_module = None
        
        # Create SRS channel estimator
        self.srs_estimator = SRSChannelEstimator(
            seq_length=config.seq_length,
            ktc=config.ktc,
            max_users=config.num_users,
            max_ports_per_user=max(config.ports_per_user),
            mmse_block_size=config.mmse_block_size,
            device=device,
            mmse_module=self.mmse_module
        ).to(device)
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set models to evaluation mode
        self.srs_estimator.eval()
        if self.mmse_module:
            self.mmse_module.eval()
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.srs_estimator.load_state_dict(checkpoint['model_state_dict'])
        
        # Load MMSE module state if available
        if self.mmse_module and 'mmse_state_dict' in checkpoint:
            self.mmse_module.load_state_dict(checkpoint['mmse_state_dict'])
    
    def evaluate_model(
        self,
        channel_models: List[str] = ["simple", "TDL-A", "TDL-B", "TDL-C"],
        snr_values: List[float] = [0, 5, 10, 15, 20, 25, 30],
        num_samples: int = 100,
        batch_size: int = 10,
        delay_spread: float = 100e-9,  # 100 ns
        sampling_rate: float = 15.36e6,  # 15.36 MHz
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Evaluate model performance across different SNR values and channel models
        
        Args:
            channel_models: List of channel models to evaluate
            snr_values: List of SNR values (in dB) to evaluate
            num_samples: Number of samples to evaluate for each condition
            batch_size: Batch size for evaluation
            delay_spread: Delay spread in seconds (for TDL models)
            sampling_rate: Sampling rate in Hz (for TDL models)
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Loop through channel models
        for channel_model in channel_models:
            print(f"\nEvaluating channel model: {channel_model}")
              # Create data generator with appropriate channel model
            data_gen = SRSDataGenerator(
                config=self.config,
                channel_model=channel_model,
                delay_spread=delay_spread,
                sampling_rate=sampling_rate,
                delta_f=30e3,        # 30 kHz subcarrier spacing
                ifft_size=4096,      # IFFT size for 5G
                cp_length=288,       # Cyclic prefix length
                ktc=4,              # Parameter for subcarrier mapping
                timing_offset_range=(-130e-9, 130e-9),  # -130ns to 130ns timing offset
                device=self.device,
                seed=seed
            )
            
            # Loop through SNR values
            for snr_db in snr_values:
                print(f"  Evaluating SNR: {snr_db} dB")
                
                total_nmse = 0
                total_samples = 0
                
                # Calculate number of batches
                num_batches = (num_samples + batch_size - 1) // batch_size
                
                # Evaluate in batches
                for _ in tqdm(range(num_batches), desc=f"SNR {snr_db} dB"):
                    curr_batch_size = min(batch_size, num_samples - total_samples)
                    if curr_batch_size <= 0:
                        break
                    
                    # Generate batch with fixed SNR
                    batch = data_gen.generate_batch(curr_batch_size, fixed_snr=snr_db)
                    
                    # Process batch
                    with torch.no_grad():
                        batch_nmse = 0
                        for i in range(curr_batch_size):
                            ls_estimate = batch['ls_estimates'][i]
                            noise_power = batch['noise_powers'][i].item()
                            
                            # Get channel estimates
                            channel_estimates = self.srs_estimator(
                                ls_estimate=ls_estimate,
                                cyclic_shifts=self.config.cyclic_shifts,
                                noise_power=noise_power
                            )
                            
                            # Calculate NMSE for each user/port
                            idx = 0
                            for u in range(self.config.num_users):
                                for p in range(self.config.ports_per_user[u]):
                                    if (u, p) in batch['true_channels']:
                                        true_channel = batch['true_channels'][(u, p)][i]
                                        est_channel = channel_estimates[idx]
                                        
                                        # Calculate NMSE
                                        nmse = calculate_nmse(true_channel, est_channel)
                                        batch_nmse += nmse
                                    
                                    idx += 1
                            
                            # Average NMSE over all user/ports in this sample
                            batch_nmse /= idx
                    
                    # Add to total
                    total_nmse += batch_nmse
                    total_samples += curr_batch_size
                
                # Calculate average NMSE
                avg_nmse = total_nmse / total_samples
                
                # Add to results
                results.append({
                    'channel_model': channel_model,
                    'snr_db': snr_db,
                    'nmse_db': avg_nmse
                })
                
                print(f"  Average NMSE at {snr_db} dB: {avg_nmse:.2f} dB")
        
        # Convert to DataFrame
        return pd.DataFrame(results)
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot evaluation results
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Path to save the plot (if None, plot is displayed)
        """
        plt.figure(figsize=(10, 6))
        
        # Get unique channel models
        channel_models = results_df['channel_model'].unique()
        
        # Plot each channel model
        for channel_model in channel_models:
            model_df = results_df[results_df['channel_model'] == channel_model]
            plt.plot(
                model_df['snr_db'], 
                model_df['nmse_db'], 
                'o-', 
                label=f"{channel_model}"
            )
        
        # Add labels and legend
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title('SRS Channel Estimation Performance')
        plt.legend()
        plt.grid(True)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate SRS Channel Estimator Performance")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--channel_models', type=str, nargs='+', 
                        default=["simple", "TDL-A", "TDL-C", "TDL-E"], 
                        help='Channel models to evaluate')
    parser.add_argument('--snr_values', type=float, nargs='+', 
                        default=[0, 5, 10, 15, 20, 25, 30], 
                        help='SNR values to evaluate')
    parser.add_argument('--num_samples', type=int, default=100, 
                        help='Number of samples for each evaluation condition')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Batch size for evaluation')
    parser.add_argument('--delay_spread', type=float, default=100e-9, 
                        help='Delay spread in seconds')
    parser.add_argument('--sampling_rate', type=float, default=15.36e6, 
                        help='Sampling rate in Hz')
    parser.add_argument('--no_mmse', action='store_true', help='Disable trainable MMSE')
    parser.add_argument('--output_dir', type=str, default='./results', 
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create configuration
    config = create_example_config()
    
    # Create evaluator
    evaluator = SRSEvaluator(
        config=config,
        checkpoint_path=args.checkpoint,
        use_trainable_mmse=not args.no_mmse
    )
    
    # Evaluate model
    results = evaluator.evaluate_model(
        channel_models=args.channel_models,
        snr_values=args.snr_values,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        delay_spread=args.delay_spread,
        sampling_rate=args.sampling_rate,
        seed=args.seed
    )
    
    # Save results to CSV
    results_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'evaluation_plot.png')
    evaluator.plot_results(results, save_path=plot_path)


if __name__ == '__main__':
    main()
