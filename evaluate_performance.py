import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Optional, Tuple, Literal
import pandas as pd
from tqdm import tqdm

from user_config import SRSConfig, create_example_config
from data_generator_refactored import SRSDataGenerator
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
        device: str = "cpu",  # Force CPU-only execution
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
        
        # SRSChannelEstimator doesn't have trainable parameters, only load MMSE module
        # self.srs_estimator.load_state_dict(checkpoint['model_state_dict'])  # Not needed
        
        # Load MMSE module state if available
        if self.mmse_module and 'mmse_state_dict' in checkpoint:
            self.mmse_module.load_state_dict(checkpoint['mmse_state_dict'])
            print("✅ Loaded trainable MMSE module parameters")
        else:
            print("⚠️  No trainable MMSE parameters to load")
    
    def evaluate_model(
        self,
        snr_range: Tuple[float, float] = (-10, 30),
        snr_step: float = 5.0,
        num_samples: int = 100,
        batch_size: int = 10,
        channel_model: str = "TDL-A",
        delay_spread: float = 300e-9,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        Evaluate model performance across SNR range
        
        Args:
            snr_range: SNR range (start_db, end_db)
            snr_step: SNR step size in dB
            num_samples: Number of samples to evaluate for each SNR
            batch_size: Batch size for evaluation
            channel_model: Channel model to use
            delay_spread: Delay spread in seconds
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with evaluation results
        """
        # Generate SNR values
        snr_start, snr_end = snr_range
        snr_values = list(np.arange(snr_start, snr_end + snr_step, snr_step))
        
        print(f"Evaluating SNR range: {snr_start} to {snr_end} dB (step: {snr_step} dB)")
        print(f"SNR points: {snr_values}")
        print(f"Channel model: {channel_model}")
        print(f"Samples per SNR: {num_samples}")
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create system config
        from system_config import create_default_system_config
        system_config = create_default_system_config()
        
        # Create channel model
        from professional_channels import SIONNAChannelModel
        try:
            sionna_channel = SIONNAChannelModel(
                system_config=system_config,
                model_type=channel_model,
                num_rx_antennas=system_config.num_rx_antennas,
                delay_spread=delay_spread,
                device=self.device
            )
            print(f"✅ Created SIONNA channel model: {channel_model}")
        except Exception as e:
            print(f"❌ Failed to create SIONNA channel: {e}")
            sionna_channel = None
        
        # Create data generator
        data_gen = SRSDataGenerator(
            config=self.config,
            channel_model=sionna_channel,
            num_rx_antennas=system_config.num_rx_antennas,
            sampling_rate=system_config.sampling_rate,
            device=self.device
        )
        
        results = []
        
        # Loop through SNR values
        for snr_db in snr_values:
            print(f"\n🔍 Evaluating SNR: {snr_db} dB")
            
            total_nmse = 0
            total_samples = 0
            
            # Calculate number of batches
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            # Evaluate in batches
            for batch_idx in tqdm(range(num_batches), desc=f"SNR {snr_db} dB"):
                curr_batch_size = min(batch_size, num_samples - total_samples)
                if curr_batch_size <= 0:
                    break
                
                # Temporarily modify config for this SNR
                original_snr_range = self.config.snr_range
                self.config.snr_range = (snr_db, snr_db)  # Fixed SNR
                
                try:
                    # Generate batch
                    batch = data_gen.generate_batch(curr_batch_size)
                    
                    # Process batch
                    with torch.no_grad():
                        # Get data
                        ls_estimates_list = batch['ls_estimates']  # List of dicts
                        true_channels_list = batch['true_channels']  # List of dicts
                        
                        # Forward pass
                        estimated_channels_list = self.srs_estimator(
                            ls_estimates=ls_estimates_list,
                            user_config=self.config
                        )
                        
                        # Calculate NMSE for each sample
                        batch_nmse = 0
                        sample_count = 0
                        
                        for i in range(curr_batch_size):
                            est_dict = estimated_channels_list[i]
                            true_dict = true_channels_list[i]
                            
                            # Calculate NMSE for each user/port in this sample
                            sample_nmse = 0
                            port_count = 0
                            
                            for (user_id, port_id) in est_dict.keys():
                                if (user_id, port_id) in true_dict:
                                    est_channel = est_dict[(user_id, port_id)]    # [num_rx_ant, seq_length]
                                    true_channel = true_dict[(user_id, port_id)]  # [num_rx_ant, seq_length]
                                    
                                    # Calculate NMSE for each antenna
                                    for ant_idx in range(est_channel.shape[0]):
                                        nmse = calculate_nmse(
                                            true_channel[ant_idx, :], 
                                            est_channel[ant_idx, :]
                                        )
                                        sample_nmse += nmse
                                        port_count += 1
                            
                            if port_count > 0:
                                sample_nmse /= port_count
                                batch_nmse += sample_nmse
                                sample_count += 1
                        
                        # Average over samples in batch
                        if sample_count > 0:
                            batch_nmse /= sample_count
                            total_nmse += batch_nmse * curr_batch_size
                            total_samples += curr_batch_size
                
                finally:
                    # Restore original SNR range
                    self.config.snr_range = original_snr_range
            
            # Calculate average NMSE for this SNR
            if total_samples > 0:
                avg_nmse = total_nmse / total_samples
            else:
                avg_nmse = float('inf')
            
            # Add to results
            results.append({
                'snr_db': snr_db,
                'nmse_db': avg_nmse,
                'num_samples': total_samples
            })
            
            print(f"  ✅ Average NMSE at {snr_db} dB: {avg_nmse:.2f} dB ({total_samples} samples)")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        print(f"\n📊 Evaluation completed! Total SNR points: {len(results_df)}")
        return results_df
    
    def plot_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None, channel_model: str = "TDL-A"):
        """
        Plot evaluation results
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Path to save the plot (if None, plot is displayed)
            channel_model: Channel model name for the plot title
        """
        plt.figure(figsize=(10, 6))
        
        # Plot NMSE vs SNR
        plt.plot(
            results_df['snr_db'], 
            results_df['nmse_db'], 
            'o-', 
            linewidth=2,
            markersize=6,
            label=f"SRS Channel Estimator ({channel_model})"
        )
        
        # Add labels and legend
        plt.xlabel('SNR (dB)')
        plt.ylabel('NMSE (dB)')
        plt.title(f'SRS Channel Estimation Performance - {channel_model}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate SRS Channel Estimator Performance")
    parser.add_argument('--checkpoint', type=str, default='best_model_tdlc.pt', help='Path to model checkpoint')
    parser.add_argument('--channel_model', type=str, default="TDL-A", 
                        help='Channel model to evaluate (e.g., TDL-A, TDL-C, TDL-E)')
    parser.add_argument('--snr_start', type=float, default=-10.0, 
                        help='Starting SNR value in dB')
    parser.add_argument('--snr_end', type=float, default=30.0, 
                        help='Ending SNR value in dB')
    parser.add_argument('--snr_step', type=float, default=5.0, 
                        help='SNR step size in dB')
    parser.add_argument('--num_samples', type=int, default=10, 
                        help='Number of samples for each SNR point')
    parser.add_argument('--batch_size', type=int, default=10, 
                        help='Batch size for evaluation')
    parser.add_argument('--delay_spread', type=float, default=30e-9, 
                        help='Delay spread in seconds')
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
        snr_range=(args.snr_start, args.snr_end),
        snr_step=args.snr_step,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        channel_model=args.channel_model,
        delay_spread=args.delay_spread,
        seed=args.seed
    )
    
    # Save results to CSV
    results_path = os.path.join(args.output_dir, f'evaluation_results_{args.channel_model}.csv')
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Plot results
    plot_path = os.path.join(args.output_dir, f'evaluation_plot_{args.channel_model}.png')
    evaluator.plot_results(results, save_path=plot_path, channel_model=args.channel_model)


if __name__ == '__main__':
    main()
