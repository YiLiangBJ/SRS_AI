"""
SNR Sampling Strategies for Training

Provides different strategies for sampling SNR values during training
to ensure good coverage across the entire SNR range.
"""

import numpy as np
from collections import deque


class SNRSampler:
    """
    Smart SNR sampler that ensures good coverage across SNR range
    
    Strategies:
    1. 'uniform': Simple uniform random (default, can cluster)
    2. 'stratified': Divide range into bins, sample uniformly from each bin
    3. 'round_robin': Cycle through SNR bins in order
    4. 'adaptive': Sample more from poorly-performing SNR regions
    """
    
    def __init__(self, snr_min, snr_max, strategy='stratified', num_bins=10):
        """
        Args:
            snr_min: Minimum SNR (dB)
            snr_max: Maximum SNR (dB)
            strategy: Sampling strategy ('uniform', 'stratified', 'round_robin', 'adaptive')
            num_bins: Number of SNR bins (for stratified/round_robin/adaptive)
        """
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.strategy = strategy
        self.num_bins = num_bins
        
        # For stratified/round_robin
        self.bin_edges = np.linspace(snr_min, snr_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
        # For round_robin
        self.current_bin = 0
        
        # For adaptive
        self.bin_losses = [float('inf')] * num_bins  # Track loss per bin
        self.bin_counts = [0] * num_bins  # Track samples per bin
        
        # Recent SNRs (to avoid clustering)
        self.recent_snrs = deque(maxlen=20)  # Remember last 20 SNRs
        
    def sample(self):
        """Sample one SNR value"""
        if self.strategy == 'uniform':
            return self._sample_uniform()
        elif self.strategy == 'stratified':
            return self._sample_stratified()
        elif self.strategy == 'round_robin':
            return self._sample_round_robin()
        elif self.strategy == 'adaptive':
            return self._sample_adaptive()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _sample_uniform(self):
        """Simple uniform random sampling"""
        snr = np.random.uniform(self.snr_min, self.snr_max)
        self.recent_snrs.append(snr)
        return snr
    
    def _sample_stratified(self):
        """
        Stratified sampling: Divide SNR range into bins, sample uniformly
        
        Ensures each bin is sampled equally often
        Good for balanced training across entire SNR range
        """
        # Choose a random bin
        bin_idx = np.random.randint(0, self.num_bins)
        
        # Sample uniformly within that bin
        bin_min = self.bin_edges[bin_idx]
        bin_max = self.bin_edges[bin_idx + 1]
        snr = np.random.uniform(bin_min, bin_max)
        
        self.recent_snrs.append(snr)
        self.bin_counts[bin_idx] += 1
        
        return snr
    
    def _sample_round_robin(self):
        """
        Round-robin sampling: Cycle through bins in order
        
        Most predictable, ensures no clustering
        Good for systematic exploration
        """
        # Use current bin
        bin_min = self.bin_edges[self.current_bin]
        bin_max = self.bin_edges[self.current_bin + 1]
        
        # Add some randomness within bin
        snr = np.random.uniform(bin_min, bin_max)
        
        self.recent_snrs.append(snr)
        self.bin_counts[self.current_bin] += 1
        
        # Move to next bin
        self.current_bin = (self.current_bin + 1) % self.num_bins
        
        return snr
    
    def _sample_adaptive(self):
        """
        Adaptive sampling: Sample more from poorly-performing regions
        
        Prioritizes SNR bins with higher loss
        Good for focusing on difficult regions
        """
        # Calculate sampling probabilities based on losses
        # Higher loss = higher probability
        if all(loss == float('inf') for loss in self.bin_losses):
            # No loss info yet, use uniform
            probabilities = np.ones(self.num_bins) / self.num_bins
        else:
            # Normalize losses to probabilities
            valid_losses = [loss if loss != float('inf') else max([l for l in self.bin_losses if l != float('inf')], default=1.0)
                           for loss in self.bin_losses]
            probabilities = np.array(valid_losses)
            probabilities = probabilities / probabilities.sum()
        
        # Sample bin based on probabilities
        bin_idx = np.random.choice(self.num_bins, p=probabilities)
        
        # Sample within bin
        bin_min = self.bin_edges[bin_idx]
        bin_max = self.bin_edges[bin_idx + 1]
        snr = np.random.uniform(bin_min, bin_max)
        
        self.recent_snrs.append(snr)
        self.bin_counts[bin_idx] += 1
        
        return snr
    
    def update_loss(self, snr, loss):
        """
        Update loss for adaptive sampling
        
        Args:
            snr: SNR value (dB)
            loss: Loss value for this SNR
        """
        # Find which bin this SNR belongs to
        bin_idx = np.searchsorted(self.bin_edges[1:], snr)
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        # Update exponential moving average of loss
        alpha = 0.1  # Smoothing factor
        if self.bin_losses[bin_idx] == float('inf'):
            self.bin_losses[bin_idx] = loss
        else:
            self.bin_losses[bin_idx] = alpha * loss + (1 - alpha) * self.bin_losses[bin_idx]
    
    def get_stats(self):
        """Get sampling statistics"""
        return {
            'strategy': self.strategy,
            'snr_range': (self.snr_min, self.snr_max),
            'num_bins': self.num_bins,
            'bin_centers': self.bin_centers.tolist(),
            'bin_counts': self.bin_counts,
            'bin_losses': [loss if loss != float('inf') else None for loss in self.bin_losses],
            'recent_snrs': list(self.recent_snrs)
        }
    
    def print_stats(self):
        """Print sampling statistics"""
        print("\n" + "="*80)
        print(f"SNR Sampler Statistics ({self.strategy})")
        print("="*80)
        print(f"SNR Range: {self.snr_min:.1f} - {self.snr_max:.1f} dB")
        print(f"Number of bins: {self.num_bins}")
        
        print(f"\nBin Statistics:")
        print(f"{'Bin':<6} {'SNR Range':<20} {'Samples':<12} {'Avg Loss':<12}")
        print("-"*80)
        
        total_samples = sum(self.bin_counts)
        for i in range(self.num_bins):
            bin_range = f"{self.bin_edges[i]:.1f} - {self.bin_edges[i+1]:.1f} dB"
            count = self.bin_counts[i]
            pct = 100 * count / total_samples if total_samples > 0 else 0
            loss_str = f"{self.bin_losses[i]:.6f}" if self.bin_losses[i] != float('inf') else "N/A"
            
            print(f"{i:<6} {bin_range:<20} {count:>6} ({pct:>5.1f}%)  {loss_str:<12}")
        
        print(f"\nTotal samples: {total_samples}")
        print("="*80)


# Convenience functions
def create_snr_sampler(snr_db, strategy='stratified', num_bins=10):
    """
    Create SNR sampler from snr_db argument
    
    Args:
        snr_db: SNR specification (scalar, list, or tuple)
        strategy: Sampling strategy
        num_bins: Number of SNR bins
    
    Returns:
        SNRSampler or None (if fixed SNR)
    """
    if isinstance(snr_db, tuple) and len(snr_db) == 2:
        snr_min, snr_max = snr_db
        return SNRSampler(snr_min, snr_max, strategy=strategy, num_bins=num_bins)
    else:
        # Fixed SNR, no sampler needed
        return None


if __name__ == "__main__":
    """Test SNR samplers"""
    
    print("="*80)
    print("Testing SNR Sampling Strategies")
    print("="*80)
    
    snr_range = (0, 30)
    num_samples = 1000
    
    strategies = ['uniform', 'stratified', 'round_robin']
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"Strategy: {strategy}")
        print(f"{'='*80}")
        
        sampler = SNRSampler(snr_range[0], snr_range[1], strategy=strategy, num_bins=10)
        
        # Sample many times
        snrs = [sampler.sample() for _ in range(num_samples)]
        
        # Print statistics
        sampler.print_stats()
        
        # Check for clustering in recent samples
        recent = list(sampler.recent_snrs)
        if len(recent) >= 5:
            recent_std = np.std(recent[-10:])
            print(f"\nRecent SNR std dev (last 10): {recent_std:.2f} dB")
            if recent_std < 3.0:
                print(f"  ⚠️  Warning: Recent SNRs are clustered!")
            else:
                print(f"  ✓ Good spread in recent SNRs")
