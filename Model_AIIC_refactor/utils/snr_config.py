"""
SNR configuration parser and sampler.

Supports two types of SNR configuration:
1. Range: Continuous sampling from [min, max]
2. Discrete: Random selection from specific values
"""

import numpy as np
from typing import Union, Tuple


class SNRConfig:
    """
    SNR Configuration handler
    
    Supports:
    1. Range-based: snr_config = {type: 'range', min: 0, max: 30}
    2. Discrete: snr_config = {type: 'discrete', values: [0, 10, 20, 30]}
    """
    
    def __init__(self, config: dict):
        """
        Initialize SNR configuration
        
        Args:
            config: SNR configuration dictionary
                   - type: 'range' or 'discrete'
                   - For range: min, max, sampling (optional), num_bins (optional)
                   - For discrete: values
                   - per_sample: bool (optional, default False)
        
        Examples:
            >>> # Range-based
            >>> config = {'type': 'range', 'min': 0, 'max': 30}
            >>> snr = SNRConfig(config)
            
            >>> # Discrete
            >>> config = {'type': 'discrete', 'values': [0, 10, 20, 30]}
            >>> snr = SNRConfig(config)
        """
        self.config_type = config.get('type', 'range')
        self.per_sample = config.get('per_sample', False)
        
        if self.config_type == 'range':
            self.min_snr = config['min']
            self.max_snr = config['max']
            self.sampling = config.get('sampling', 'uniform')
            self.num_bins = config.get('num_bins', 10)
            
            # Initialize SNRSampler if stratified
            if self.sampling == 'stratified':
                from .snr_sampler import SNRSampler
                self.sampler = SNRSampler(
                    snr_min=self.min_snr,
                    snr_max=self.max_snr,
                    strategy='stratified',
                    num_bins=self.num_bins
                )
            else:
                self.sampler = None
        
        elif self.config_type == 'discrete':
            self.snr_values = config['values']
        
        else:
            raise ValueError(
                f"Unknown SNR config type: '{self.config_type}'. "
                f"Valid types: 'range', 'discrete'"
            )
    
    def sample(self, batch_size: int = 1) -> Union[float, Tuple[float, float]]:
        """
        Sample SNR value(s)
        
        Args:
            batch_size: Batch size (for per_sample mode)
        
        Returns:
            - If per_sample=False: single SNR value (float)
            - If per_sample=True: (min_snr, max_snr) tuple for compatibility
        
        Note: per_sample SNR is handled in data_generator, this returns the range
        """
        if self.config_type == 'range':
            if self.per_sample:
                # Return range tuple for per-sample sampling in data_generator
                return (self.min_snr, self.max_snr)
            else:
                # Sample single SNR for entire batch
                if self.sampler:
                    return self.sampler.sample()
                else:
                    return np.random.uniform(self.min_snr, self.max_snr)
        
        elif self.config_type == 'discrete':
            if self.per_sample:
                # Cannot do per-sample with discrete values in current design
                # Fallback: return a range approximation
                return (min(self.snr_values), max(self.snr_values))
            else:
                # Random selection from discrete values
                return float(np.random.choice(self.snr_values))
    
    def get_snr_for_data_generator(self) -> Union[float, Tuple[float, float]]:
        """
        Get SNR in format compatible with data_generator
        
        Returns:
            - Single float: use this SNR for entire batch
            - Tuple (min, max): sample randomly for each sample (if per_sample=True)
        """
        return self.sample()
    
    def __repr__(self):
        if self.config_type == 'range':
            return f"SNRConfig(range: [{self.min_snr}, {self.max_snr}], sampling={self.sampling}, per_sample={self.per_sample})"
        else:
            return f"SNRConfig(discrete: {self.snr_values}, per_sample={self.per_sample})"


def parse_snr_config(config: dict) -> SNRConfig:
    """
    Parse SNR configuration dictionary
    
    Args:
        config: SNR configuration dictionary or legacy format
    
    Returns:
        SNRConfig object
    
    Examples:
        >>> # New format
        >>> config = {'type': 'range', 'min': 0, 'max': 30}
        >>> snr = parse_snr_config(config)
        
        >>> # Legacy format (for backward compatibility)
        >>> config = {'snr_range': [0, 30]}
        >>> snr = parse_snr_config(config)
    """
    # Check if it's already in new format
    if 'type' in config:
        return SNRConfig(config)
    
    # Legacy format conversion
    if 'snr_range' in config:
        snr_range = config['snr_range']
        return SNRConfig({
            'type': 'range',
            'min': snr_range[0],
            'max': snr_range[1],
            'per_sample': config.get('snr_per_sample', False),
            'sampling': config.get('snr_sampling', 'uniform'),
            'num_bins': config.get('snr_num_bins', 10)
        })
    
    # Default
    return SNRConfig({'type': 'range', 'min': 0, 'max': 30})


__all__ = ['SNRConfig', 'parse_snr_config']
