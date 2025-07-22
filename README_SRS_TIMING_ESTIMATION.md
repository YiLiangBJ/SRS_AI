# SRS Timing and Delay Spread Estimation Integration Guide

This document provides guidance on integrating the SRS timing and delay spread estimation functionality into the existing SRS Channel Estimation training pipeline.

## Overview

The SRS timing and delay spread estimation module provides functionality to:

1. **Timing Estimation**: Estimate timing offsets for SRS signals to correct for synchronization errors
2. **Delay Spread Estimation**: Estimate the delay spread of the channel to optimize MMSE estimation
3. **Phase Correction**: Apply timing uncertainty correction to channel estimates
4. **Performance Analysis**: Collect statistics on timing estimation performance

## Files Created

1. **`srs_timing_delay_estimation.py`**: Core implementation of the timing and delay spread estimation algorithm ported from MATLAB
2. **`srs_timing_integration_example.py`**: Integration wrapper and example demonstrating usage with the existing pipeline

## Core Function

### `estimate_timing_and_delay_spread()`

The main estimation function takes the following key inputs:

```python
def estimate_timing_and_delay_spread(
    simu_params: Dict[str, Any],           # Simulation parameters
    ue_srs_transmit: List[int],            # UE indices transmitting SRS
    ue_group_port: Dict,                   # UE group to port mapping
    n_rx: int,                             # Number of receive antennas
    time_in: np.ndarray,                   # Time domain signal [samples, antennas, slots]
    num_idft: int,                         # Number of IDFT points
    is_fft: bool,                          # FFT processing flag
    n_cs_loop: np.ndarray,                 # Cyclic shift values
    ue_idx_loop: np.ndarray,               # UE indices in processing loop
    port_idx_loop: np.ndarray,             # Port indices in processing loop
    i_group: int,                          # Current group index
    ts: float,                             # Sampling period
    n_cs_max_srs: int,                     # Maximum cyclic shift
    est_offset_all: List[float]            # Accumulated offset estimates
) -> Tuple[...]
```

**Returns:**
- `delay_in_samples_srs`: Timing delays per UE and port
- `delay_spread_in_samples_srs`: Delay spreads per UE and port
- `n_leak_cs`: Cyclic shift leakage values
- `phasor_timing_uncertainty_esti`: Timing uncertainty phasors for correction
- `n_point_retain_info`: Retained point information for window sizing
- `est_offset_all`: Updated list of all estimated offsets
- `interval`: Timing intervals for each UE and port
- `pdp_info`: Power Delay Profile information
- `energy_sig`: Signal energy per UE and port

## Integration Wrapper Class

### `SRSTimingDelayEstimator`

A wrapper class that simplifies integration with the existing pipeline:

```python
from srs_timing_integration_example import SRSTimingDelayEstimator

# Initialize with configuration
config = {
    'sampling_period': 1e-6,     # Sampling period in seconds
    'n_cs_max_srs': 8,          # Maximum cyclic shift value
    'num_idft': 512,            # Number of IDFT points
    'is_fft': True              # FFT processing flag
}

estimator = SRSTimingDelayEstimator(config)
```

## Integration Steps

### 1. In the Training Loop

Add timing estimation before channel estimation:

```python
# In your training loop (e.g., in trainMLPmmse.py)
def train_step_with_timing_estimation(model, batch_data, estimator):
    # Extract time domain signal and UE configuration from batch
    time_domain_signal = batch_data['time_domain_signal']  # [samples, antennas, slots]
    simu_params = batch_data['simu_params']
    ue_config = batch_data['ue_config']
    
    # Perform timing and delay spread estimation
    timing_results = estimator.estimate_for_batch(
        simu_params, ue_config, time_domain_signal
    )
    
    # Apply timing correction to channel estimates
    channel_estimates = model.get_channel_estimates(batch_data)
    
    for ue_id in ue_config['transmitting_ues']:
        channel_estimates[ue_id] = estimator.apply_timing_correction(
            channel_estimates[ue_id], timing_results, ue_id
        )
    
    # Continue with standard training using corrected estimates
    loss = model.compute_loss(channel_estimates, batch_data['true_channels'])
    return loss, timing_results
```

### 2. In the Data Generator

Modify data generation to include time domain signals:

```python
# In data_generator_refactored.py
def generate_batch_with_timing_info(self, batch_size):
    # ... existing data generation ...
    
    # Generate time domain received signal for timing estimation
    time_domain_signal = self.generate_time_domain_srs_signal(
        channel_matrix, srs_sequences, noise_power
    )
    
    batch_data = {
        'frequency_domain_signal': freq_signal,
        'time_domain_signal': time_domain_signal,  # Add this
        'channel_matrix': channel_matrix,
        'simu_params': self.simu_params,
        'ue_config': ue_config,
        # ... other batch data ...
    }
    
    return batch_data
```

### 3. Configuration Integration

Add timing estimation parameters to `user_config.py`:

```python
# In user_config.py
TIMING_ESTIMATION_CONFIG = {
    'enable_timing_estimation': True,
    'sampling_period': 1e-6,
    'n_cs_max_srs': 8,
    'num_idft': 512,
    'is_fft': True,
    'nr_used_for_timing_delay_est': 4,  # Number of RX antennas for estimation
}
```

### 4. Model Integration

Integrate with the SRS Channel Estimator:

```python
# In model_Traditional.py or model_AIpart.py
class SRSChannelEstimator:
    def __init__(self, config):
        # ... existing initialization ...
        
        if config.TIMING_ESTIMATION_CONFIG['enable_timing_estimation']:
            self.timing_estimator = SRSTimingDelayEstimator(
                config.TIMING_ESTIMATION_CONFIG
            )
        else:
            self.timing_estimator = None
    
    def forward(self, batch_data):
        # Perform timing estimation if enabled
        timing_results = None
        if self.timing_estimator:
            timing_results = self.timing_estimator.estimate_for_batch(
                batch_data['simu_params'],
                batch_data['ue_config'],
                batch_data['time_domain_signal']
            )
        
        # Standard channel estimation
        channel_estimates = self.estimate_channels(batch_data)
        
        # Apply timing correction if available
        if timing_results:
            for ue_id in batch_data['ue_config']['transmitting_ues']:
                channel_estimates[ue_id] = self.timing_estimator.apply_timing_correction(
                    channel_estimates[ue_id], timing_results, ue_id
                )
        
        return channel_estimates, timing_results
```

## Performance Considerations

### Memory Usage
- The time domain signal requires additional memory: `[samples, antennas, slots]`
- Consider processing in smaller chunks for very large batch sizes

### Computational Complexity
- Timing estimation adds O(N log N) complexity for FFT operations
- Delay spread estimation involves statistical analysis of power delay profiles

### Distributed Training
- Timing estimation is performed independently per process/GPU
- Results can be aggregated across processes for global statistics

## Validation and Testing

### Unit Testing
```python
# Test timing estimation on known synthetic data
python srs_timing_delay_estimation.py
```

### Integration Testing
```python
# Test integration with pipeline
python srs_timing_integration_example.py
```

### Performance Monitoring
```python
# Get estimation statistics during training
stats = estimator.get_estimation_statistics()
print(f"Mean timing offset: {stats['mean_offset']:.3f} samples")
print(f"Timing offset std: {stats['std_offset']:.3f} samples")
```

## Usage Examples

### Basic Usage in Training Script

```python
# In train_distributed.py
from srs_timing_integration_example import SRSTimingDelayEstimator

def main():
    # ... existing setup ...
    
    # Initialize timing estimator
    timing_estimator = SRSTimingDelayEstimator(config.TIMING_ESTIMATION_CONFIG)
    
    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(data_loader):
            # Estimate timing and delay spread
            timing_results = timing_estimator.estimate_for_batch(
                batch_data['simu_params'],
                batch_data['ue_config'],
                batch_data['time_domain_signal']
            )
            
            # Train model with timing-corrected data
            loss = train_step_with_timing_correction(
                model, batch_data, timing_results
            )
            
            # Log timing statistics periodically
            if batch_idx % 100 == 0:
                stats = timing_estimator.get_estimation_statistics()
                logger.info(f"Timing stats: {stats}")
```

### Advanced Usage with Adaptive Parameters

```python
# Adaptive timing estimation based on channel conditions
def adaptive_timing_estimation(estimator, batch_data, snr_threshold=10):
    # Estimate SNR from received signal
    estimated_snr = estimate_snr(batch_data['time_domain_signal'])
    
    if estimated_snr > snr_threshold:
        # High SNR: use full precision timing estimation
        return estimator.estimate_for_batch(
            batch_data['simu_params'],
            batch_data['ue_config'],
            batch_data['time_domain_signal']
        )
    else:
        # Low SNR: use simplified estimation or skip
        return None
```

## Error Handling

The implementation includes proper error handling for:
- Empty search windows in timing estimation
- Invalid array dimensions in cyclic shift calculations
- Missing UE configuration data
- Boundary conditions in delay spread estimation

## Future Enhancements

1. **GPU Acceleration**: Port compute-intensive parts to CUDA for faster processing
2. **Adaptive Algorithms**: Implement adaptive timing tracking for time-varying channels
3. **Multi-User Interference**: Extend to handle inter-user interference in timing estimation
4. **Real-Time Processing**: Optimize for real-time systems with streaming data

## Dependencies

The implementation requires:
- `numpy` for numerical computations
- `scipy.stats` for statistical distributions
- `torch` for tensor operations and integration with PyTorch models

## Notes

- The current implementation assumes CPU processing; GPU acceleration can be added
- Timing correction is applied in the frequency domain using phase rotation
- The algorithm is designed to work with both synthetic and real SRS data
- Performance may vary based on channel conditions and system parameters
