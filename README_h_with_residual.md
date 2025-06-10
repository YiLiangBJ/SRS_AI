# Using h_with_residual/phasor as Input for MMSE Matrices

This document explains the modified SRS channel estimation flow implemented in `train_debug_h_with_residual.py`, which uses `h_with_residual/phasor` as input for MMSE matrix generation instead of the original `ls_estimate`.

## Key Changes

1. **Using `h_with_residual/phasor` as Input**: 
   - The MMSE matrices are now generated using `h_with_residual/phasor` as input instead of `ls_estimate`
   - This provides a cleaner, more processed signal as input to the MMSE matrix generation network

2. **Two-Pass Approach**:
   - First run: The SRS estimator processes the `ls_estimate` to generate `h_with_residual/phasor`
   - MMSE matrix generation: The MMSE matrix generator uses `h_with_residual/phasor` to create C and R matrices
   - Second run: The SRS estimator runs again with the new MMSE matrices to produce the final channel estimate

3. **Unified Network**:
   - All users/ports share the same network for training the MMSE matrices
   - This allows losses from all users/ports to be combined for gradient updates

## Implementation Details

The two-pass approach is implemented as follows in the forward pass:

```python
# First run SRS estimator to generate h_with_residual/phasor
channel_estimates_initial = self.srs_estimator(
    ls_estimate=ls_estimate,
    cyclic_shifts=self.config.cyclic_shifts,
    noise_power=noise_power
)

# Check if h_with_residual/phasor is available
if self.srs_estimator.current_h_with_residual_phasor is not None:
    # Use h_with_residual/phasor as input to generate MMSE matrices
    C, R = self.mmse_module(self.srs_estimator.current_h_with_residual_phasor)
    
    # Set MMSE matrices
    self.srs_estimator.set_mmse_matrices(C=C, R=R)
    
    # Run estimator again with new matrices
    channel_estimates = self.srs_estimator(
        ls_estimate=ls_estimate,
        cyclic_shifts=self.config.cyclic_shifts,
        noise_power=noise_power
    )
```

## Training Flow

1. Generate a batch of training data
2. For each sample in the batch:
   - Run the SRS estimator once to get `h_with_residual/phasor`
   - Generate MMSE matrices using `h_with_residual/phasor` as input
   - Run the SRS estimator again with the new MMSE matrices
   - Calculate loss against true channel
3. Backpropagate the combined loss through the entire network
4. Update network parameters

## Benefits

1. **Better Input Signal**: Using `h_with_residual/phasor` provides a cleaner, more processed signal for MMSE matrix generation.
2. **Shared Network**: Using a shared network for all users/ports enables efficient training with combined losses.
3. **Improved Performance**: This approach should lead to better MMSE matrix estimates and, ultimately, improved channel estimation.

## Usage

To train the model using this approach:

```
python train_debug_h_with_residual.py --epochs 100 --train_batches 100 --val_batches 20 --batch_size 32
```

Optional arguments:
- `--val_every`: Validate every n epochs (default: 1)
- `--save_every`: Save checkpoint every n epochs (default: 5) 
- `--no_mmse`: Disable trainable MMSE
- `--enable_plotting`: Enable plotting
- `--save_dir`: Save directory (default: './checkpoints_modified')
- `--load_checkpoint`: Load checkpoint file
