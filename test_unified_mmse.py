import torch
import torch.nn as nn
import numpy as np
from model import SRSChannelEstimator, TrainableMMSEModule

# Create test inputs
seq_length = 64
batch_size = 1

# Create estimator
estimator = SRSChannelEstimator(seq_length=seq_length)
print(f"Created estimator with seq_length={seq_length}")

# Create MMSE module
mmse_module = TrainableMMSEModule(seq_length=seq_length, use_complex_input=True)
print(f"Created MMSE module")

# Create dummy inputs
ls_estimate = torch.randn(seq_length, dtype=torch.complex64)
cyclic_shifts = [[0], [2]]  # Two users with one port each
noise_power = 0.01

print(f"Running first pass to generate h_with_residual/phasor")
# First run to get h_with_residual/phasor
estimator(ls_estimate, cyclic_shifts, noise_power)

# Check if attribute exists and has value
if hasattr(estimator, 'current_h_with_residual_phasor'):
    print(f"current_h_with_residual_phasor attribute exists")
    if estimator.current_h_with_residual_phasor is not None:
        print(f"current_h_with_residual_phasor has value with shape: {estimator.current_h_with_residual_phasor.shape}")
        
        # Try to generate MMSE matrices
        try:
            C, R, _ = mmse_module(estimator.current_h_with_residual_phasor)
            print(f"Generated MMSE matrices - C shape: {C.shape}, R shape: {R.shape}")
            
            # Set matrices and run again
            estimator.set_mmse_matrices(C=C, R=R)
            output = estimator(ls_estimate, cyclic_shifts, noise_power)
            print(f"Second forward pass successful")
        except Exception as e:
            print(f"Error using current_h_with_residual_phasor: {e}")
    else:
        print(f"current_h_with_residual_phasor is None")
else:
    print(f"current_h_with_residual_phasor attribute does not exist")
