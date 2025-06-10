from model import SRSChannelEstimator
import torch

# Create an instance of the estimator
estimator = SRSChannelEstimator(seq_length=64)  # smaller size for testing

# Check if the attribute exists
print(f"Has current_h_with_residual_phasor attribute: {hasattr(estimator, 'current_h_with_residual_phasor')}")

# Create a dummy input
dummy_input = torch.randn(64, dtype=torch.complex64)
cyclic_shifts = [[0, 1], [0, -1]]  # Two users, first with 2 ports, second with 1 port
noise_power = 0.01

# Process through forward pass
try:
    output = estimator(dummy_input, cyclic_shifts, noise_power)
    print(f"Forward pass successful, output length: {len(output)}")
    print(f"current_h_with_residual_phasor set: {estimator.current_h_with_residual_phasor is not None}")
except Exception as e:
    print(f"Error during forward pass: {e}")
