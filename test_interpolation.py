import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SRSChannelEstimator

def test_linear_interpolation():
    # Initialize SRS estimator with minimal parameters for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    model = SRSChannelEstimator(
        seq_length=12,  # Use small values for visualization
        ktc=4,
        max_users=1,
        max_ports_per_user=1,
        mmse_block_size=4,
        device=device
    )
    
    # Create a simple test signal (after OCC averaging)
    # Simulating h_avg with 3 points (if Locc=4, original signal would be 12 points)
    h_avg = torch.tensor([1+2j, 3+4j, 5+6j], device=device)
    
    # Use the updated linear interpolation method
    h_interpolated = model._linear_interpolation(h_avg, target_length=12)
    
    # Print original and interpolated values
    print("Original averaged values (h_avg):")
    print(h_avg)
    print("\nInterpolated values (should have centers at positions 1.5, 5.5, 9.5):")
    print(h_interpolated)
    
    # Validate that the interpolated values match the original values at the center positions
    group_size = 12 // len(h_avg)  # Should be 4
    center_positions = [(i + 0.5) * group_size for i in range(len(h_avg))]
    
    print("\nGroup size:", group_size)
    print("Center positions of each group:", center_positions)
    
    # Visualize the results
    plt.figure(figsize=(12, 6))
    
    # Plot real part
    plt.subplot(1, 2, 1)
    plt.title('Real part of interpolated signal')
    x_positions = np.arange(12)
    plt.scatter(x_positions, torch.real(h_interpolated).cpu().numpy(), label='Interpolated points')
    for i, pos in enumerate(center_positions):
        plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'Center {i+1}', (pos, torch.real(h_avg[i]).item()), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.grid(True)
    plt.legend()
    
    # Plot imaginary part
    plt.subplot(1, 2, 2)
    plt.title('Imaginary part of interpolated signal')
    plt.scatter(x_positions, torch.imag(h_interpolated).cpu().numpy(), label='Interpolated points')
    for i, pos in enumerate(center_positions):
        plt.axvline(x=pos, color='r', linestyle='--', alpha=0.5)
        plt.annotate(f'Center {i+1}', (pos, torch.imag(h_avg[i]).item()), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('interpolation_test.png')
    print("\nVisualization saved as 'interpolation_test.png'")

if __name__ == "__main__":
    test_linear_interpolation()
