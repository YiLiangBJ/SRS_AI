import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class SRSChannelEstimator(nn.Module):
    """
    SRS Channel Estimator using AI-enhanced methods
    
    This module implements the SRS channel estimation process as described,
    with flexibility to incorporate AI-based components for improved estimation.
    """
    
    def __init__(
        self,
        seq_length: int,
        ktc: int = 4,
        max_users: int = 8,
        max_ports_per_user: int = 4,
        mmse_block_size: int = 12,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the SRS Channel Estimator
        
        Args:
            seq_length: Length of SRS sequence (L)
            ktc: Configuration parameter (ktc=4 -> K=12, ktc=2 -> K=8)
            max_users: Maximum number of users to support
            max_ports_per_user: Maximum number of ports per user to support
            mmse_block_size: Size of blocks for MMSE filtering
            device: Computation device
        """
        super().__init__()
        self.seq_length = seq_length
        self.ktc = ktc
        self.K = 12 if ktc == 4 else 8  # Number of cyclic shifts
        self.max_users = max_users
        self.max_ports_per_user = max_ports_per_user
        self.mmse_block_size = mmse_block_size
        self.device = device
        
        # Initialize parameters for MMSE filter matrices
        # These could be trainable or set by traditional methods
        self.C_matrix = None
        self.R_matrix = None

    def forward(
        self, 
        ls_estimate: torch.Tensor,
        cyclic_shifts: List[List[int]], 
        noise_power: Optional[float] = None,
        delay_search_range: Tuple[int, int] = (-10, 10)
    ) -> List[torch.Tensor]:
        """
        Forward pass through the channel estimation process
        
        Args:
            ls_estimate: LS channel estimate sequence h (complex tensor of shape [L])
            cyclic_shifts: List of cyclic shifts for each user's ports [[n_0^0, n_0^1...], [n_1^0, n_1^1...]]
            noise_power: Estimated noise power (if None, will be estimated from ls_estimate)
            delay_search_range: Range of possible delay offsets to search (min_offset, max_offset)
            
        Returns:
            List of channel estimates for each user's ports
        """
        # Validate inputs
        assert ls_estimate.shape[0] == self.seq_length, f"Expected sequence length {self.seq_length}, got {ls_estimate.shape[0]}"
        
        # Get number of users and ports
        num_users = len(cyclic_shifts)
        num_ports_per_user = [len(shifts) for shifts in cyclic_shifts]
        
        # Compute Locc based on user/port configuration
        Locc = self._compute_locc(cyclic_shifts)
        
        # Transform LS estimate to time domain
        h_time = self._idft(ls_estimate)
        
        # For each user and port, estimate timing offset and process
        h_processed_list = []
        timing_offsets = {}
        
        for u in range(num_users):
            for p in range(num_ports_per_user[u]):
                # Get cyclic shift for this user and port
                n_u_p = cyclic_shifts[u][p]
                  # Calculate ideal peak location
                ideal_peak = (self.K - n_u_p) % self.K * self.seq_length // self.K
                
                # Estimate timing offset within specified search range
                T_u_p = self._estimate_timing_offset(h_time, ideal_peak, delay_search_range)
                timing_offsets[(u, p)] = T_u_p
                
                # Calculate cyclic shift amount
                m_u_p = (T_u_p + ideal_peak)
                
                # Generate phasor for shifting
                phasor = self._generate_phasor(m_u_p)
                
                # Shift the channel to align peak at position 0
                h_u_p = ls_estimate * phasor
                
                # Apply OCC de-multiplexing
                h_avg = self._apply_occ_demux(h_u_p, Locc)
                
                # Linear interpolation back to full length
                h_interpolated = self._linear_interpolation(h_avg, self.seq_length)
                
                h_processed_list.append((u, p, h_interpolated, phasor))
        
        # Reconstruct combined signal
        h_reconstructed = torch.zeros_like(ls_estimate, dtype=torch.complex64)
        for u, p, h_interp, phasor in h_processed_list:
            # Shift back to original position
            h_reconstructed += h_interp / phasor
        
        # Calculate residual
        residual = ls_estimate - h_reconstructed
        
        # Add residual back to each estimate
        final_estimates = []
        for u in range(num_users):
            for p in range(num_ports_per_user[u]):
                for _, _, h_interp, phasor in h_processed_list:
                    # if item[0] == u and item[1] == p:
                    h_with_residual = h_interp + residual * phasor
                    
                    # Apply MMSE filtering
                    if noise_power is None:
                        noise_power = self._estimate_noise_power(ls_estimate)
                    
                    h_mmse = self._apply_mmse_filter(h_with_residual, noise_power)
                    final_estimates.append(h_mmse)
        
        return final_estimates
    
    def _compute_locc(self, cyclic_shifts: List[List[int]]) -> int:
        """
        Compute Locc based on user configuration
        
        Note: This is a simplified implementation. In a real system, this would
        follow 3GPP specifications based on ktc, number of users, etc.
        """
        # For example with ktc=4, 2 users with 2 ports each, Locc=4
        # This is a placeholder - in a real system this would be calculated based on 3GPP specs
        return 4 if self.ktc == 4 else 2
    
    def _idft(self, freq_domain: torch.Tensor) -> torch.Tensor:
        """Apply IDFT to convert from frequency domain to time domain"""
        return torch.fft.ifft(freq_domain, dim=0)
    
    def _dft(self, time_domain: torch.Tensor) -> torch.Tensor:
        """Apply DFT to convert from time domain to frequency domain"""
        return torch.fft.fft(time_domain, dim=0)
    
    def _estimate_timing_offset(self, h_time: torch.Tensor, ideal_peak: int, search_range: Tuple[int, int] = (-10, 10)) -> int:
        """
        Estimate timing offset based on CIR peak position relative to ideal position,
        searching only within a limited range around the ideal peak
        
        Args:
            h_time: Channel impulse response in time domain
            ideal_peak: Ideal peak position
            search_range: Range around ideal_peak to search for the actual peak (min_offset, max_offset)
            
        Returns:
            Timing offset in samples
        """
        # Get magnitude of h_time
        h_mag = torch.abs(h_time)
        
        # Define the search range around ideal_peak
        min_offset, max_offset = search_range
        L = self.seq_length
        
        # Calculate search start and end positions with circular wrapping
        start_pos = (ideal_peak + min_offset) % L
        end_pos = (ideal_peak + max_offset) % L
        
        # Create mask for the search range
        search_mask = torch.zeros_like(h_mag, dtype=torch.bool)
        
        # Handle wraparound case
        if start_pos <= end_pos:
            # Simple case: start to end in a continuous segment
            search_mask[start_pos:end_pos+1] = True
        else:
            # Wraparound case: need two segments (end of array and beginning)
            search_mask[start_pos:] = True  # From start_pos to the end
            search_mask[:end_pos+1] = True  # From beginning to end_pos
        
        # Find peak position only within the masked region
        # First ensure we don't have all zeros in the mask
        if torch.any(search_mask):
            # Get the magnitude values only in the search range
            search_values = torch.where(search_mask, h_mag, torch.tensor(-float('inf'), device=h_mag.device))
            peak_idx = torch.argmax(search_values).item()
        else:
            # Fallback to full search if mask is empty (shouldn't happen with reasonable search_range)
            peak_idx = torch.argmax(h_mag).item()
        
        # Calculate timing offset relative to ideal peak
        # Handle circular distance calculation
        if peak_idx > ideal_peak:
            if peak_idx - ideal_peak > L/2:  # Wraparound case
                offset = peak_idx - L - ideal_peak
            else:
                offset = peak_idx - ideal_peak
        else:
            if ideal_peak - peak_idx > L/2:  # Wraparound case
                offset = peak_idx + L - ideal_peak
            else:
                offset = peak_idx - ideal_peak
                
        return int(offset)
    
    def _generate_phasor(self, m: int) -> torch.Tensor:
        """
        Generate phasor for cyclic shifting in frequency domain
        
        Args:
            m: Shifting amount
            
        Returns:
            Phasor vector of length L
        """
        n = torch.arange(self.seq_length, device=self.device)
        return torch.exp(1j * 2 * np.pi * m * n / self.seq_length)
    
    def _apply_occ_demux(self, h: torch.Tensor, Locc: int) -> torch.Tensor:
        """
        Apply OCC demultiplexing by averaging every Locc points
        
        Args:
            h: Input sequence
            Locc: OCC length
            
        Returns:
            Averaged sequence of length L/Locc
        """
        L = self.seq_length
        h_reshaped = h.reshape(L // Locc, Locc)
        h_avg = torch.mean(h_reshaped, dim=1)
        return h_avg
    def _linear_interpolation(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Apply linear interpolation to increase sequence length, 
        properly positioning averaged points at the center of their respective groups
        
        Args:
            h_avg: Input sequence (averaged values from groups of samples)
            target_length: Target length after interpolation (original sequence length)
            
        Returns:
            Interpolated sequence of target_length
        """
        # Extract real and imaginary parts
        h_real = torch.real(h_avg)
        h_imag = torch.imag(h_avg)
        
        # Calculate group size (Locc) based on input and target length
        group_size = target_length // len(h_avg)
        
        # Create properly positioned input indices (at the center of each group)
        # For example, if group_size=4, the centers would be at indices 1.5, 5.5, 9.5, etc.
        orig_indices = np.array([np.mean(np.arange(group_size)) + i * group_size for i in range(len(h_avg))])
        
        # Create output indices (all integer positions from 0 to target_length-1)
        new_indices = np.arange(target_length)
        
        # Convert tensors to numpy arrays for interpolation
        h_real_np = h_real.cpu().numpy()
        h_imag_np = h_imag.cpu().numpy()
        
        # Use numpy's interp function with the properly positioned indices
        real_interp_np = np.interp(new_indices, orig_indices, h_real_np)
        imag_interp_np = np.interp(new_indices, orig_indices, h_imag_np)
        
        # Convert back to torch tensors
        real_interp = torch.tensor(real_interp_np, device=self.device, dtype=torch.float32)
        imag_interp = torch.tensor(imag_interp_np, device=self.device, dtype=torch.float32)
        
        # Combine back to complex
        return torch.complex(real_interp, imag_interp)
    
    def _estimate_noise_power(self, h: torch.Tensor) -> float:
        """
        Estimate noise power from adjacent frequency domain samples
        
        Args:
            h: Frequency domain channel estimate
            
        Returns:
            Estimated noise power
        """
        # Calculate differences between adjacent samples
        diff = h[1:] - h[:-1]
        
        # Calculate average power of differences (divided by 2 as explained in theory)
        noise_power = torch.mean(torch.abs(diff)**2) / 2
        return noise_power.item()
    def _apply_mmse_filter(self, h: torch.Tensor, noise_power: float) -> torch.Tensor:
        """
        Apply MMSE filtering in blocks: h_mmse = C * (C + R)^-1 * h
        
        Process the channel in blocks of size mmse_block_size to improve efficiency.
        Each block is filtered separately with a smaller MMSE matrix.
        
        Args:
            h: Input channel estimate
            noise_power: Noise power estimate
            
        Returns:
            MMSE filtered channel estimate
        """
        # Get sequence length and block size
        L = self.seq_length
        block_size = self.mmse_block_size
        
        # Create output tensor
        h_mmse = torch.zeros_like(h)
        
        # Process each block separately
        num_blocks = (L + block_size - 1) // block_size  # Ceiling division
        
        for i in range(num_blocks):
            # Calculate block indices with proper handling of the last block
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, L)
            current_block_size = end_idx - start_idx
            
            if current_block_size < 2:  # Skip processing for very small blocks
                h_mmse[start_idx:end_idx] = h[start_idx:end_idx]
                continue
            
            # Extract block
            h_block = h[start_idx:end_idx]
            
            # Get block-specific MMSE matrices
            C_block = self._get_block_C_matrix(current_block_size)
            R_block = self._get_block_R_matrix(noise_power, current_block_size)
            
            # Apply MMSE filter to block
            h_block_vec = h_block.reshape(-1, 1)  # Convert to column vector
            
            # MMSE formula: C * (C + R)^-1 * h
            mmse_filter = C_block @ torch.inverse(C_block + R_block)
            h_block_mmse = mmse_filter @ h_block_vec
            
            # Store filtered block in output
            h_mmse[start_idx:end_idx] = h_block_mmse.reshape(-1)
        
        return h_mmse
    def _get_block_C_matrix(self, block_size: int) -> torch.Tensor:
        """
        Get channel correlation matrix C for a specific block size
        
        Args:
            block_size: Size of the block to process
            
        Returns:
            C matrix for MMSE filtering of the specified block
        """
        # Construct based on exponential decay model for this block
        C = torch.zeros((block_size, block_size), dtype=torch.complex64, device=self.device)
        
        # Exponential power delay profile
        tau = 0.1  # Time constant
        for i in range(block_size):
            for j in range(block_size):
                delay_diff = abs(i - j)
                # Convert scalar to tensor before using torch.exp
                exponent = torch.tensor(-delay_diff / (tau * block_size), device=self.device)
                C[i, j] = torch.exp(exponent)
        
        return C
    
    def _get_block_R_matrix(self, noise_power: float, block_size: int) -> torch.Tensor:
        """
        Get noise correlation matrix R for a specific block size
        
        Args:
            noise_power: Estimated noise power
            block_size: Size of the block to process
            
        Returns:
            R matrix for MMSE filtering of the specified block
        """
        # Diagonal matrix with noise power for this block size
        R = torch.eye(block_size, device=self.device) * noise_power
        return R
    
    def _get_C_matrix(self) -> torch.Tensor:
        """
        Get channel correlation matrix C for the entire sequence
        (This method is kept for compatibility but blocks should use _get_block_C_matrix)
        
        Returns:
            C matrix for MMSE filtering
        """
        if self.C_matrix is not None:
            return self.C_matrix
        else:
            # Traditional method: construct based on exponential decay model
            L = self.seq_length
            C = torch.zeros((L, L), dtype=torch.complex64, device=self.device)
            
            # Exponential power delay profile
            tau = 0.1  # Time constant
            for i in range(L):
                for j in range(L):
                    delay_diff = abs(i - j)
                    # Convert scalar to tensor before using torch.exp
                    exponent = torch.tensor(-delay_diff / (tau * L), device=self.device)
                    C[i, j] = torch.exp(exponent)
            
            return C
    
    def _get_R_matrix(self, noise_power: float) -> torch.Tensor:
        """
        Get noise correlation matrix R for the entire sequence
        (This method is kept for compatibility but blocks should use _get_block_R_matrix)
        
        Args:
            noise_power: Estimated noise power
            
        Returns:
            R matrix for MMSE filtering
        """
        if self.R_matrix is not None:
            return self.R_matrix
        else:
            # Traditional method: diagonal matrix with noise power
            L = self.seq_length
            R = torch.eye(L, device=self.device) * noise_power
            return R

    def set_mmse_matrices(self, C: torch.Tensor = None, R: torch.Tensor = None):
        """
        Set custom MMSE filter matrices
        
        Args:
            C: Channel correlation matrix
            R: Noise correlation matrix
        """
        if C is not None:
            self.C_matrix = C
        if R is not None:
            self.R_matrix = R


class TrainableMMSEModule(nn.Module):
    """
    Trainable MMSE Filter Module
    
    This module can be used to train the C and R matrices for MMSE filtering
    """
    def __init__(self, seq_length: int, hidden_dim: int = 64):
        """
        Initialize the trainable MMSE module
        
        Args:
            seq_length: Length of sequence (L)
            hidden_dim: Hidden dimension for the neural network
        """
        super().__init__()
        self.seq_length = seq_length
        
        # Networks to generate C and R matrices
        self.C_generator = nn.Sequential(
            nn.Linear(seq_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length * seq_length * 2)  # 2 for real and imaginary parts
        )
        
        self.R_generator = nn.Sequential(
            nn.Linear(1, hidden_dim),  # Input is noise power
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length * seq_length)  # Only real part (noise is real)
        )
    
    def forward(self, channel_stats: torch.Tensor, noise_power: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate C and R matrices
        
        Args:
            channel_stats: Channel statistics (can be frequency domain samples)
            noise_power: Estimated noise power
            
        Returns:
            C: Channel correlation matrix
            R: Noise correlation matrix
        """
        # Generate C matrix
        C_flat = self.C_generator(channel_stats)
        C_real = C_flat[:self.seq_length * self.seq_length].reshape(self.seq_length, self.seq_length)
        C_imag = C_flat[self.seq_length * self.seq_length:].reshape(self.seq_length, self.seq_length)
        C = torch.complex(C_real, C_imag)
        
        # Make C Hermitian (required for covariance matrix)
        C = (C + C.conj().transpose(0, 1)) / 2
          # Generate R matrix
        R = self.R_generator(noise_power.view(1, 1)).reshape(self.seq_length, self.seq_length)
        
        # Make R positive definite
        R = R @ R.transpose(0, 1) + torch.eye(self.seq_length, device=R.device) * 1e-6
        
        return C, R
