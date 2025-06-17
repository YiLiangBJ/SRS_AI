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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mmse_module = None
    ):
        """
        Initialize the SRS Channel Estimator
        
        Args:
            seq_length: Length of SRS sequence (L)
            ktc: Configuration parameter (ktc=4 -> K=12, ktc=2 -> K=8)
            max_users: Maximum number of users to support
            max_ports_per_user: Maximum number of ports across all users (different users can have different numbers of ports)
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
        self.device = device        # Initialize parameters for MMSE filter matrices
        self.mmse_module = mmse_module
        # These could be trainable or set by traditional methods
        self.C_matrix = None
        self.R_matrix = None
        
        # Initialize h_with_residual/phasor for use in MMSE matrix generation
        # 存储每个用户/端口的h_with_residual_phasor
        self.current_h_with_residual_phasors = {}
        # 保留单个变量作为向后兼容
        self.current_h_with_residual_phasor = None

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
        
        with torch.no_grad():
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
                    phasor_m = self._generate_phasor(m_u_p)
                    phasor_T = self._generate_phasor(T_u_p)
                    phasor_ideal = self._generate_phasor(ideal_peak)
                    
                    # Shift the channel to align peak at position 0
                    h_u_p = ls_estimate * phasor_m
                    
                    # Apply OCC de-multiplexing
                    h_avg = self._apply_occ_demux(h_u_p, Locc)
                    
                    # Linear interpolation back to full length
                    h_interpolated = self._linear_interpolation(h_avg, self.seq_length)
                    
                    h_processed_list.append((u, p, h_interpolated, phasor_m, phasor_T, phasor_ideal))
        
            # Reconstruct combined signal
            h_reconstructed = torch.zeros_like(ls_estimate, dtype=torch.complex64)
            for u, p, h_interp, phasor_m, _, _ in h_processed_list:
                # Shift back to original position
                h_reconstructed += h_interp * torch.conj(phasor_m)
        
            # Calculate residual
            residual = ls_estimate - h_reconstructed
          # Add residual back to each estimate
        final_estimates = []
        
        # # Create a dictionary to store processed channels for each user-port combination
        # processed_channels = {}
        # for u, p, h_interp, phasor_m, phasor_T, phasor_ideal in h_processed_list:
        #     processed_channels[(u, p)] = (h_interp, phasor_m, phasor_T)
        #   # Process each user-port combination to get final estimates
        # for u in range(num_users):
        #     for p in range(num_ports_per_user[u]):
        #         h_interp, phasor_m, phasor_T = processed_channels[(u, p)]
        for u, p, h_interp, phasor_m, phasor_T, _ in h_processed_list:

            with torch.no_grad():
                # Add residual with appropriate phase correction
                h_with_residual = h_interp + residual * phasor_m
                
                # 保存相位校正后的信道信息，可以用作MMSE矩阵生成的输入
                # 为每个用户/端口单独存储h_with_residual_phasor
                self.current_h_with_residual_phasors[(u, p)] = h_with_residual * torch.conj(phasor_T)
                # 同时更新单个变量以保持向后兼容 - 保存最后一个处理的值
                # self.current_h_with_residual_phasor = h_with_residual / phasor_m
                
            # 如果存在 MMSE 模块，使用它生成 MMSE 矩阵
            if self.mmse_module is not None:
                # 使用该用户/端口的 h_with_residual_phasor 生成 MMSE 矩阵
                C, R = self.mmse_module(h_with_residual)
                # 设置该用户/端口的 MMSE 矩阵
                self.set_mmse_matrices(C=C, R=R, user_port=(u, p))

            # Apply MMSE filtering
            if noise_power is None:
                noise_power = self._estimate_noise_power(ls_estimate)
            
            h_mmse = self._apply_mmse_filter(h_with_residual, noise_power, (u,p))
            # phasor_m = self._generate_phasor(timing_offsets[(u, p)])
            h_mmse_aligned = h_mmse * torch.conj(phasor_T)
            final_estimates.append(h_mmse_aligned)
        
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
        h_mag = torch.abs(h_time)**2
        
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
        
    def _apply_mmse_filter(self, h: torch.Tensor, noise_power: float, user_port: Tuple[int, int] = None) -> torch.Tensor:
        """
        Apply MMSE filtering in blocks: h_mmse = C * (C + R)^-1 * h
        
        Process the channel in blocks of size mmse_block_size to improve efficiency.
        Each block is filtered separately with a smaller MMSE matrix.
        
        Args:
            h: Input channel estimate
            noise_power: Noise power estimate
            user_port: Optional tuple of (user_idx, port_idx) to use specific matrices
            
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
              # Get block-specific MMSE matrices - use user_port specific if provided
            C_block = self._get_block_C_matrix(current_block_size, user_port)
            R_block = self._get_block_R_matrix(noise_power, current_block_size, user_port)
            
            # Apply MMSE filter to block
            h_block_vec = h_block.reshape(-1, 1)  # Convert to column vector
            
            # MMSE formula: C * (C + R)^-1 * h
            mmse_filter = C_block @ torch.inverse(C_block + R_block)
            h_block_mmse = mmse_filter @ h_block_vec
            
            # Store filtered block in output
            h_mmse[start_idx:end_idx] = h_block_mmse.reshape(-1)
        return h_mmse
        
    def _get_block_C_matrix(self, block_size: int, user_port: Tuple[int, int] = None) -> torch.Tensor:
        """
        Get channel correlation matrix C for a specific block size
        
        Args:
            block_size: Size of the block to process
            user_port: Optional tuple of (user_idx, port_idx) for user/port specific matrices
            
        Returns:
            C matrix for MMSE filtering of the specified block
        """
        # Check if we have a user-specific matrix
        if user_port is not None and hasattr(self, 'C_matrices') and user_port in self.C_matrices:
            user_matrix = self.C_matrices[user_port]
            # Check if dimensions match
            if user_matrix.shape[0] == block_size:
                return user_matrix
        
        # If not, try the global matrix
        if self.C_matrix is not None:
            # Check if dimensions match
            if self.C_matrix.shape[0] == block_size:
                return self.C_matrix
            else:
                # We have a matrix but with different dimensions
                # For now, fallback to traditional method
                pass
        
        # Fallback to traditional method: construct based on exponential decay model for this block
        C = torch.zeros((block_size, block_size), dtype=torch.complex64, device=self.device)
        
        # Exponential power delay profile
        tau = 0.1  # Time constant
        for i in range(block_size):
            for j in range(block_size):
                delay_diff = abs(i - j)                # Convert scalar to tensor before using torch.exp
                exponent = torch.tensor(-delay_diff / (tau * block_size), device=self.device)
                C[i, j] = torch.exp(exponent)
        
        return C
        
    def _get_block_R_matrix(self, noise_power: float, block_size: int, user_port: Tuple[int, int] = None) -> torch.Tensor:
        """
        Get noise correlation matrix R for a specific block size
        
        Args:
            noise_power: Estimated noise power
            block_size: Size of the block to process
            user_port: Optional tuple of (user_idx, port_idx) for user/port specific matrices
            
        Returns:
            R matrix for MMSE filtering of the specified block
        """
        # Check if we have a user-specific matrix
        if user_port is not None and hasattr(self, 'R_matrices') and user_port in self.R_matrices:
            user_matrix = self.R_matrices[user_port]
            # Check if dimensions match
            if user_matrix.shape[0] == block_size:
                return user_matrix
        
        # If not, try the global matrix
        if self.R_matrix is not None:
            # Check if dimensions match
            if self.R_matrix.shape[0] == block_size:
                return self.R_matrix
            else:
                # We have a matrix but with different dimensions
                # For now, fallback to traditional method
                pass
                
        # Fallback to traditional method: diagonal matrix with noise power
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
            
    def set_mmse_matrices(self, C: torch.Tensor = None, R: torch.Tensor = None, user_port: Tuple[int, int] = None):
        """
        Set custom MMSE filter matrices
        
        Args:
            C: Channel correlation matrix
            R: Noise correlation matrix
            user_port: Optional tuple of (user_idx, port_idx) for user/port specific matrices
        """
        if user_port is not None:
            # Initialize dictionaries if they don't exist yet
            if not hasattr(self, 'C_matrices'):
                self.C_matrices = {}
            if not hasattr(self, 'R_matrices'):
                self.R_matrices = {}
            
            # Store matrices for this specific user/port
            if C is not None:
                self.C_matrices[user_port] = C
            if R is not None:
                self.R_matrices[user_port] = R
        else:
            # Legacy behavior - set global matrices
            if C is not None:
                self.C_matrix = C
            if R is not None:
                self.R_matrix = R