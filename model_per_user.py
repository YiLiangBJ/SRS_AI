import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class SRSChannelEstimatorPerUser(nn.Module):
    """
    SRS Channel Estimator using AI-enhanced methods with per-user and per-port MMSE matrices
    
    This module implements the SRS channel estimation process as described,
    with separate C and R matrices for each user and port.
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
        self.device = device
        
        # 为每个用户的每个端口存储不同的C和R矩阵
        # 使用字典来保存每个用户和端口对应的矩阵
        self.C_matrices = {}
        self.R_matrices = {}

    def forward(
        self, 
        ls_estimate: torch.Tensor,
        cyclic_shifts: List[List[int]],
        noise_power: float
    ) -> List[torch.Tensor]:
        """
        Run through channel estimation process
        
        Args:
            ls_estimate: Least squares channel estimate in frequency domain
            cyclic_shifts: Cyclic shifts for each user and port
            noise_power: Estimated noise power
            
        Returns:
            List of channel estimates for each user/port
        """
        # Convert LS estimate to time domain
        time_domain = self._frequency_to_time(ls_estimate)
        
        # Get time domain estimates for each user/port
        user_time_estimates = []
        user_port_indices = []  # 跟踪用户和端口索引
        
        for u, user_shifts in enumerate(cyclic_shifts):
            for p, shift in enumerate(user_shifts):
                if shift >= 0:  # Skip inactive ports (marked with -1)
                    # Extract user's channel based on cyclic shift
                    user_channel = self._extract_user_channel(time_domain, shift)
                    user_time_estimates.append(user_channel)
                    user_port_indices.append((u, p))  # 保存用户和端口的索引
        
        # Apply MMSE filtering for each user/port
        filtered_estimates = []
        for idx, (user_channel, (user_idx, port_idx)) in enumerate(zip(user_time_estimates, user_port_indices)):
            # 为每个用户和端口应用特定的MMSE滤波
            filtered = self._apply_mmse_filter(user_channel, noise_power, user_idx, port_idx)
            filtered_estimates.append(filtered)
        
        return filtered_estimates

    def _frequency_to_time(self, freq_domain: torch.Tensor) -> torch.Tensor:
        """
        Convert frequency domain estimate to time domain
        
        Args:
            freq_domain: Frequency domain signal
            
        Returns:
            Time domain signal
        """
        # Apply inverse FFT to convert to time domain
        time_domain = torch.fft.ifft(freq_domain)
        return time_domain
    
    def _extract_user_channel(self, time_domain: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Extract a user's channel based on cyclic shift
        
        Args:
            time_domain: Time domain signal
            shift: Cyclic shift for this user
            
        Returns:
            Extracted channel for this user
        """
        # Adjust for cyclic shift
        L = self.seq_length
        user_channel = torch.roll(time_domain, shifts=shift, dims=0)
        
        # Keep only first K=12 taps (assuming non-zero channel taps within this range)
        user_channel = user_channel[:self.K]
        
        return user_channel
    
    def _apply_mmse_filter(self, h_ls: torch.Tensor, noise_power: float, user_idx: int, port_idx: int) -> torch.Tensor:
        """
        Apply MMSE filtering to a user's channel
        
        Args:
            h_ls: User's least squares channel estimate in time domain
            noise_power: Estimated noise power
            user_idx: User index
            port_idx: Port index for that user
            
        Returns:
            MMSE filtered channel
        """
        # Initialize filtered channel same shape as input
        h_mmse = torch.zeros_like(h_ls)
        
        # Get user-port specific key
        user_port_key = (user_idx, port_idx)
        
        # Get length of this channel
        L = len(h_ls)
        
        # Process in blocks if L > mmse_block_size
        if L <= self.mmse_block_size:
            # Small enough to process as one block
            # Use user and port specific matrices if available
            # 使用指定用户和端口的C和R矩阵
            C_matrix = self._get_block_C_matrix(L, user_idx, port_idx)
            R_matrix = self._get_block_R_matrix(noise_power, L, user_idx, port_idx)
            
            # Apply MMSE filter: C * (C + R)^-1 * h
            mmse_filter = C_matrix @ torch.inverse(C_matrix + R_matrix)
            h_mmse = mmse_filter @ h_ls.reshape(-1, 1)
            h_mmse = h_mmse.reshape(-1)
        else:
            # Process in blocks
            for start_idx in range(0, L, self.mmse_block_size):
                end_idx = min(start_idx + self.mmse_block_size, L)
                current_block_size = end_idx - start_idx
                
                # Extract current block
                h_block = h_ls[start_idx:end_idx]
                
                # Get user and port specific MMSE matrices for this block
                C_block = self._get_block_C_matrix(current_block_size, user_idx, port_idx)
                R_block = self._get_block_R_matrix(noise_power, current_block_size, user_idx, port_idx)
                
                # Apply MMSE filter to block
                h_block_vec = h_block.reshape(-1, 1)  # Convert to column vector
                
                # MMSE formula: C * (C + R)^-1 * h
                mmse_filter = C_block @ torch.inverse(C_block + R_block)
                h_block_mmse = mmse_filter @ h_block_vec
                
                # Store filtered block in output
                h_mmse[start_idx:end_idx] = h_block_mmse.reshape(-1)
        
        return h_mmse
        
    def _get_block_C_matrix(self, block_size: int, user_idx: int, port_idx: int) -> torch.Tensor:
        """
        Get channel correlation matrix C for a specific block size and user/port
        
        Args:
            block_size: Size of the block to process
            user_idx: User index
            port_idx: Port index
            
        Returns:
            C matrix for MMSE filtering of the specified block
        """
        # Generate key for this user and port
        user_port_key = (user_idx, port_idx)
        
        # If we have a stored C_matrix for this user and port, use it directly
        if user_port_key in self.C_matrices:
            C_matrix = self.C_matrices[user_port_key]
            # Check if dimensions match
            if C_matrix.shape[0] == block_size:
                return C_matrix
        
        # Fallback to traditional method: construct based on exponential decay model for this block
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
        
    def _get_block_R_matrix(self, noise_power: float, block_size: int, user_idx: int, port_idx: int) -> torch.Tensor:
        """
        Get noise correlation matrix R for a specific block size and user/port
        
        Args:
            noise_power: Estimated noise power
            block_size: Size of the block to process
            user_idx: User index
            port_idx: Port index
            
        Returns:
            R matrix for MMSE filtering of the specified block
        """
        # Generate key for this user and port
        user_port_key = (user_idx, port_idx)
        
        # If we have a stored R_matrix for this user and port, use it directly
        if user_port_key in self.R_matrices:
            R_matrix = self.R_matrices[user_port_key]
            # Check if dimensions match
            if R_matrix.shape[0] == block_size:
                return R_matrix
                
        # Fallback to traditional method: diagonal matrix with noise power
        R = torch.eye(block_size, device=self.device) * noise_power
        return R

    def set_mmse_matrices(self, C: torch.Tensor = None, R: torch.Tensor = None, user_idx: int = 0, port_idx: int = 0):
        """
        Set custom MMSE filter matrices for specific user and port
        
        Args:
            C: Channel correlation matrix
            R: Noise correlation matrix
            user_idx: User index
            port_idx: Port index
        """
        user_port_key = (user_idx, port_idx)
        
        if C is not None:
            self.C_matrices[user_port_key] = C
        if R is not None:
            self.R_matrices[user_port_key] = R
