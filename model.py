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
            h_reconstructed += h_interp / phasor_m
        
        # Calculate residual
        residual = ls_estimate - h_reconstructed
          # Add residual back to each estimate
        final_estimates = []
        
        # Create a dictionary to store processed channels for each user-port combination
        processed_channels = {}
        for u, p, h_interp, phasor_m, phasor_T, phasor_ideal in h_processed_list:
            processed_channels[(u, p)] = (h_interp, phasor_m, phasor_T)
          # Process each user-port combination to get final estimates
        for u in range(num_users):
            for p in range(num_ports_per_user[u]):
                h_interp, phasor_m, phasor_T = processed_channels[(u, p)]
                  # Add residual with appropriate phase correction
                h_with_residual = h_interp + residual * phasor_m
                
                # 保存相位校正后的信道信息，可以用作MMSE矩阵生成的输入
                # 为每个用户/端口单独存储h_with_residual_phasor
                self.current_h_with_residual_phasors[(u, p)] = h_with_residual / phasor_T
                # 同时更新单个变量以保持向后兼容 - 保存最后一个处理的值
                # self.current_h_with_residual_phasor = h_with_residual / phasor_m
                  
                # Apply MMSE filtering
                if noise_power is None:
                    noise_power = self._estimate_noise_power(ls_estimate)
                
                h_mmse = self._apply_mmse_filter(h_with_residual, noise_power)
                phasor_m = self._generate_phasor(timing_offsets[(u, p)])
                h_mmse_aligned = h_mmse / phasor_m
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


class TrainableMMSEModule(nn.Module):
    """
    Trainable MMSE Filter Module
    
    This module can be used to train the C and R matrices for MMSE filtering
    """    
    def __init__(self, seq_length: int, mmse_block_size: int = 12, hidden_dim: int = 64, use_complex_input: bool = False):
        """
        Initialize the trainable MMSE module with Cholesky decomposition
        
        Args:
            seq_length: Length of sequence (L)
            mmse_block_size: Size of blocks for MMSE filtering
            hidden_dim: Hidden dimension for the neural network
            use_complex_input: Whether to use complex input (real + imaginary parts)
        """
        super().__init__()
        self.seq_length = seq_length
        self.mmse_block_size = mmse_block_size
        self.use_complex_input = use_complex_input
        
        # Input dimension depends on whether we use complex input
        input_dim = seq_length * 2 if use_complex_input else seq_length
        
        # Calculate parameter counts for Cholesky factors
        n = mmse_block_size
        # 下三角矩阵的参数计算
        # 对角线元素 (只有实部) = n
        # 严格下三角元素 (实部+虚部) = n*(n-1)/2
        diag_size = n  # 对角线元素数量
        off_diag_size = n * (n - 1) // 2  # 严格下三角元素数量
        
        # 实部参数 = 对角线元素 + 严格下三角元素
        real_params = diag_size + off_diag_size
        # 虚部参数 = 只有严格下三角元素有虚部
        imag_params = off_diag_size
        
        # 总参数量
        c_matrix_size = real_params + imag_params
        r_matrix_size = real_params + imag_params
          # Networks to generate C and R matrices (only upper triangular parts)
        self.C_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, c_matrix_size)  # Only upper triangular elements
        )
          # R矩阵也使用完整的频域样本作为输入，而不仅是噪声功率
        self.R_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Use channel data instead of just noise power
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, r_matrix_size)  # Only upper triangular elements
        )    
    def forward(self, channel_stats: torch.Tensor, noise_power: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate C and R matrices for specified block size using Cholesky decomposition
        
        Args:
            channel_stats: Channel statistics (can be frequency domain samples)
               - If use_complex_input=False: magnitude of frequency domain samples (shape [L])
               - If use_complex_input=True: concatenated real and imag parts (shape [2*L]) or complex tensor
            noise_power: Estimated noise power (not used directly anymore, kept for API compatibility)
            
        Returns:
            C: Channel correlation matrix (mmse_block_size x mmse_block_size)
            R: Noise and interference correlation matrix (mmse_block_size x mmse_block_size)
            pd_loss: 正定性约束损失 (在使用Cholesky分解时不再需要，但保留用于API兼容)
        """
        # Process input based on configuration
        if self.use_complex_input and channel_stats.is_complex():
            # Split complex input into real and imaginary parts and concatenate
            input_tensor = torch.cat([torch.real(channel_stats), torch.imag(channel_stats)])
        else:
            # Use input as-is (should be magnitude if use_complex_input=False)
            input_tensor = channel_stats
        
        n = self.mmse_block_size
        
        # 基于Cholesky分解生成C矩阵
        # 生成下三角矩阵L，然后计算C = L @ L^H
        # 为了生成下三角矩阵，我们需要n*(n+1)//2个参数（实部和虚部）
        
        # 生成下三角矩阵L的元素
        C_flat = self.C_generator(input_tensor)
        
        # 计算实部和虚部的参数数量
        # 对角线元素只有实部，其他元素有实部和虚部
        diag_size = n  # 对角线元素数量
        off_diag_size = n * (n - 1) // 2  # 非对角线元素数量
        
        # 实部总共需要的参数数量 = 对角线元素 + 非对角线元素
        real_size = diag_size + off_diag_size
        # 虚部只需要非对角线元素
        imag_size = off_diag_size
        
        # 分割网络输出为实部和虚部
        L_real_flat = C_flat[:real_size]
        L_imag_flat = C_flat[real_size:real_size + imag_size]
        
        # 创建零矩阵
        L_real = torch.zeros((n, n), device=C_flat.device)
        L_imag = torch.zeros((n, n), device=C_flat.device)
        
        # 填充下三角矩阵的实部
        idx = 0
        for i in range(n):
            for j in range(i + 1):  # j <= i 表示下三角
                if i == j:  # 对角线元素只有实部，并且必须为正
                    L_real[i, j] = torch.nn.functional.softplus(L_real_flat[idx])
                else:
                    L_real[i, j] = L_real_flat[idx]
                idx += 1
        
        # 填充下三角矩阵的虚部（对角线元素没有虚部）
        idx = 0
        for i in range(n):
            for j in range(i):  # j < i 表示严格下三角
                L_imag[i, j] = L_imag_flat[idx]
                idx += 1
                
        # 组合成复数下三角矩阵L
        L = torch.complex(L_real, L_imag)
        
        # 计算C = L @ L^H (L^H是L的共轭转置)
        # 这样构造的矩阵自然是厄米特正定的
        C = L @ L.conj().transpose(0, 1)
        
        # 对R矩阵执行相同的操作
        # 使用与C相同的输入，让网络自己学习提取噪声和干扰特性
        R_flat = self.R_generator(input_tensor)
        
        # 分割网络输出为实部和虚部
        L_real_flat = R_flat[:real_size]
        L_imag_flat = R_flat[real_size:real_size + imag_size]
        
        # 创建零矩阵
        L_real = torch.zeros((n, n), device=R_flat.device)
        L_imag = torch.zeros((n, n), device=R_flat.device)
        
        # 填充下三角矩阵的实部
        idx = 0
        for i in range(n):
            for j in range(i + 1):  # j <= i 表示下三角
                if i == j:  # 对角线元素只有实部，并且必须为正
                    L_real[i, j] = torch.nn.functional.softplus(L_real_flat[idx])
                else:
                    L_real[i, j] = L_real_flat[idx]
                idx += 1
        
        # 填充下三角矩阵的虚部（对角线元素没有虚部）
        idx = 0
        for i in range(n):
            for j in range(i):  # j < i 表示严格下三角
                L_imag[i, j] = L_imag_flat[idx]
                idx += 1
                
        # 组合成复数下三角矩阵L
        L = torch.complex(L_real, L_imag)
        
        # 计算R = L @ L^H
        R = L @ L.conj().transpose(0, 1)        # 使用Cholesky分解生成的矩阵已经保证是厄米特正定的
        # 但为了数值稳定性，我们可以添加一个很小的对角加载
        epsilon = 1e-6
        C = C + torch.eye(n, device=C.device) * epsilon
        R = R + torch.eye(n, device=R.device) * epsilon
        
        # 为了与原来的API兼容，仍然返回pd_loss（但不再需要计算）
        pd_loss = None if not self.training else torch.tensor(0.0, device=C.device)
        
        return C, R, pd_loss
    def _ensure_positive_definite(self, matrix: torch.Tensor, min_eigenvalue: float = 1e-6) -> torch.Tensor:
        """
        确保复矩阵是正定的，方法是确保所有特征值都大于指定的最小值
        
        Args:
            matrix: 输入的复矩阵
            min_eigenvalue: 最小特征值阈值
            
        Returns:
            正定的复矩阵
        """
        # 检查矩阵是否已经是厄米特矩阵
        n = matrix.shape[0]
        is_hermitian = torch.allclose(matrix, matrix.conj().transpose(0, 1), atol=1e-5)
        
        if not is_hermitian:
            # 如果不是厄米特矩阵，先将其转换为厄米特矩阵
            matrix = 0.5 * (matrix + matrix.conj().transpose(0, 1))
        
        # 计算特征值和特征向量
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            
            # 对特征值进行限制，确保都大于min_eigenvalue
            adjusted_eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
            
            # 重建矩阵 U * diag(λ) * U^H
            reconstructed = eigenvectors @ torch.diag(adjusted_eigenvalues) @ eigenvectors.conj().transpose(0, 1)
            
            # 确保结果是厄米特矩阵
            reconstructed = 0.5 * (reconstructed + reconstructed.conj().transpose(0, 1))
            
            return reconstructed
        except Exception:
            # 如果分解失败，使用简单的对角加载
            return matrix + torch.eye(n, device=matrix.device) * min_eigenvalue
    def compute_positive_definite_loss(self, matrix: torch.Tensor, margin: float = 1e-4) -> torch.Tensor:
        """
        计算矩阵正定性约束的损失
        促使网络学习生成自然正定的矩阵，而不是依赖后处理
        
        Args:
            matrix: 输入复矩阵
            margin: 特征值下界的边界值
            
        Returns:
            表示非正定程度的损失值
        """        # 计算厄米特性损失 - 直接在原始矩阵上操作
        hermitian_loss = torch.norm(matrix - matrix.conj().transpose(0, 1)) 
        
        # 添加对角元素的约束 - 直接在原始矩阵上操作
        diag_real = torch.diagonal(torch.real(matrix), 0)
        diag_imag = torch.diagonal(torch.imag(matrix), 0)
        
        # 对角元素虚部应为0，实部应为正
        diag_imag_loss = torch.sum(diag_imag**2)  # 惩罚对角元素的虚部
        diag_real_neg_loss = torch.sum(torch.clamp(-diag_real + margin, min=0)**2)  # 惩罚对角元素小于margin
        
        # 计算特征值相关的损失 - 直接使用原始矩阵本身计算特征值
        try:
            # 使用原始矩阵matrix计算特征值，而不是matrix_h
            eigenvalues = torch.linalg.eigvalsh(matrix)
            
            # 对于小于margin的特征值，给予惩罚
            eigenvalue_loss = torch.sum(torch.clamp(margin - eigenvalues, min=0)**2)
            
            # 组合所有损失，包括厄米特性损失
            return eigenvalue_loss + diag_imag_loss + diag_real_neg_loss + hermitian_loss
        except Exception:
            # 如果分解失败，返回一个基于矩阵结构的约束损失
            n = matrix.shape[0]
            identity = torch.eye(n, device=matrix.device)
            structure_loss = torch.norm(matrix @ matrix.conj().transpose(0, 1) - identity)
            
            # 即使特征值计算失败，也要返回对角元素和厄米特性的约束
            return structure_loss + diag_imag_loss + diag_real_neg_loss + hermitian_loss
