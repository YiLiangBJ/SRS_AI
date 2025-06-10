import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from model import SRSChannelEstimator

class SRSChannelEstimatorUnified(SRSChannelEstimator):
    """
    SRS Channel Estimator using AI-enhanced methods with unified MMSE approach
    
    This module extends the SRSChannelEstimator to use h_with_residual/phasor
    as input for MMSE matrix generation, with all users sharing the same network.
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
        Initialize the SRS Channel Estimator with unified MMSE
        
        Args:
            seq_length: Length of SRS sequence (L)
            ktc: Configuration parameter (ktc=4 -> K=12, ktc=2 -> K=8)
            max_users: Maximum number of users to support
            max_ports_per_user: Maximum number of ports across all users
            mmse_block_size: Size of blocks for MMSE filtering
            device: Computation device
        """
        SRSChannelEstimator.__init__(
            self,
            seq_length=seq_length,
            ktc=ktc,
            max_users=max_users,
            max_ports_per_user=max_ports_per_user,
            mmse_block_size=mmse_block_size,
            device=device
        )
        
        # 用于存储临时的h_with_residual/phasor值
        self.current_channel_input = None
        
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
        time_domain = self._idft(ls_estimate)
        
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
        
        # 计算合并信号
        h_reconstructed = torch.zeros_like(ls_estimate, dtype=torch.complex64)
        processed_channels = {}
        timing_offsets = {}
        
        for idx, (user_channel, (u, p)) in enumerate(zip(user_time_estimates, user_port_indices)):
            # 计算理想峰值位置
            n_u_p = cyclic_shifts[u][p]
            ideal_peak = (self.K - n_u_p) % self.K * self.seq_length // self.K
            
            # 估计时间偏移
            T_u_p = self._estimate_timing_offset(user_channel, ideal_peak, (-10, 10))
            timing_offsets[(u, p)] = T_u_p
            
            # 计算循环移位量
            m_u_p = (T_u_p + ideal_peak)
            
            # 生成相位因子
            phasor = self._generate_phasor(m_u_p)
            
            # 移位信道以对齐峰值
            h_u_p = ls_estimate * phasor
            
            # 应用OCC解复用
            Locc = self._compute_locc(cyclic_shifts)
            h_avg = self._apply_occ_demux(h_u_p, Locc)
            
            # 线性插值回到完整长度
            h_interpolated = self._linear_interpolation(h_avg, self.seq_length)
            
            processed_channels[(u, p)] = (h_interpolated, phasor)
            h_reconstructed += h_interpolated / phasor
        
        # 计算残差
        residual = ls_estimate - h_reconstructed
            
        # Apply MMSE filtering for each user/port
        final_estimates = []
        
        for u, p in user_port_indices:
            # 获取处理后的信道和相位因子
            h_interp, phasor = processed_channels[(u, p)]
            
            # 添加带有适当相位校正的残差
            h_with_residual = h_interp + residual * phasor
            
            # 将h_with_residual/phasor作为MMSE生成的输入
            self.current_channel_input = h_with_residual / phasor
            
            # 应用MMSE滤波
            h_mmse = self._apply_mmse_filter(h_with_residual, noise_power)
            
            # 相位校正和最终估计
            h_mmse_aligned = h_mmse / phasor
            final_estimates.append(h_mmse_aligned)
        
        return final_estimates
        
    def _apply_mmse_filter(self, h: torch.Tensor, noise_power: float) -> torch.Tensor:
        """
        Apply MMSE filtering in blocks using pre-calculated matrices
        
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
            
            # Get input for MMSE matrix generation
            if self.current_channel_input is not None:
                input_block = self.current_channel_input[start_idx:end_idx]
            else:
                input_block = h_block
            
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
