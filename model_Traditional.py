"""
SRS Channel Estimation Models using AI-enhanced methods

This module provides traditional MMSE filtering with AI enhancements for SRS channel estimation.
All computations are forced to run on CPU only.
"""

import os

# Force CPU-only execution - disable all CUDA/GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from model_AIpart import TrainableMMSEModule
# 保存原始 __repr__
original_repr = torch.Tensor.__repr__

# 自定义 __repr__，只显示形状和关键信息
def custom_repr(self):
    shape_str = str(tuple(self.shape))
    return f"Tensor(shape={shape_str}, dtype={self.dtype}, device={self.device})"

# 应用补丁
torch.Tensor.__repr__ = custom_repr
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class SRSChannelEstimator(nn.Module):
    def _generate_phasor_batch_ports(self, m_values: torch.Tensor, seq_length: int) -> torch.Tensor:
        """
        并行化端口的phasor生成
        m_values: [batch_size, total_ports]
        返回: [batch_size, total_ports, seq_length]
        """
        batch_size, total_ports = m_values.shape
        n = torch.arange(seq_length, device=self.device).view(1, 1, seq_length)
        m_expanded = m_values.unsqueeze(-1)  # [batch_size, total_ports, 1]
        phasors = torch.exp(1j * 2 * np.pi * m_expanded * n / seq_length)
        return phasors

    def _apply_occ_demux_batch_ports(self, h: torch.Tensor, Locc_tensor: torch.Tensor) -> torch.Tensor:
        """
        完全tensor并行化端口OCC去复用，无for
        h: [batch_size, num_rx_ant, total_ports, seq_length]
        Locc_tensor: [total_ports]
        返回: [batch_size, num_rx_ant, total_ports, seq_length//Locc]
        """
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        # 计算每个端口的分组长度
        reduced_lengths = (seq_length // Locc_tensor).to(torch.int64)  # [total_ports]
        max_reduced_length = reduced_lengths.max().item()
        # 完全tensor化，无for
        # 先将h展平成(batch_size*num_rx_ant, total_ports, seq_length)
        h_flat = h.reshape(batch_size*num_rx_ant, total_ports, seq_length)
        # 构造mask，所有端口统一处理
        # 计算每个端口的分组长度和分组数
        # 生成每个端口的分组数(reduced_length)和分组长度(Locc)
        # 生成最大分组数，便于统一shape
        max_reduced_length = reduced_lengths.max().item()
        # 构造分组索引
        idx = torch.arange(seq_length, device=h.device).unsqueeze(0).unsqueeze(0)  # [1,1,seq_length]
        # 生成每个端口的分组长度
        Locc_expand = Locc_tensor.view(1, total_ports, 1)
        reduced_length_expand = reduced_lengths.view(1, total_ports, 1)
        # 计算每个端口的有效分组数
        valid_mask = idx < (Locc_expand * reduced_length_expand)
        # 对每个端口，分组reshape
        h_masked = torch.where(valid_mask, h_flat, torch.zeros_like(h_flat))
        # 统一reshape为(batch_size*num_rx_ant, total_ports, max_reduced_length, Locc)
        h_grouped = torch.zeros(batch_size*num_rx_ant, total_ports, max_reduced_length, Locc_expand.max().item(), dtype=h.dtype, device=h.device)
        for p in range(total_ports):
            Locc = Locc_tensor[p].item()
            reduced_length = reduced_lengths[p].item()
            h_p = h_masked[:, p, :Locc*reduced_length]
            h_p_reshaped = h_p.reshape(batch_size*num_rx_ant, reduced_length, Locc)
            h_grouped[:, p, :reduced_length, :Locc] = h_p_reshaped
        # 在最后一维求平均
        h_avg = torch.mean(h_grouped, dim=-1)
        h_avg = h_avg.reshape(batch_size, num_rx_ant, total_ports, max_reduced_length)
        return h_avg

    def _linear_interpolation_batch_ports(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        完全tensor并行化端口线性插值，无for
        h_avg: [batch_size, num_rx_ant, total_ports, reduced_length]
        返回: [batch_size, num_rx_ant, total_ports, target_length]
        """
        batch_size, num_rx_ant, total_ports, reduced_length = h_avg.shape
        lseg = target_length // reduced_length
        hk_interp = torch.zeros(batch_size, num_rx_ant, total_ports, reduced_length * lseg, device=h_avg.device, dtype=h_avg.dtype)
        centerSc = (lseg - 1) / 2
        # slope: [batch_size, num_rx_ant, total_ports, reduced_length-1]
        slope = (h_avg[..., 1:] - h_avg[..., :-1]) / lseg
        # Left Edge (从0开始)
        xseg = torch.arange(0, int(centerSc)+1, device=h_avg.device)
        len_edge = xseg.shape[0]
        hinterp = slope[..., :1].expand(-1, -1, -1, len_edge) * (xseg-centerSc).view(1,1,1,-1) + h_avg[..., :1].expand(-1, -1, -1, len_edge)
        hk_interp[..., :len_edge] = hinterp
        # Middle SCs tensor化 (从0开始)
        mid_idx = torch.arange(reduced_length-1, device=h_avg.device)
        seg_start = centerSc + mid_idx * lseg + 1
        seg_end = seg_start + lseg
        xseg = torch.arange(int(centerSc)+1, lseg+int(centerSc)+1, device=h_avg.device).view(1, lseg) + mid_idx.view(-1, 1) * lseg
        slope_mid = slope.unsqueeze(-1)
        h_comp_mid = h_avg[..., :-1].unsqueeze(-1)
        xseg_center = xseg - centerSc
        hinterp = slope_mid * xseg_center.view(1,1,1,reduced_length-1,lseg) + h_comp_mid
        for k in range(reduced_length-1):
            hk_interp[..., int(seg_start[k]):int(seg_end[k])] = hinterp[..., k, :]
        # Right Edge (从0开始)
        xseg = torch.arange(int(xseg[0]), reduced_length*lseg, device=h_avg.device)
        len_edge = xseg.shape[0]
        hinterp = slope[..., -1:].expand(-1, -1, -1, len_edge) * (xseg-centerSc).view(1,1,1,-1) + h_avg[..., -1:].expand(-1, -1, -1, len_edge)
        hk_interp[..., int(xseg[0]):] = hinterp
        return hk_interp

    def _apply_mmse_filter_batch_ports(self, h: torch.Tensor) -> torch.Tensor:
        """
        并行化端口的MMSE滤波
        h: [batch_size, num_rx_ant, total_ports, seq_length]
        返回: [batch_size, num_rx_ant, total_ports, seq_length]
        """
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        # 假设mmse_module支持端口并行
        h_mmse = self.mmse_module.forward_batch_ports(h)
        return h_mmse

    def __init__(
        self,
        mmse_module: TrainableMMSEModule,
        device: str = "cpu"
    ):
        """
        SRSChannelEstimator 顶层初始化
        1. 先实例化底层 MMSE 模块
        2. 作为参数传递给 SRSChannelEstimator
        """
        super().__init__()
        self.device = device
        self.mmse_module = mmse_module

    def forward(
        self,
        ls_estimates: torch.Tensor,  # [batch_size, num_rx_ant, seq_length]
        srs_config
    ) -> torch.Tensor:
        """
        完全tensor化的SRS信道估计器forward
        输入: ls_estimates [batch_size, num_rx_ant, seq_length]
        输出: [batch_size, num_users, max_ports_per_user, num_rx_ant, seq_length]
        """
        batch_size, num_rx_ant, seq_length = ls_estimates.shape
        cyclic_shifts = srs_config.current_cyclic_shifts
        num_users = srs_config.num_users
        ports_per_user = srs_config.ports_per_user
        max_ports = max(ports_per_user)
        device = ls_estimates.device
        # K = srs_config.K
        # 自动设置 ifft_size 和 delay_search_range
        ifft_size = max(seq_length, 32)
        if srs_config.current_ktc == 4:
            K = 12
            dist = int(ifft_size / K / 2)
        else:
            K = 8
            dist = int(ifft_size / K / 2)
        delay_search_range = (-dist, dist)

        # 初始化输出张量
        total_ports = sum(ports_per_user)
        # 1. 时域变换
        h_time = torch.fft.ifft(ls_estimates, dim=-1, n=ifft_size)  # [batch_size, num_rx_ant, seq_length]
        h_power_sum = torch.sum(torch.abs(h_time) ** 2, dim=1)  # [batch_size, seq_length]

        # 2. 构造所有端口参数
        # tensor化展开所有端口参数
        n_u_p_tensor = torch.cat([
            torch.tensor(cyclic_shifts[user_id][:ports_per_user[user_id]], device=device)
            for user_id in range(num_users)
        ])  # [total_ports]
        ideal_peak_tensor = torch.cat([
            ((K - torch.tensor(cyclic_shifts[user_id][:ports_per_user[user_id]], device=device)) % K * ifft_size // K)
            for user_id in range(num_users)
        ])  # [total_ports]

        # 3. 批量估计timing offset
        timing_offset = self._estimate_timing_offset_batch_ports(h_power_sum, ideal_peak_tensor, delay_search_range)  # [batch_size, total_ports]

        m_value = timing_offset + ideal_peak_tensor.unsqueeze(0)  # [batch_size, total_ports]

        # 4. 批量生成phasor
        phasor_m = self._generate_phasor_batch_ports(m_value, seq_length)  # [batch_size, total_ports, seq_length]
        phasor_T = self._generate_phasor_batch_ports(timing_offset, seq_length)  # [batch_size, total_ports, seq_length]

        # 5. 广播到端口维度
        ls_estimates_expand = ls_estimates.unsqueeze(2).expand(-1, -1, total_ports, -1)  # [batch_size, num_rx_ant, total_ports, seq_length]
        phasor_m_expand = phasor_m.unsqueeze(1)  # [batch_size, 1, total_ports, seq_length]
        h_shifted = ls_estimates_expand * phasor_m_expand  # [batch_size, num_rx_ant, total_ports, seq_length]

        # 6. OCC去复用和插值
        Locc = 4  # [total_ports]
        h_reshape = h_shifted.reshape(batch_size, num_rx_ant, total_ports, seq_length//Locc, Locc)
        h_avg = torch.mean(h_reshape, dim=-1)
        h_interpolated = self._linear_interpolation_batch_ports(h_avg, seq_length)  # [batch_size, num_rx_ant, total_ports, seq_length]

        # 7. 计算global residual
        h_with_timing = h_interpolated * torch.conj(phasor_m_expand)
        total_reconstructed_signal = torch.sum(h_with_timing, dim=2)  # [batch_size, num_rx_ant, seq_length]
        global_residual = ls_estimates - total_reconstructed_signal  # [batch_size, num_rx_ant, seq_length]
        global_residual_expand = global_residual.unsqueeze(2).expand(-1, -1, total_ports, -1)
        h_with_residual = h_interpolated + global_residual_expand * phasor_m_expand

        # 8. MMSE滤波和恢复timing信息
        h_mmse_aligned = self._apply_mmse_filter_batch_ports(h_with_residual)  # [batch_size, num_rx_ant, total_ports, seq_length]
        phasor_T_expand = phasor_T.unsqueeze(1)  # [batch_size, 1, total_ports, seq_length]
        out = h_mmse_aligned * torch.conj(phasor_T_expand)
        return out
    
        # 添加单样本版本的辅助方法
    def _apply_occ_demux_single(self, h: torch.Tensor, Locc: int) -> torch.Tensor:
        """单样本版本的OCC去复用"""
        num_rx_ant, L = h.shape
        h_reshaped = h.reshape(num_rx_ant, L // Locc, Locc)
        h_avg = torch.mean(h_reshaped, dim=-1)
        return h_avg

    def _linear_interpolation_single(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """单样本版本的线性插值"""
        num_rx_ant, reduced_length = h_avg.shape
        h_interp = torch.zeros(num_rx_ant, target_length, dtype=torch.complex64, device=self.device)
        
        for a in range(num_rx_ant):
            h_interp[a, :] = self._linear_interpolation(h_avg[a, :], target_length)
        
        return h_interp

    def _apply_mmse_filter_single(self, h_input: torch.Tensor, user_port: Optional[Tuple[int, int]] = None):
        """单样本版本的MMSE滤波"""
        if self.mmse_module is None:
            return h_input
        
        # 添加batch维度，调用batch版本，然后移除batch维度
        h_batch = h_input.unsqueeze(0)
        h_filtered_batch = self._apply_mmse_filter_batch(h_batch, user_port)
        return h_filtered_batch.squeeze(0)

    def _apply_mmse_filter_chunked(self, h_input: torch.Tensor, user_port: Optional[Tuple[int, int]] = None):
        """
        Apply MMSE filtering using chunked processing for variable sequence lengths
        
        This method handles variable-length sequences by:
        1. Using the MLP to generate C and R matrices for each chunk
        2. Applying MMSE filtering to each chunk separately
        3. Concatenating the results to form the full-length filtered output
        
        Args:
            h_input: Input channel estimates [num_rx_ant, seq_length]
            user_port: Optional user/port tuple for per-user matrices
            
        Returns:
            Filtered channel estimates [num_rx_ant, seq_length]
        """
        num_rx_ant, seq_length = h_input.shape
        
        # Get chunk size from MMSE block size
        chunk_size = self.mmse_block_size
        
        # Calculate number of chunks needed
        num_chunks = (seq_length + chunk_size - 1) // chunk_size
        
        # Prepare output tensor
        h_mmse = torch.zeros_like(h_input)
        
        # For the reference antenna, use the most significant one
        antenna_energies = torch.sum(torch.abs(h_input)**2, dim=1)
        max_energy_antenna_idx = torch.argmax(antenna_energies).item()
        reference_antenna_signal = h_input[max_energy_antenna_idx]
        
        # Generate C and R matrices for all chunks using the MMSE module
        if self.mmse_module is not None:
            C_matrices, R_matrices = self.mmse_module.forward_chunked(
                reference_antenna_signal, chunk_size=chunk_size)
        else:
            raise ValueError("MMSE module not available. Cannot perform MMSE filtering.")
        
        # Process each chunk separately
        for i in range(num_chunks):
            # Calculate chunk indices
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, seq_length)
            
            # Get chunk data
            chunk_data = h_input[:, start_idx:end_idx]
            
            # Reshape for MMSE filtering [num_rx_ant, chunk_actual_size] -> [num_rx_ant*chunk_actual_size]
            chunk_actual_size = end_idx - start_idx
            h_chunk_flat = chunk_data.reshape(-1)
            
            # Get C and R matrices for this chunk
            C = C_matrices[i]  # [mmse_block_size, mmse_block_size]
            R = R_matrices[i]  # [mmse_block_size, mmse_block_size]
            
            # If chunk is smaller than mmse_block_size, resize matrices
            if chunk_actual_size < chunk_size:
                C = C[:chunk_actual_size, :chunk_actual_size]
                R = R[:chunk_actual_size, :chunk_actual_size]
            
            # Perform MMSE filtering on the chunk
            # Calculate W = C / (C + R)
            W = torch.linalg.solve(C + R, C)
            
            # Apply filter to each antenna independently
            for ant in range(num_rx_ant):
                ant_chunk = chunk_data[ant]  # [chunk_actual_size]
                # Apply MMSE filter
                h_mmse[ant, start_idx:end_idx] = torch.mv(W, ant_chunk)
        
        return h_mmse
    
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
    
    def _generate_phasor_batch(self, m_values: torch.Tensor, seq_length: int = None) -> torch.Tensor:
        """
        Generate phase factors in batch mode
        
        Args:
            m_values: Shift amounts [batch_size]
            seq_length: Sequence length to use (if None, uses self.seq_length)
            
        Returns:
            phasors: Phase factors [batch_size, seq_length]
        """
        if seq_length is None:
            seq_length = self.seq_length
        batch_size = m_values.shape[0]
        n = torch.arange(seq_length, device=self.device).unsqueeze(0)  # [1, seq_length]
        m_expanded = m_values.unsqueeze(1)  # [batch_size, 1]
        
        # 广播计算 [batch_size, seq_length]
        phasors = torch.exp(1j * 2 * np.pi * m_expanded * n / seq_length)
        return phasors
    
    def _apply_occ_demux_batch(self, h: torch.Tensor, Locc: int) -> torch.Tensor:
        """
        Apply OCC demultiplexing in batch mode
        
        Args:
            h: Input sequence [batch_size, num_rx_ant, seq_length]
            Locc: OCC length
            
        Returns:
            h_avg: Averaged sequence [batch_size, num_rx_ant, seq_length//Locc]
        """
        batch_size, num_rx_ant, L = h.shape
        # 重塑为 [batch_size, num_rx_ant, L//Locc, Locc]
        h_reshaped = h.reshape(batch_size, num_rx_ant, L // Locc, Locc)
        # 在最后一维求平均 [batch_size, num_rx_ant, L//Locc]
        h_avg = torch.mean(h_reshaped, dim=-1)
        return h_avg
    
    def _linear_interpolation_batch(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Linear interpolation in batch mode
        
        Args:
            h_avg: Input sequence [batch_size, num_rx_ant, reduced_length]
            target_length: Target length
            
        Returns:
            h_interp: Interpolated sequence [batch_size, num_rx_ant, target_length]
        """
        batch_size, num_rx_ant, reduced_length = h_avg.shape
        
        # 计算组大小
        group_size = target_length // reduced_length
        
        # 创建输入和输出索引
        orig_indices = torch.tensor([np.mean(np.arange(group_size)) + i * group_size 
                                   for i in range(reduced_length)], device=self.device)
        new_indices = torch.arange(target_length, dtype=torch.float32, device=self.device)
        
        # Batch interpolation
        h_real = torch.real(h_avg)  # [batch_size, num_rx_ant, reduced_length]
        h_imag = torch.imag(h_avg)  # [batch_size, num_rx_ant, reduced_length]
        
        # 使用PyTorch的插值函数
        real_interp = torch.zeros(batch_size, num_rx_ant, target_length, device=self.device)
        imag_interp = torch.zeros(batch_size, num_rx_ant, target_length, device=self.device)
        
        for b in range(batch_size):
            for a in range(num_rx_ant):
                # 对每个天线进行插值
                real_interp[b, a, :] = torch.tensor(
                    np.interp(new_indices.cpu().numpy(), orig_indices.cpu().numpy(), h_real[b, a, :].cpu().numpy()),
                    device=self.device, dtype=torch.float32
                )
                imag_interp[b, a, :] = torch.tensor(
                    np.interp(new_indices.cpu().numpy(), orig_indices.cpu().numpy(), h_imag[b, a, :].cpu().numpy()),
                    device=self.device, dtype=torch.float32
                )
        
        return torch.complex(real_interp, imag_interp)
    
    def _estimate_noise_power_batch(self, h: torch.Tensor) -> torch.Tensor:
        """
        Estimate noise power in batch mode
        
        Args:
            h: Frequency domain channel estimates [batch_size, num_rx_ant, seq_length]
            
        Returns:
            noise_powers: Noise powers [batch_size]
        """
        # 计算相邻样本差分 [batch_size, num_rx_ant, seq_length-1]
        diff = h[..., 1:] - h[..., :-1]
        
        # 计算功率并在天线和频率维度上平均 [batch_size]
        noise_powers = torch.mean(torch.abs(diff)**2, dim=(1, 2)) / 2
        
        return noise_powers
    
    def _apply_mmse_filter_batch(self, h: torch.Tensor, user_port: Tuple[int, int] = None) -> torch.Tensor:
            
        batch_size, num_rx_ant, L = h.shape
        block_size = self.mmse_block_size
        
        # 初始化输出
        h_mmse = torch.zeros_like(h)
        
        # 处理每个块
        num_blocks = (L + block_size - 1) // block_size
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, L)
            current_block_size = end_idx - start_idx
            
            if current_block_size < 2:
                h_mmse[..., start_idx:end_idx] = h[..., start_idx:end_idx]
                continue
            
            # 提取块 [batch_size, num_rx_ant, current_block_size]
            h_block = h[..., start_idx:end_idx]
            
            # 从 MLP 获取 MMSE 矩阵
            C_block = self._get_block_C_matrix(user_port)
            R_block = self._get_block_R_matrix(user_port)
            
            # 批处理应用MMSE
            for b in range(batch_size):
                mmse_filter = C_block @ torch.inverse(C_block + R_block)

                for a in range(num_rx_ant):
                    h_vec = h_block[b, a, :].reshape(-1, 1)  # [current_block_size, 1]
                    h_filtered = mmse_filter @ h_vec
                    h_mmse[b, a, start_idx:end_idx] = h_filtered.reshape(-1)
        
        return h_mmse
    
    def _generate_phasor(self, m: int, seq_length: int = None) -> torch.Tensor:
        """
        Generate phasor for cyclic shifting in frequency domain
        
        Args:
            m: Shifting amount
            seq_length: Sequence length to use (if None, uses self.seq_length)
            
        Returns:
            Phasor vector of length L
        """
        if seq_length is None:
            seq_length = self.seq_length
        n = torch.arange(seq_length, device=self.device)
        return torch.exp(1j * 2 * np.pi * m * n / seq_length)
    
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
        
        # Handle edge cases
        if group_size == 0:
            # If target_length < len(h_avg), just truncate or use simple interpolation
            if target_length <= len(h_avg):
                # Simple linear interpolation between available points
                orig_indices = np.linspace(0, len(h_avg) - 1, len(h_avg))
            else:
                group_size = 1  # Fallback to avoid division by zero
        
        # Create properly positioned input indices (at the center of each group)
        # For example, if group_size=4, the centers would be at indices 1.5, 5.5, 9.5, etc.
        if group_size > 0:
            orig_indices = np.array([np.mean(np.arange(group_size)) + i * group_size for i in range(len(h_avg))])
        else:
            # Fallback: evenly spaced indices
            orig_indices = np.linspace(0, target_length - 1, len(h_avg))
        
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
        [OBSOLETE] Apply MMSE filtering in blocks - LEGACY METHOD
        
        ⚠️  This method is OBSOLETE and should not be used. 
        Use the batch version with MLP-based MMSE module instead.
        
        Traditional C/R matrix calculation methods have been deprecated in favor of 
        the TrainableMMSEModule (MLP-based) approach.
        
        Args:
            h: Input channel estimate
            noise_power: Noise power estimate
            user_port: Optional tuple of (user_idx, port_idx) to use specific matrices
            
        Returns:
            MMSE filtered channel estimate (or unfiltered if no MLP module)
        """
        print("WARNING: _apply_mmse_filter is OBSOLETE. Use _apply_mmse_filter_batch with MLP MMSE module.")
        
        if self.mmse_module is None:
            return h  # Return unfiltered if no MLP module
            
        # For single sample, convert to batch and call batch version
        h_batch = h.unsqueeze(0)  # Add batch dimension
        noise_powers = torch.tensor([noise_power], device=h.device)
        
        h_mmse_batch = self._apply_mmse_filter_batch(h_batch, noise_powers, user_port)
        return h_mmse_batch.squeeze(0)  # Remove batch dimension
        
    def _get_block_C_matrix(self, user_port: Tuple[int, int] = None) -> torch.Tensor:
        return self.C_matrices[user_port]
        
    def _get_block_R_matrix(self, user_port: Tuple[int, int] = None) -> torch.Tensor:
        return self.R_matrices[user_port]

    
    def _get_C_matrix(self) -> torch.Tensor:
        """
        [OBSOLETE] Get channel correlation matrix C for the entire sequence - LEGACY METHOD
        
        ⚠️  This method is OBSOLETE and should not be used.
        Traditional C/R matrix calculation methods have been deprecated in favor of 
        the TrainableMMSEModule (MLP-based) approach.
        
        Returns:
            C matrix for MMSE filtering
        """
        print("WARNING: _get_C_matrix is OBSOLETE. Use TrainableMMSEModule instead.")
        
        if self.C_matrix is not None:
            return self.C_matrix
        else:
            # Fallback: return identity matrix instead of exponential model
            L = self.seq_length
            return torch.eye(L, dtype=torch.complex64, device=self.device)
    def _get_R_matrix(self, noise_power: float) -> torch.Tensor:
        """
        [OBSOLETE] Get noise correlation matrix R for the entire sequence - LEGACY METHOD
        
        ⚠️  This method is OBSOLETE and should not be used.
        Traditional C/R matrix calculation methods have been deprecated in favor of 
        the TrainableMMSEModule (MLP-based) approach.
        
        Args:
            noise_power: Estimated noise power
            
        Returns:
            R matrix for MMSE filtering
        """
        print("WARNING: _get_R_matrix is OBSOLETE. Use TrainableMMSEModule instead.")
        
        if self.R_matrix is not None:
            return self.R_matrix
        else:
            # Fallback: diagonal matrix with noise power
            L = self.seq_length
            R = torch.eye(L, device=self.device) * noise_power
            return R
            
    def set_mmse_matrices(self, C: torch.Tensor = None, R: torch.Tensor = None, user_port: Tuple[int, int] = None):
        """
        Set MMSE filter matrices generated by the MLP-based TrainableMMSEModule
        
        This method is used to store the C and R matrices generated by the TrainableMMSEModule.
        It should NOT be used to set manually calculated matrices using traditional methods.
        
        Args:
            C: Channel correlation matrix from MLP
            R: Noise correlation matrix from MLP  
            user_port: Optional tuple of (user_idx, port_idx) for user/port specific matrices
        """
        if user_port is not None:
            # Initialize dictionaries if they don't exist yet
            if not hasattr(self, 'C_matrices'):
                self.C_matrices = {}
            if not hasattr(self, 'R_matrices'):
                self.R_matrices = {}
            
            # Store matrices for this specific user/port (from MLP)
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

    def _estimate_timing_offset_batch_ports(self, h_power_sum: torch.Tensor, ideal_peak_tensor: torch.Tensor, search_range: Tuple[int, int]) -> torch.Tensor:
        """
        完全tensor化端口的时序偏移估计，无for
        h_power_sum: [batch_size, seq_length]
        ideal_peak_tensor: [total_ports]
        search_range: (min_offset, max_offset)
        返回: [batch_size, total_ports]
        """
        batch_size, seq_length = h_power_sum.shape
        total_ports = ideal_peak_tensor.shape[0]
        min_offset, max_offset = search_range
        # [batch_size, total_ports, seq_length]
        ideal_peak = ideal_peak_tensor.view(1, total_ports, 1).expand(batch_size, total_ports, seq_length)
        idx = torch.arange(seq_length, device=h_power_sum.device).view(1, 1, seq_length)
        start_pos = (ideal_peak_tensor + min_offset) % seq_length  # [total_ports]
        end_pos = (ideal_peak_tensor + max_offset) % seq_length    # [total_ports]
        start_pos = start_pos.view(1, total_ports, 1)
        end_pos = end_pos.view(1, total_ports, 1)
        # wraparound mask: [batch_size, total_ports, seq_length]
        mask1 = (start_pos <= end_pos) & (idx >= start_pos) & (idx <= end_pos)
        mask2 = (start_pos > end_pos) & ((idx >= start_pos) | (idx <= end_pos))
        search_mask = mask1 | mask2  # [1, total_ports, seq_length]
        search_mask = search_mask.expand(batch_size, total_ports, seq_length)
        h_mag = h_power_sum.unsqueeze(1).expand(-1, total_ports, -1)  # [batch_size, total_ports, seq_length]
        masked = torch.where(search_mask, h_mag, torch.tensor(-float('inf'), device=h_mag.device))
        peak_pos = torch.argmax(masked, dim=-1)  # [batch_size, total_ports]
        ideal_peak_val = ideal_peak_tensor.view(1, total_ports).expand(batch_size, total_ports)
        offset = (peak_pos - ideal_peak_val + seq_length) % seq_length
        offset = torch.where(offset > seq_length // 2, offset - seq_length, offset)
        return offset.to(torch.int32)

    def _linear_interpolation_batch_ports(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        完全tensor化端口的线性插值，无for和np.interp，使用PyTorch插值实现
        h_avg: [batch_size, num_rx_ant, total_ports, reduced_length]
        返回: [batch_size, num_rx_ant, total_ports, target_length]
        """
        batch_size, num_rx_ant, total_ports, reduced_length = h_avg.shape
        lseg = target_length // reduced_length
        hk_interp = torch.zeros(batch_size, num_rx_ant, total_ports, reduced_length * lseg, device=h_avg.device, dtype=h_avg.dtype)
        centerSc = (lseg - 1) / 2
        # slope: [batch_size, num_rx_ant, total_ports, reduced_length-1]
        slope = (h_avg[..., 1:] - h_avg[..., :-1]) / lseg
        # Left Edge
        len_edge = lseg // 2 
        xseg = torch.arange(len_edge, device=h_avg.device)
        hinterp4d = slope[..., :1].expand(-1, -1, -1, len_edge) * (xseg-centerSc).view(1,1,1,-1) + h_avg[..., :1].expand(-1, -1, -1, len_edge)
        hk_interp[..., :len_edge] = hinterp4d
        # Middle SCs tensor化
        mid_idx = torch.arange(reduced_length-1, device=h_avg.device)
        # seg_start = len_edge + mid_idx * lseg
        seg_end = len_edge + lseg
        xseg = torch.arange(len_edge, lseg+len_edge, device=h_avg.device).view(1, lseg) + mid_idx.view(-1, 1) * lseg
        slope_mid = slope.unsqueeze(-1)
        h_comp_mid = h_avg[..., :-1].unsqueeze(-1)
        xseg_center = xseg - (centerSc + mid_idx.view(-1, 1) * lseg)
        hinterp5d = slope_mid * xseg_center.view(1,1,1,reduced_length-1,lseg) + h_comp_mid
        hk_interp[..., len_edge:-len_edge] = hinterp5d.reshape(*hinterp5d.shape[:-2], hinterp5d.shape[-2] * hinterp5d.shape[-1])
        # Right Edge
        xseg = torch.arange(xseg[-1,-1]+1, reduced_length*lseg, device=h_avg.device)
        hinterp4d = slope[..., -1:].expand(-1, -1, -1, len_edge) * (xseg-(centerSc+(reduced_length-1)*lseg)).view(1,1,1,-1) + h_avg[..., -1:].expand(-1, -1, -1, len_edge)
        hk_interp[..., -len_edge:] = hinterp4d
        return hk_interp

    def _apply_mmse_filter_batch_ports(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        h_flat = h.reshape(-1, seq_length)  # [batch_size*num_rx_ant*total_ports, seq_length]
        # 直接拼接每个端口的 MMSE 估计结果，shape [seq_length]
        h_mmse_flat = torch.stack([self.mmse_module.forward(x) for x in h_flat], dim=0)  # [N, seq_length]
        h_mmse = h_mmse_flat.reshape(batch_size, num_rx_ant, total_ports, seq_length)
        return h_mmse

