import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class SRSChannelEstimator(nn.Module):
    """
    SRS Channel Estimator using AI-enhanced methods
    
    ⚠️  IMPORTANT ARCHITECTURAL CHANGE ⚠️
    
    This module has been refactored to ONLY support MLP-based MMSE filtering.
    All traditional C/R matrix calculation methods have been marked as OBSOLETE.
    
    CURRENT MMSE APPROACH:
    - Uses TrainableMMSEModule (MLP-based) for C and R matrix generation
    - C and R matrices are learned through neural networks with proper initialization
    - Traditional exponential decay and diagonal noise models are deprecated
    
    OBSOLETE METHODS (marked with warnings):
    - _get_block_C_matrix: replaced by TrainableMMSEModule
    - _get_block_R_matrix: replaced by TrainableMMSEModule  
    - _get_C_matrix: replaced by TrainableMMSEModule
    - _get_R_matrix: replaced by TrainableMMSEModule
    - _apply_mmse_filter: use _apply_mmse_filter_batch instead
    
    This module implements the complete SRS channel estimation process:
    1. LS estimation from received signals
    2. OCC demultiplexing and interpolation  
    3. Global residual calculation and application
    4. MLP-based MMSE filtering (NEW: only supported method)
    5. Timing offset estimation and recovery
    """
    def __init__(
        self,
        seq_length: int,
        ktc: int = 4,
        max_users: int = 8,
        max_ports_per_user: int = 4,
        mmse_block_size: int = 12,
        device: str = "cpu",  # Force CPU-only execution
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
        ls_estimates: List[torch.Tensor],  # 修改类型注解
        user_config,
        true_channels_dict: Optional[List[Dict[Tuple[int, int],torch.tensor]]] = None, # Debug purpose
        delay_search_range: Tuple[int, int] = (-30, 30)) -> List[Dict[Tuple[int, int], torch.Tensor]]:  # 修改返回类型

        # 从用户配置获取循环移位信息
        cyclic_shifts = user_config.cyclic_shifts
        batch_size = len(ls_estimates)
        
        # 初始化结果列表
        channel_estimates_list = []
        
        # 遍历每个batch样本
        for batch_idx in range(batch_size):
            sample_ls_estimates = ls_estimates[batch_idx]  # 当前样本的张量
            num_rx_ant, seq_length = sample_ls_estimates.shape

            sample_channel_estimates = {}  # 当前样本的结果
            
            # 第一阶段：预处理当前样本的所有用户/端口
            preprocessed_data = {}
            # 转换到时域 [num_rx_ant, seq_length]
            h_time = torch.fft.ifft(sample_ls_estimates, dim=-1)
            # 计算所有天线的功率和 [seq_length]
            h_power_sum = torch.sum(torch.abs(h_time)**2, dim=0)
            # 从用户配置中获取所有(user_id, port_id)组合
            for user_id in range(user_config.num_users):
                for port_id in range(user_config.ports_per_user[user_id]):

                    # 获取该端口的循环移位
                    n_u_p = cyclic_shifts[user_id][port_id]
                    
                    # 计算理想峰值位置
                    ideal_peak = (self.K - n_u_p) % self.K * self.seq_length // self.K

                    # 估计时间偏移
                    with torch.no_grad():
                        # 估计timing offset
                        timing_offset = self._estimate_timing_offset(
                            h_power_sum, ideal_peak, delay_search_range
                        )
                        
                        # 计算循环移位量
                        m_value = timing_offset + ideal_peak
                        
                        # 生成相位因子 [seq_length]
                        phasor_m = self._generate_phasor(m_value)
                        phasor_T = self._generate_phasor(timing_offset)
                        
                        # 移位信道以将峰值对齐到位置0 [num_rx_ant, seq_length]
                        h_shifted = sample_ls_estimates * phasor_m.unsqueeze(0)  # 广播到天线维度
                        
                        # 应用OCC去复用 [num_rx_ant, seq_length//Locc]
                        Locc = self._compute_locc([[n_u_p]])
                        h_avg = self._apply_occ_demux_single(h_shifted, Locc)
                        
                        # 线性插值回完整长度 [num_rx_ant, seq_length]
                        h_interpolated = self._linear_interpolation_single(h_avg, self.seq_length)
                        
                        # 保存预处理结果
                        preprocessed_data[(user_id, port_id)] = {
                            'h_interpolated': h_interpolated,
                            'timing_offset': timing_offset,
                            'phasor_m': phasor_m,
                            'phasor_T': phasor_T,
                            'n_u_p': n_u_p
                        }
                
            # 第二阶段：计算当前样本的全局残差
            total_reconstructed_signal = torch.zeros(num_rx_ant, self.seq_length, 
                                                dtype=torch.complex64, device=self.device)
            
            for (user_id, port_id), data in preprocessed_data.items():
                h_interpolated = data['h_interpolated']
                phasor_m = data['phasor_m']
                # 恢复时延信息
                h_with_timing = h_interpolated * torch.conj(phasor_m).unsqueeze(0)
                total_reconstructed_signal += h_with_timing
            
            # 计算全局残差
            global_residual = sample_ls_estimates - total_reconstructed_signal
            
            # 第三阶段：应用残差校正和MMSE滤波
            for (user_id, port_id), data in preprocessed_data.items():
                h_interpolated = data['h_interpolated']
                phasor_m = data['phasor_m']
                phasor_T = data['phasor_T']
                
                # 添加全局残差
                h_with_residual = h_interpolated + global_residual * phasor_m.unsqueeze(0)
                
                # MMSE滤波 - 使用能量最大的天线来计算C和R矩阵
                # 计算每个天线的总能量 [num_rx_ant]
                antenna_energies = torch.sum(torch.abs(h_with_residual)**2, dim=1)  # [num_rx_ant]
                # 找到能量最大的天线索引
                max_energy_antenna_idx = torch.argmax(antenna_energies).item()
                # 使用能量最大的天线计算C和R
                reference_antenna_signal = h_with_residual[max_energy_antenna_idx, :]  # [seq_length]
                C, R = self.mmse_module(reference_antenna_signal)
                self.set_mmse_matrices(C=C, R=R, user_port=(user_id, port_id))
                
                h_mmse_aligned = self._apply_mmse_filter_single(h_with_residual, user_port=(user_id, port_id))
                
                # 恢复时延信息
                final_estimates = h_mmse_aligned * torch.conj(phasor_T).unsqueeze(0)
                
                sample_channel_estimates[(user_id, port_id)] = final_estimates
            
            # 添加当前样本的结果到批次结果
            channel_estimates_list.append(sample_channel_estimates)
        
        return channel_estimates_list
    
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

    def _apply_mmse_filter_single(self, h: torch.Tensor, user_port: Tuple[int, int] = None) -> torch.Tensor:
        """单样本版本的MMSE滤波"""
        if self.mmse_module is None:
            return h
        
        # 添加batch维度，调用batch版本，然后移除batch维度
        h_batch = h.unsqueeze(0)
        h_filtered_batch = self._apply_mmse_filter_batch(h_batch, user_port)
        return h_filtered_batch.squeeze(0)

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
    
    def _generate_phasor_batch(self, m_values: torch.Tensor) -> torch.Tensor:
        """
        批处理生成相位因子
        
        Args:
            m_values: 移位量 [batch_size]
            
        Returns:
            phasors: 相位因子 [batch_size, seq_length]
        """
        batch_size = m_values.shape[0]
        n = torch.arange(self.seq_length, device=self.device).unsqueeze(0)  # [1, seq_length]
        m_expanded = m_values.unsqueeze(1)  # [batch_size, 1]
        
        # 广播计算 [batch_size, seq_length]
        phasors = torch.exp(1j * 2 * np.pi * m_expanded * n / self.seq_length)
        return phasors
    
    def _apply_occ_demux_batch(self, h: torch.Tensor, Locc: int) -> torch.Tensor:
        """
        批处理应用OCC去复用
        
        Args:
            h: 输入序列 [batch_size, num_rx_ant, seq_length]
            Locc: OCC长度
            
        Returns:
            h_avg: 平均后的序列 [batch_size, num_rx_ant, seq_length//Locc]
        """
        batch_size, num_rx_ant, L = h.shape
        # 重塑为 [batch_size, num_rx_ant, L//Locc, Locc]
        h_reshaped = h.reshape(batch_size, num_rx_ant, L // Locc, Locc)
        # 在最后一维求平均 [batch_size, num_rx_ant, L//Locc]
        h_avg = torch.mean(h_reshaped, dim=-1)
        return h_avg
    
    def _linear_interpolation_batch(self, h_avg: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        批处理线性插值
        
        Args:
            h_avg: 输入序列 [batch_size, num_rx_ant, reduced_length]
            target_length: 目标长度
            
        Returns:
            h_interp: 插值后的序列 [batch_size, num_rx_ant, target_length]
        """
        batch_size, num_rx_ant, reduced_length = h_avg.shape
        
        # 计算组大小
        group_size = target_length // reduced_length
        
        # 创建输入和输出索引
        orig_indices = torch.tensor([np.mean(np.arange(group_size)) + i * group_size 
                                   for i in range(reduced_length)], device=self.device)
        new_indices = torch.arange(target_length, dtype=torch.float32, device=self.device)
        
        # 批处理插值
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
        批处理估计噪声功率
        
        Args:
            h: 频域信道估计 [batch_size, num_rx_ant, seq_length]
            
        Returns:
            noise_powers: 噪声功率 [batch_size]
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

    def _estimate_timing_offset_batch(self, h_time_batch: torch.Tensor, ideal_peaks: torch.Tensor, search_range: Tuple[int, int] = (-10, 10)) -> torch.Tensor:
        """
        批处理估计时序偏移量，基于CIR峰值位置相对于理想位置的偏移量
        仅在理想峰值周围的有限范围内搜索实际峰值
        
        Args:
            h_time_batch: 时域信道脉冲响应功率，形状 [batch_size, seq_length]
            ideal_peaks: 理想峰值位置，形状 [batch_size] (每个样本的理想峰值位置)
            search_range: 搜索范围，元组 (min_offset, max_offset)
            
        Returns:
            批处理时序偏移量，形状 [batch_size]
        """
        # 计算每个样本的时序偏移量
        timing_offsets = []
        for h_time, ideal_peak in zip(h_time_batch, ideal_peaks):
            offset = self._estimate_timing_offset(h_time, ideal_peak.item(), search_range)
            timing_offsets.append(offset)
        
        return torch.tensor(timing_offsets, device=self.device)
    
    def _estimate_timing_offset(self, h_time: torch.Tensor, ideal_peak: int, search_range: Tuple[int, int] = (-10, 10)) -> int:
        """
        Estimate timing offset based on CIR peak position relative to ideal position,
        searching only within a limited range around the ideal peak (单样本版本)
        
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