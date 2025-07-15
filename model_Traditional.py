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
        ls_estimates_dict: Dict[Tuple[int, int], torch.Tensor],
        user_config,  # SRS配置对象
        noise_powers: Optional[torch.Tensor] = None,
        delay_search_range: Tuple[int, int] = (-10, 10)
    ) -> Dict[Tuple[int, int], torch.Tensor]:
        """
        Forward pass through the channel estimation process - 原生支持多用户多端口多天线批处理
        包含完整的残差计算逻辑，在所有用户/端口处理后进行全局残差计算
        
        Args:
            ls_estimates_dict: LS信道估计字典 Dict[(user_id, port_id), torch.Tensor]
                             每个张量形状: [batch_size, num_rx_ant, seq_length] 
            user_config: SRS配置对象，包含用户数、端口数、循环移位等信息
            noise_powers: 估计的噪声功率 [batch_size] (如果为None则从ls_estimate估计)
            delay_search_range: 延迟搜索范围 (min_offset, max_offset)
            
        Returns:
            Dict[(user_id, port_id), torch.Tensor]: 每个用户端口的信道估计
            每个张量形状: [batch_size, num_rx_ant, seq_length]
        """
        # 从用户配置获取循环移位信息
        cyclic_shifts = user_config.cyclic_shifts
        
        # 第一阶段：预处理所有用户/端口，计算基本信道估计和时序偏移
        preprocessed_data = {}
        batch_size = None
        
        for (user_id, port_id), ls_estimates in ls_estimates_dict.items():
            # 验证输入形状: [batch_size, num_rx_ant, seq_length]
            current_batch_size, num_rx_ant, seq_length = ls_estimates.shape
            if batch_size is None:
                batch_size = current_batch_size
            assert current_batch_size == batch_size, "All LS estimates must have same batch size"
            assert seq_length == self.seq_length, f"Expected sequence length {self.seq_length}, got {seq_length}"
            
            # 获取该端口的循环移位
            n_u_p = cyclic_shifts[user_id][port_id]
            
            # 计算理想峰值位置
            ideal_peak = (self.K - n_u_p) % self.K * self.seq_length // self.K
            
            # 转换到时域 [batch_size, num_rx_ant, seq_length]
            h_time = torch.fft.ifft(ls_estimates, dim=-1)
            
            # 估计时间偏移 - 先对所有天线求和再估计timing
            with torch.no_grad():
                # 计算所有天线的功率和 [batch_size, seq_length]
                h_power_sum = torch.sum(torch.abs(h_time)**2, dim=1)
                
                # 创建理想峰值位置张量 [batch_size]
                ideal_peaks = torch.full((batch_size,), ideal_peak, device=self.device, dtype=torch.int)
                
                # 批处理估计timing offset [batch_size]
                timing_offsets = self._estimate_timing_offset_batch(
                    h_power_sum, ideal_peaks, delay_search_range
                )
                
                # 计算循环移位量 [batch_size]
                m_values = timing_offsets + ideal_peak
                
                # 生成相位因子 [batch_size, seq_length]
                phasors_m = self._generate_phasor_batch(m_values)
                phasors_T = self._generate_phasor_batch(timing_offsets)
                
                # 移位信道以将峰值对齐到位置0 [batch_size, num_rx_ant, seq_length]
                h_shifted = ls_estimates * phasors_m.unsqueeze(1)  # 广播到天线维度
                
                # 应用OCC去复用 [batch_size, num_rx_ant, seq_length//Locc]
                Locc = self._compute_locc([[n_u_p]])
                h_avg = self._apply_occ_demux_batch(h_shifted, Locc)
                
                # 线性插值回完整长度 [batch_size, num_rx_ant, seq_length]
                h_interpolated = self._linear_interpolation_batch(h_avg, self.seq_length)
                
                # 保存预处理结果
                preprocessed_data[(user_id, port_id)] = {
                    'ls_estimates': ls_estimates,
                    'h_interpolated': h_interpolated,
                    'timing_offsets': timing_offsets,
                    'phasors_m': phasors_m,
                    'phasors_T': phasors_T,
                    'n_u_p': n_u_p
                }
        
        # 第二阶段：计算全局残差
        # 重建所有用户的信号（包含时延信息）并求和得到全局残差
        total_reconstructed_signal = torch.zeros(batch_size, num_rx_ant, self.seq_length, 
                                               dtype=torch.complex64, device=self.device)
        
        for (user_id, port_id), data in preprocessed_data.items():
            h_interpolated = data['h_interpolated']
            timing_offsets = data['timing_offsets']
            phasors_T = data['phasors_T']
            phasors_m = data['phasors_m']
            # 恢复时延信息 (加回timing offset)
            h_with_timing = h_interpolated * torch.conj(phasors_m).unsqueeze(1)
            
            # 累加到总重建信号
            total_reconstructed_signal += h_with_timing
        
        # 计算全局残差：原始LS总和 - 重建信号总和
        global_residual = ls_estimates_dict[(0,0)] - total_reconstructed_signal
        
        # 第三阶段：应用残差校正、MMSE滤波并生成最终估计
        channel_estimates_dict = {}
        
        for (user_id, port_id), data in preprocessed_data.items():
            h_interpolated = data['h_interpolated']
            timing_offsets = data['timing_offsets']
            phasors_T = data['phasors_T']
            phasors_m = data['phasors_m']
            ls_estimates = data['ls_estimates']
            
            # 将全局残差添加到当前用户的插值信道估计
            h_with_residual = h_interpolated + global_residual * phasors_m.unsqueeze(1)
            
            # 保存用于MMSE的信道信息
            self.current_h_with_residual_phasors[(user_id, port_id)] = h_with_residual
            
            # 如果存在MMSE模块，使用它生成MMSE矩阵 (暂时处理第一个样本)
            if self.mmse_module is not None:
                C, R = self.mmse_module(h_with_residual[0, 0, :])  # 使用第一个样本第一个天线
                self.set_mmse_matrices(C=C, R=R, user_port=(user_id, port_id))
            
            # 批处理估计噪声功率
            if noise_powers is None:
                noise_powers_estimated = self._estimate_noise_power_batch(ls_estimates)
            else:
                noise_powers_estimated = noise_powers
            
            # 批处理应用MMSE滤波到对齐后的信道
            h_mmse_aligned = self._apply_mmse_filter_batch(
                h_with_residual, 
                noise_powers_estimated,
                user_port=(user_id, port_id)
            )
            
            # 恢复时延信息作为最终输出 (乘以conj(phasors_T))
            final_estimates = h_mmse_aligned * torch.conj(phasors_T).unsqueeze(1)
            
            channel_estimates_dict[(user_id, port_id)] = final_estimates
        
        return channel_estimates_dict
    
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
    
    def _apply_mmse_filter_batch(self, h: torch.Tensor, noise_powers: torch.Tensor, user_port: Tuple[int, int] = None) -> torch.Tensor:
        """
        批处理应用MMSE滤波 - 只支持 MLP 方法
        
        传统的 C/R 矩阵计算方法已废弃。现在只使用基于 MLP 的 TrainableMMSEModule。
        如果没有提供 mmse_module，将返回原始输入不做任何滤波。
        
        Args:
            h: 输入信道估计 [batch_size, num_rx_ant, seq_length]
            noise_powers: 噪声功率 [batch_size]
            user_port: 用户端口标识
            
        Returns:
            h_mmse: MMSE滤波后的信道估计 [batch_size, num_rx_ant, seq_length]
        """
        if self.mmse_module is None:
            # 如果没有 MLP MMSE 模块，直接返回原始输入
            print("WARNING: No trainable MMSE module provided. Returning unfiltered channel estimates.")
            return h
            
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