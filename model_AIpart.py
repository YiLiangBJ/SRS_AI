import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class TrainableMMSEModule(nn.Module):    
    """
    Trainable MMSE Filter Module using Cholesky factor construction
    
    This module can be used to train the C and R matrices for MMSE filtering,
    using Cholesky factor construction to ensure positive definiteness without post-processing.
    
    Rather than generating matrices C and R directly and then enforcing constraints,
    we parameterize their Cholesky factors (L matrices where C = L*L^H) directly,
    which guarantees that the resulting matrices are Hermitian positive definite.
    """
    def __init__(self, seq_length: int, mmse_block_size: int = 12, hidden_dim: int = 64, use_complex_input: bool = False):
        """
        Initialize the trainable MMSE module with Cholesky factor construction.
        
        This is the ONLY supported MMSE method. All traditional C/R calculation methods 
        are obsolete and have been removed in favor of this MLP-based approach.
        
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
        
        # Calculate parameter counts for Cholesky factors (lower triangular matrices)
        n = mmse_block_size
        # Diagonal elements (real part only) = n
        # Strictly lower triangular elements (real + imaginary parts) = n*(n-1)/2
        diag_size = n  # Number of diagonal elements
        off_diag_size = n * (n - 1) // 2  # Number of strictly lower triangular elements
        
        # Real parameters = diagonal elements + strictly lower triangular elements
        real_params = diag_size + off_diag_size
        # Imaginary parameters = only strictly lower triangular elements have imaginary parts
        imag_params = off_diag_size
        
        # 总参数量
        c_matrix_size = real_params + imag_params
        r_matrix_size = real_params + imag_params
          # Networks to generate Cholesky factors (L matrices) for C and R
        # Initialize with Xavier/Glorot initialization for better random initialization
        self.C_factor_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 增加神经元数量
            nn.LayerNorm(hidden_dim * 2),         # 添加批标准化
            nn.LeakyReLU(0.1),                     # 使用LeakyReLU
            nn.Dropout(0.2),                       # 添加Dropout防止过拟合
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, c_matrix_size)
        )
        
        # R矩阵的Cholesky因子生成器，也使用完整的频域样本作为输入
        self.R_factor_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),  # 增加神经元数量
            nn.LayerNorm(hidden_dim * 2),         # 添加批标准化
            nn.LeakyReLU(0.1),                     # 使用LeakyReLU
            nn.Dropout(0.2),                       # 添加Dropout防止过拟合
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, r_matrix_size)
        )
        
        # Apply Xavier initialization for better random initial values
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights with Xavier/Glorot initialization for better randomness.
        This ensures that the initial MMSE matrices have reasonable random values.
        """
        for module in [self.C_factor_generator, self.R_factor_generator]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        # Small random bias initialization
                        nn.init.uniform_(layer.bias, -0.1, 0.1)
    
    def forward(self, channel_stats: torch.Tensor, noise_power: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:        
        """
        Generate C and R matrices for specified block size using Cholesky factor construction
        
        Args:
            channel_stats: Channel statistics (can be frequency domain samples)
               - If use_complex_input=False: magnitude of frequency domain samples (shape [L])
               - If use_complex_input=True: concatenated real and imag parts (shape [2*L]) or complex tensor
            noise_power: Estimated noise power (not used directly anymore, kept for API compatibility)
            
        Returns:
            C: Channel correlation matrix (mmse_block_size x mmse_block_size)
            R: Noise and interference correlation matrix (mmse_block_size x mmse_block_size)
        """
        # Process input based on configuration
        if self.use_complex_input and channel_stats.is_complex():
            # Split complex input into real and imaginary parts and concatenate
            input_tensor = torch.cat([torch.real(channel_stats), torch.imag(channel_stats)])
        else:
            # Use input as-is (should be magnitude if use_complex_input=False)
            input_tensor = channel_stats
        
        n = self.mmse_block_size        # 使用Cholesky因子构造法确保C矩阵为正定Hermitian
        # 直接生成下三角矩阵L，然后通过C = L @ L^H构造正定Hermitian矩阵
          # 生成C矩阵对应的下三角Cholesky因子L的元素
        C_flat = self.C_factor_generator(input_tensor)
        
        # 计算实部和虚部的参数数量
        diag_size = n  # 对角线元素数量
        off_diag_size = n * (n - 1) // 2  # 严格下三角元素数量
        real_size = diag_size + off_diag_size
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
          # 对R矩阵执行相同的操作 - 生成其对应的Cholesky因子
        R_flat = self.R_factor_generator(input_tensor)
        
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
        R = L @ L.conj().transpose(0, 1)
          # 为了数值稳定性，添加一个很小的对角加载
        epsilon = 1e-6
        C = C + torch.eye(n, device=C.device) * epsilon
        R = R + torch.eye(n, device=R.device) * epsilon
        
        return C, R
    
    def forward_chunked(self, channel_stats: torch.Tensor, chunk_size: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a variable-length sequence in fixed-size chunks and concatenate the results
        
        This method handles variable sequence lengths by:
        1. Breaking the input sequence into chunks of size chunk_size (default 12)
        2. Processing each chunk independently through the MLP
        3. Concatenating the C and R matrices for each chunk
        
        Args:
            channel_stats: Channel statistics - can be any length (not restricted to self.seq_length)
            chunk_size: Size of chunks to process (default: 12)
            
        Returns:
            C: List of channel correlation matrices for each chunk
            R: List of noise correlation matrices for each chunk
        """
        # Get sequence length from input
        if self.use_complex_input and channel_stats.is_complex():
            actual_length = channel_stats.shape[0]
        else:
            actual_length = channel_stats.shape[0] // 2 if self.use_complex_input else channel_stats.shape[0]
            
        # Calculate number of chunks needed
        num_chunks = (actual_length + chunk_size - 1) // chunk_size
        
        # Lists to store C and R matrices for each chunk
        C_matrices = []
        R_matrices = []
        
        # Process each chunk
        for i in range(num_chunks):
            # Extract chunk
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, actual_length)
            
            if self.use_complex_input and channel_stats.is_complex():
                chunk = channel_stats[start_idx:end_idx]
                # Pad if necessary
                if end_idx - start_idx < chunk_size:
                    padding_size = chunk_size - (end_idx - start_idx)
                    zeros = torch.zeros(padding_size, device=chunk.device, dtype=chunk.dtype)
                    chunk = torch.cat([chunk, zeros])
            else:
                # For concatenated real/imag or real-only inputs
                if self.use_complex_input:
                    # For concatenated real/imag
                    real_start = start_idx
                    real_end = min(start_idx + chunk_size, actual_length)
                    imag_start = actual_length + start_idx
                    imag_end = min(actual_length + start_idx + chunk_size, 2 * actual_length)
                    
                    chunk_real = channel_stats[real_start:real_end]
                    chunk_imag = channel_stats[imag_start:imag_end]
                    
                    # Pad if necessary
                    if real_end - real_start < chunk_size:
                        padding_size = chunk_size - (real_end - real_start)
                        zeros = torch.zeros(padding_size, device=chunk_real.device, dtype=chunk_real.dtype)
                        chunk_real = torch.cat([chunk_real, zeros])
                        chunk_imag = torch.cat([chunk_imag, zeros])
                        
                    chunk = torch.cat([chunk_real, chunk_imag])
                else:
                    # For real-only inputs
                    chunk = channel_stats[start_idx:end_idx]
                    # Pad if necessary
                    if end_idx - start_idx < chunk_size:
                        padding_size = chunk_size - (end_idx - start_idx)
                        zeros = torch.zeros(padding_size, device=chunk.device, dtype=chunk.dtype)
                        chunk = torch.cat([chunk, zeros])
            
            # Process chunk through MLP
            C_chunk, R_chunk = self.forward(chunk)
            
            # Store matrices
            C_matrices.append(C_chunk)
            R_matrices.append(R_chunk)
        
        return C_matrices, R_matrices
