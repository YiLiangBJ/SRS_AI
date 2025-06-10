import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class TrainableMMSEPerUserModule(nn.Module):    
    """
    Trainable MMSE Filter Module using Cholesky factor construction
    with separate matrix generation for each user and port
    
    This module can be used to train the C and R matrices for MMSE filtering,
    using Cholesky factor construction to ensure positive definiteness without post-processing.
    Each user and port gets its own C and R matrices.
    """
    def __init__(self, 
                seq_length: int, 
                num_users: int,
                ports_per_user: List[int],
                mmse_block_size: int = 12, 
                hidden_dim: int = 64, 
                use_complex_input: bool = False):
        """
        Initialize the trainable MMSE module with Cholesky factor construction
        
        Args:
            seq_length: Length of sequence (L)
            num_users: Number of users in the system
            ports_per_user: List of the number of ports for each user
            mmse_block_size: Size of blocks for MMSE filtering
            hidden_dim: Hidden dimension for the neural network
            use_complex_input: Whether to use complex input (real + imaginary parts)
        """
        super().__init__()
        self.seq_length = seq_length
        self.mmse_block_size = mmse_block_size
        self.use_complex_input = use_complex_input
        self.num_users = num_users
        self.ports_per_user = ports_per_user
        
        # Input dimension depends on whether we use complex input
        input_dim = seq_length * 2 if use_complex_input else seq_length
        
        # Calculate parameter counts for Cholesky factors (lower triangular matrices)
        n = mmse_block_size
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

        # 为每个用户的每个端口创建单独的网络
        self.C_generators = nn.ModuleDict()
        self.R_generators = nn.ModuleDict()

        for u in range(num_users):
            for p in range(ports_per_user[u]):
                key = f"user{u}_port{p}"
                
                # 为该用户和端口创建C矩阵生成器
                self.C_generators[key] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, c_matrix_size)
                )
                
                # 为该用户和端口创建R矩阵生成器
                self.R_generators[key] = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, r_matrix_size)
                )

    def forward(self, 
                channel_stats: torch.Tensor, 
                user_idx: int, 
                port_idx: int, 
                noise_power: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:        
        """
        Generate C and R matrices for specified user and port using Cholesky factor construction
        
        Args:
            channel_stats: Channel statistics (can be frequency domain samples)
               - If use_complex_input=False: magnitude of frequency domain samples (shape [L])
               - If use_complex_input=True: concatenated real and imag parts (shape [2*L]) or complex tensor
            user_idx: User index
            port_idx: Port index for that user
            noise_power: Estimated noise power (not used directly, kept for API compatibility)
            
        Returns:
            C: Channel correlation matrix (mmse_block_size x mmse_block_size)
            R: Noise and interference correlation matrix (mmse_block_size x mmse_block_size)
        """
        # 检查用户和端口是否有效
        if user_idx >= self.num_users or port_idx >= self.ports_per_user[user_idx]:
            raise ValueError(f"Invalid user_idx ({user_idx}) or port_idx ({port_idx})")
            
        # 获取对应的生成器
        key = f"user{user_idx}_port{port_idx}"
        c_generator = self.C_generators[key]
        r_generator = self.R_generators[key]
        
        # Process input based on configuration
        if self.use_complex_input and channel_stats.is_complex():
            # Split complex input into real and imaginary parts and concatenate
            input_tensor = torch.cat([torch.real(channel_stats), torch.imag(channel_stats)])
        else:
            # Use input as-is (should be magnitude if use_complex_input=False)
            input_tensor = channel_stats
        
        n = self.mmse_block_size
        
        # 计算实部和虚部的参数数量
        diag_size = n  # 对角线元素数量
        off_diag_size = n * (n - 1) // 2  # 严格下三角元素数量
        real_size = diag_size + off_diag_size
        imag_size = off_diag_size
        
        # 生成C矩阵对应的下三角Cholesky因子L的元素
        C_flat = c_generator(input_tensor)
        
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
        R_flat = r_generator(input_tensor)
        
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
        
        return C, R
