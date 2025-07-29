import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt

class TrainableMMSEModule(nn.Module):    
    def mmse_chunked_filter(self, h: torch.Tensor) -> torch.Tensor:
        """
        按chunk分块，逐块做MMSE处理，最后拼接输出。
        h: [batch_size, num_rx_ant, total_ports, seq_length]，seq_length可大于mmse_block_size
        返回: [batch_size, num_rx_ant, total_ports, seq_length]
        """
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        block_size = self.mmse_block_size
        num_chunks = (seq_length + block_size - 1) // block_size
        out = []
        for i in range(num_chunks):
            start = i * block_size
            end = min((i+1)*block_size, seq_length)
            chunk = h[..., start:end]  # [batch_size, num_rx_ant, total_ports, chunk_len]
            # 如果chunk不足block_size，补零
            if chunk.shape[-1] < block_size:
                pad = torch.zeros(*chunk.shape[:-1], block_size-chunk.shape[-1], dtype=chunk.dtype, device=chunk.device)
                chunk = torch.cat([chunk, pad], dim=-1)
            # MMSE处理
            chunk_mmse = self.forward_batch_ports(chunk)  # [batch_size, num_rx_ant, total_ports, block_size]
            # 去掉补零部分
            chunk_mmse = chunk_mmse[..., :end-start]
            out.append(chunk_mmse)
        # 拼接所有chunk
        return torch.cat(out, dim=-1)

    def forward_batch_ports(self, h: torch.Tensor) -> torch.Tensor:
        """
        并行端口MMSE滤波接口，输入h: [batch_size, num_rx_ant, total_ports, mmse_block_size]
        返回: [batch_size, num_rx_ant, total_ports, mmse_block_size]
        """
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        assert seq_length == self.mmse_block_size, f"输入序列长度必须为mmse_block_size={self.mmse_block_size}"
        h_flat = h.reshape(-1, seq_length)  # [batch_size*num_rx_ant*total_ports, mmse_block_size]
        # 逐个处理每个端口的chunk
        h_mmse_flat = torch.stack([self.forward(x)[0] for x in h_flat], dim=0)  # [N, mmse_block_size]
        h_mmse = h_mmse_flat.reshape(batch_size, num_rx_ant, total_ports, seq_length)
        return h_mmse

    def __init__(self, mmse_block_size: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.mmse_block_size = mmse_block_size
        n = mmse_block_size
        diag_size = n
        off_diag_size = n * (n - 1) // 2
        real_params = diag_size + off_diag_size
        imag_params = off_diag_size
        c_matrix_size = real_params + imag_params
        r_matrix_size = real_params + imag_params
        mlp_input_dim = n * 2  # 只输入chunk的实部和虚部
        self.C_factor_generator = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, c_matrix_size)
        )
        self.R_factor_generator = nn.Sequential(
            nn.Linear(mlp_input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, r_matrix_size)
        )
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
    
    def forward(self, channel_stats: torch.Tensor, noise_power: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        device = channel_stats.device
        seq_length = channel_stats.shape[0]
        block_size = self.mmse_block_size
        num_chunks = seq_length // block_size
        # 直接分块，无需补零
        chunks = channel_stats.view(num_chunks, block_size)  # [num_chunks, block_size]
        chunks_real = torch.cat([torch.real(chunks), torch.imag(chunks)], dim=1)  # [num_chunks, block_size*2]
        C_factors = self.C_factor_generator(chunks_real)  # [num_chunks, c_matrix_size]
        R_factors = self.R_factor_generator(chunks_real)  # [num_chunks, r_matrix_size]
        C_matrices = [self._construct_matrix_from_cholesky_params(C_factors[i], block_size) for i in range(num_chunks)]
        R_matrices = [self._construct_matrix_from_cholesky_params(R_factors[i], block_size) for i in range(num_chunks)]
        return C_matrices, R_matrices
    
    def _process_chunk_with_features(self, chunk: torch.Tensor, cnn_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a chunk along with CNN features through MLP.
        
        Args:
            chunk: Channel chunk [mmse_block_size] (complex)
            cnn_features: CNN extracted features [mmse_block_size] (complex)
            
        Returns:
            Tuple of (C_factors, R_factors) for Cholesky decomposition
        """
        # Combine chunk and CNN features
        # Convert complex tensors to real representation
        chunk_real = torch.cat([torch.real(chunk), torch.imag(chunk)])  # [mmse_block_size * 2]
        features_real = torch.cat([torch.real(cnn_features), torch.imag(cnn_features)])  # [mmse_block_size * 2]
        
        # Concatenate: [mmse_block_size * 4]
        mlp_input = torch.cat([chunk_real, features_real])
        
        # Add batch dimension for linear layers
        if mlp_input.dim() == 1:
            mlp_input = mlp_input.unsqueeze(0)  # [1, mmse_block_size * 4]
        
        # Generate Cholesky factors
        C_factors = self.C_factor_generator(mlp_input).squeeze(0)  # Remove batch dimension
        R_factors = self.R_factor_generator(mlp_input).squeeze(0)  # Remove batch dimension
        
        return C_factors, R_factors
    
    def _construct_matrix_from_cholesky_params(self, factors: torch.Tensor, matrix_size: int) -> torch.Tensor:
        n = matrix_size
        device = factors.device
        
        # Calculate parameter counts
        diag_size = n  # 对角线元素数量
        off_diag_size = n * (n - 1) // 2  # 严格下三角元素数量
        real_size = diag_size + off_diag_size
        imag_size = off_diag_size
        
        # Split factors into real and imaginary parts
        L_real_flat = factors[:real_size]
        L_imag_flat = factors[real_size:real_size + imag_size]
        
        # Create zero matrices
        L_real = torch.zeros((n, n), device=device)
        L_imag = torch.zeros((n, n), device=device)
        
    # ...existing code...
    def forward_chunked(self, channel_stats: torch.Tensor, chunk_size: int = 12) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        # Ensure chunk_size matches our mmse_block_size for consistency
        if chunk_size != self.mmse_block_size:
            print(f"Warning: chunk_size ({chunk_size}) != mmse_block_size ({self.mmse_block_size}). Using mmse_block_size.")
        
        # Use the new forward method that generates unique matrices for each chunk
        return self.forward(channel_stats)
    
    def forward_batch_ports(self, h: torch.Tensor) -> torch.Tensor:
        batch_size, num_rx_ant, total_ports, seq_length = h.shape
        # 以每个端口独立处理为例，支持端口并行
        # 这里假设每个端口都用同样的MMSE结构（如MLP/CNN等），如需区分可扩展
        h_flat = h.reshape(-1, seq_length)  # [batch_size*num_rx_ant*total_ports, seq_length]
        # 这里调用原有的forward_chunked或forward方法，假设其支持batch输入
        # 如果原forward不支持batch，则需扩展为支持batch
        # 这里以MLP为例，假设self.mmse_filter支持(batch, seq_length)输入
        if hasattr(self, 'mmse_filter'):
            # 这里需要定义chunk，或者根据实际需求修改
            # 由于chunk未定义，建议将此分支实现完善或移除
            raise NotImplementedError("mmse_filter 分支未实现，请完善 chunk 的定义或移除此分支。")
        else:
            # 兼容原有forward逻辑，逐个处理（但推荐改为tensor化）
            h_mmse_flat = torch.stack([self.forward(x)[0][0] for x in h_flat], dim=0)
        h_mmse = h_mmse_flat.reshape(batch_size, num_rx_ant, total_ports, seq_length)
        return h_mmse
