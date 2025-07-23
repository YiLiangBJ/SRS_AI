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
        Initialize the hybrid CNN-MLP MMSE module.
        
        This module uses a CNN to extract global features from the full sequence,
        then processes chunks of the sequence along with these features using an MLP.
        
        Args:
            seq_length: Length of sequence (L) - will be chunked into blocks of mmse_block_size
            mmse_block_size: Size of blocks for MMSE filtering (default: 12)
            hidden_dim: Hidden dimension for the neural network
            use_complex_input: Whether to use complex input (real + imaginary parts)
        """
        super().__init__()
        self.seq_length = seq_length
        self.mmse_block_size = mmse_block_size
        self.use_complex_input = use_complex_input
        
        # Calculate number of chunks
        self.num_chunks = seq_length // mmse_block_size
        if seq_length % mmse_block_size != 0:
            self.num_chunks += 1  # Add one more chunk for remainder
        
        # CNN feature extractor: full sequence -> 12-point complex feature
        # Input: [seq_length] (complex) or [seq_length*2] (real+imag)
        # Output: [mmse_block_size] (complex) = [mmse_block_size*2] (real+imag)
        cnn_input_channels = 2 if use_complex_input else 1  # Real+Imag or magnitude only
        self.cnn_feature_extractor = nn.Sequential(
            # 1D CNN to extract global features
            nn.Conv1d(cnn_input_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Adaptive pooling to get exactly mmse_block_size output
            nn.AdaptiveAvgPool1d(mmse_block_size),
            # Final conv to get 2 channels (real + imaginary parts)
            nn.Conv1d(32, 2, kernel_size=1)
        )
        
        # MLP input dimension: 
        # - mmse_block_size chunk (complex) = mmse_block_size * 2 (real+imag)
        # - CNN features (complex) = mmse_block_size * 2 (real+imag)
        # Total = mmse_block_size * 4
        mlp_input_dim = mmse_block_size * 4
        
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
        
        # MLP networks to generate Cholesky factors
        # Input: chunk (12*2) + CNN features (12*2) = 48 dimensions
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
    
    def forward(self, channel_stats: torch.Tensor, noise_power: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:        
        """
        Forward pass of the hybrid CNN-MLP MMSE module.
        
        NEW APPROACH: Generate different C and R matrices for each chunk to preserve local characteristics.
        
        Process:
        1. Extract global features from full sequence using CNN
        2. Split sequence into chunks of mmse_block_size
        3. For each chunk, combine with CNN features and process through MLP
        4. Generate unique C and R matrices for each chunk
        
        Args:
            channel_stats: Channel statistics tensor [seq_length] (complex)
            noise_power: Estimated noise power (not used directly anymore, kept for API compatibility)
            
        Returns:
            C_matrices: List of channel correlation matrices for each chunk
            R_matrices: List of noise correlation matrices for each chunk
        """
        device = channel_stats.device
        seq_length = channel_stats.shape[0]
        
        # Step 1: Extract global features using CNN
        cnn_features = self._extract_cnn_features(channel_stats)  # [mmse_block_size] complex
        
        # Step 2: Split sequence into chunks and generate unique matrices for each
        C_matrices = []
        R_matrices = []
        
        # Calculate actual number of chunks needed for this sequence
        actual_num_chunks = (seq_length + self.mmse_block_size - 1) // self.mmse_block_size
        
        for chunk_idx in range(actual_num_chunks):
            start_idx = chunk_idx * self.mmse_block_size
            end_idx = min(start_idx + self.mmse_block_size, seq_length)
            
            # Extract chunk
            chunk = channel_stats[start_idx:end_idx]
            
            # Pad chunk if it's smaller than mmse_block_size
            if chunk.shape[0] < self.mmse_block_size:
                padding_size = self.mmse_block_size - chunk.shape[0]
                padding = torch.zeros(padding_size, dtype=chunk.dtype, device=device)
                chunk = torch.cat([chunk, padding])
            
            # Process chunk with CNN features to get unique Cholesky factors
            C_factors, R_factors = self._process_chunk_with_features(chunk, cnn_features)
            
            # Construct unique matrices for this chunk
            C_matrix = self._construct_matrix_from_cholesky_params(C_factors, self.mmse_block_size)
            R_matrix = self._construct_matrix_from_cholesky_params(R_factors, self.mmse_block_size)
            
            # Store the unique matrices
            C_matrices.append(C_matrix)
            R_matrices.append(R_matrix)
        
        return C_matrices, R_matrices
    
    def _extract_cnn_features(self, channel_stats: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from the full sequence using CNN.
        
        Args:
            channel_stats: Full sequence [seq_length] (complex)
            
        Returns:
            CNN features [mmse_block_size] (complex)
        """
        # Prepare input for CNN
        if self.use_complex_input and channel_stats.is_complex():
            # Split complex into real and imaginary parts: [2, seq_length]
            cnn_input = torch.stack([torch.real(channel_stats), torch.imag(channel_stats)], dim=0)
        else:
            # Use magnitude only: [1, seq_length]
            if channel_stats.is_complex():
                cnn_input = torch.abs(channel_stats).unsqueeze(0)
            else:
                cnn_input = channel_stats.unsqueeze(0)
        
        # Add batch dimension: [1, channels, seq_length]
        cnn_input = cnn_input.unsqueeze(0)
        
        # Extract features using CNN: [1, 2, mmse_block_size]
        cnn_output = self.cnn_feature_extractor(cnn_input)
        
        # Remove batch dimension and convert back to complex: [mmse_block_size]
        cnn_output = cnn_output.squeeze(0)  # [2, mmse_block_size]
        real_part = cnn_output[0, :]  # [mmse_block_size]
        imag_part = cnn_output[1, :]  # [mmse_block_size]
        
        return torch.complex(real_part, imag_part)
    
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
        """
        Construct Hermitian positive definite matrix from Cholesky factors.
        
        Args:
            factors: Flattened Cholesky factors
            matrix_size: Size of the output matrix (n x n)
            
        Returns:
            Hermitian positive definite matrix (matrix_size x matrix_size)
        """
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
        
        # Fill lower triangular matrix (real part)
        idx = 0
        for i in range(n):
            for j in range(i + 1):  # j <= i for lower triangular
                if i == j:  # Diagonal elements only have real part and must be positive
                    L_real[i, j] = torch.nn.functional.softplus(L_real_flat[idx])
                else:
                    L_real[i, j] = L_real_flat[idx]
                idx += 1
        
        # Fill lower triangular matrix (imaginary part, only off-diagonal)
        idx = 0
        for i in range(n):
            for j in range(i):  # j < i for strictly lower triangular
                L_imag[i, j] = L_imag_flat[idx]
                idx += 1
                
        # Combine into complex lower triangular matrix L
        L = torch.complex(L_real, L_imag)
        
        # Calculate matrix = L @ L^H (L^H is conjugate transpose of L)
        # This naturally creates a Hermitian positive definite matrix
        matrix = L @ L.conj().transpose(0, 1)
        
        # Add small diagonal loading for numerical stability
        epsilon = 1e-6
        matrix = matrix + torch.eye(n, device=device) * epsilon
        
        return matrix
    
    def forward_chunked(self, channel_stats: torch.Tensor, chunk_size: int = 12) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Process a variable-length sequence in fixed-size chunks using hybrid CNN-MLP approach.
        
        This method now generates unique C and R matrices for each chunk, preserving local characteristics.
        
        Args:
            channel_stats: Channel statistics - can be any length (not restricted to self.seq_length)
            chunk_size: Size of chunks to process (default: 12, should match mmse_block_size)
            
        Returns:
            C_matrices: List of channel correlation matrices for each chunk (each is unique)
            R_matrices: List of noise correlation matrices for each chunk (each is unique)
        """
        # Ensure chunk_size matches our mmse_block_size for consistency
        if chunk_size != self.mmse_block_size:
            print(f"Warning: chunk_size ({chunk_size}) != mmse_block_size ({self.mmse_block_size}). Using mmse_block_size.")
        
        # Use the new forward method that generates unique matrices for each chunk
        return self.forward(channel_stats)
