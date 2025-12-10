"""
Loss functions for channel separator training.

Supports multiple loss types:
- NMSE: Normalized Mean Square Error
- Weighted: SNR-weighted loss
- Log: Logarithmic loss
- Normalized: Per-sample normalized loss
"""

import torch
import torch.nn.functional as F


def nmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Normalized Mean Square Error (NMSE)
    
    NMSE = MSE / target_power
    
    Args:
        pred: (B, P, L*2) or (B, P, L) predicted channels
        target: (B, P, L*2) or (B, P, L) target channels
    
    Returns:
        loss: scalar tensor
    """
    mse = (pred - target).pow(2).mean()
    target_power = target.pow(2).mean()
    return mse / (target_power + 1e-10)


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    SNR-weighted loss
    
    Weights loss inversely proportional to SNR:
    - Low SNR: higher weight (more important to learn)
    - High SNR: lower weight (easier cases)
    
    weight = 1 / (1 + SNR_linear)
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: SNR in dB for this batch
    
    Returns:
        loss: scalar tensor
    """
    mse = (pred - target).pow(2).mean()
    
    # Convert SNR from dB to linear
    snr_linear = 10 ** (snr_db / 10)
    
    # Weight: inverse of SNR (emphasize low SNR cases)
    weight = 1.0 / (1.0 + snr_linear)
    
    return mse * weight


def log_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic loss
    
    loss = log10(MSE + eps)
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
    
    Returns:
        loss: scalar tensor
    """
    mse = (pred - target).pow(2).mean()
    return torch.log10(mse + 1e-10)


def normalized_loss(pred: torch.Tensor, target: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Normalized loss (per-sample normalization)
    
    Normalizes by both target power and SNR
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: SNR in dB
    
    Returns:
        loss: scalar tensor
    """
    mse = (pred - target).pow(2).mean()
    target_power = target.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    
    # Normalize by both target power and SNR
    return mse / (target_power * snr_linear + 1e-10)


def calculate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    snr_db: float,
    loss_type: str = 'nmse'
) -> torch.Tensor:
    """
    Calculate loss based on specified type
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: SNR in dB for this batch
        loss_type: Loss function type
                  - 'nmse': Normalized MSE
                  - 'weighted': SNR-weighted
                  - 'log': Logarithmic
                  - 'normalized': Per-sample normalized
    
    Returns:
        loss: scalar tensor
    
    Raises:
        ValueError: If loss_type is unknown
    """
    if loss_type == 'nmse':
        return nmse_loss(pred, target)
    elif loss_type == 'weighted':
        return weighted_loss(pred, target, snr_db)
    elif loss_type == 'log':
        return log_loss(pred, target)
    elif loss_type == 'normalized':
        return normalized_loss(pred, target, snr_db)
    else:
        raise ValueError(
            f"Unknown loss_type: '{loss_type}'. "
            f"Valid options: 'nmse', 'weighted', 'log', 'normalized'"
        )


# Export all loss functions
__all__ = [
    'nmse_loss',
    'weighted_loss',
    'log_loss',
    'normalized_loss',
    'calculate_loss'
]
