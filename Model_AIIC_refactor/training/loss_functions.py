"""
Loss functions for channel separator training.

Supports multiple loss types:
- NMSE: Global batch NMSE
- Weighted: SNR-weighted per-sample NMSE
- Log: Mean log10(per-sample NMSE)
- Normalized: Mean per-sample NMSE
"""

from typing import Union

import torch


EPSILON = 1e-10


def _squared_magnitude(value: torch.Tensor) -> torch.Tensor:
    """Return |value|^2 for both complex tensors and real-valued tensors."""
    if torch.is_complex(value):
        return value.real.pow(2) + value.imag.pow(2)
    return value.pow(2)


def _reduction_dims(value: torch.Tensor) -> tuple[int, ...]:
    """All non-batch dimensions reduced when forming per-sample losses."""
    return tuple(range(1, value.ndim))


def _per_sample_nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute one NMSE value per batch element."""
    dims = _reduction_dims(pred)
    error_power = _squared_magnitude(pred - target).mean(dim=dims)
    target_power = _squared_magnitude(target).mean(dim=dims)
    return error_power / (target_power + EPSILON)


def _prepare_snr_tensor(
    snr_db: Union[float, torch.Tensor, list, tuple],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize scalar or per-sample SNR inputs to a batch-shaped tensor."""
    if torch.is_tensor(snr_db):
        snr_tensor = snr_db.to(device=device, dtype=dtype).reshape(-1)
    elif isinstance(snr_db, (list, tuple)):
        snr_tensor = torch.as_tensor(snr_db, device=device, dtype=dtype).reshape(-1)
    else:
        snr_tensor = torch.full((batch_size,), float(snr_db), device=device, dtype=dtype)

    if snr_tensor.numel() == 1:
        return snr_tensor.expand(batch_size)
    if snr_tensor.numel() != batch_size:
        raise ValueError(f'Expected snr_db to have 1 or {batch_size} entries, got {snr_tensor.numel()}')
    return snr_tensor


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
    error_power = _squared_magnitude(pred - target).mean()
    target_power = _squared_magnitude(target).mean()
    return error_power / (target_power + EPSILON)


def weighted_loss(pred: torch.Tensor, target: torch.Tensor, snr_db: Union[float, torch.Tensor, list, tuple]) -> torch.Tensor:
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
    per_sample_nmse = _per_sample_nmse(pred, target)
    snr_tensor = _prepare_snr_tensor(
        snr_db=snr_db,
        batch_size=per_sample_nmse.shape[0],
        device=per_sample_nmse.device,
        dtype=per_sample_nmse.dtype,
    )
    snr_linear = torch.pow(torch.full_like(snr_tensor, 10.0), snr_tensor / 10.0)
    weight = 1.0 / (1.0 + snr_linear)
    return (per_sample_nmse * weight).mean()


def log_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Mean log-NMSE loss.
    
    loss = mean(log10(per_sample_nmse + eps))
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
    
    Returns:
        loss: scalar tensor
    """
    per_sample_nmse = _per_sample_nmse(pred, target)
    return torch.log10(per_sample_nmse + EPSILON).mean()


def normalized_loss(pred: torch.Tensor, target: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Mean per-sample NMSE.
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: Unused compatibility argument kept for API stability
    
    Returns:
        loss: scalar tensor
    """
    del snr_db
    return _per_sample_nmse(pred, target).mean()


def calculate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    snr_db: Union[float, torch.Tensor, list, tuple],
    loss_type: str = 'nmse'
) -> torch.Tensor:
    """
    Calculate loss based on specified type
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: Batch SNR or per-sample SNR values
        loss_type: Loss function type
                  - 'nmse': Normalized MSE
                  - 'weighted': SNR-weighted
                  - 'log': Mean log-NMSE
                  - 'normalized': Mean per-sample NMSE
    
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
