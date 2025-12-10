"""
Evaluation metrics for channel separator models.
"""

import torch
from typing import Dict, List


def calculate_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate Normalized Mean Square Error (NMSE)
    
    Args:
        pred: (B, P, L*2) or (B, P, L) predicted channels
        target: (B, P, L*2) or (B, P, L) target channels
    
    Returns:
        nmse: NMSE value (scalar)
    """
    mse = (pred - target).pow(2).mean()
    target_power = target.pow(2).mean()
    nmse = mse / (target_power + 1e-10)
    return nmse.item()


def calculate_nmse_db(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Calculate NMSE in dB
    
    Args:
        pred: Predicted channels
        target: Target channels
    
    Returns:
        nmse_db: NMSE in dB
    """
    nmse = calculate_nmse(pred, target)
    return 10 * torch.log10(torch.tensor(nmse)).item()


def calculate_per_port_nmse(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    """
    Calculate NMSE for each port separately
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
    
    Returns:
        nmse_list: List of NMSE values for each port
    """
    num_ports = pred.shape[1]
    nmse_list = []
    
    for p in range(num_ports):
        mse = (pred[:, p] - target[:, p]).pow(2).mean()
        target_power = target[:, p].pow(2).mean()
        nmse = mse / (target_power + 1e-10)
        nmse_list.append(nmse.item())
    
    return nmse_list


def calculate_per_port_nmse_db(pred: torch.Tensor, target: torch.Tensor) -> List[float]:
    """
    Calculate NMSE in dB for each port
    
    Args:
        pred: Predicted channels
        target: Target channels
    
    Returns:
        nmse_db_list: List of NMSE values in dB for each port
    """
    nmse_list = calculate_per_port_nmse(pred, target)
    return [10 * torch.log10(torch.tensor(nmse)).item() for nmse in nmse_list]


def evaluate_model(
    pred: torch.Tensor,
    target: torch.Tensor,
    snr_db: float = None
) -> Dict[str, any]:
    """
    Comprehensive evaluation of model predictions
    
    Args:
        pred: (B, P, L*2) predicted channels
        target: (B, P, L*2) target channels
        snr_db: Optional SNR in dB for context
    
    Returns:
        metrics: Dictionary with evaluation metrics
                - 'nmse': Overall NMSE
                - 'nmse_db': Overall NMSE in dB
                - 'per_port_nmse': List of per-port NMSE
                - 'per_port_nmse_db': List of per-port NMSE in dB
                - 'snr_db': Input SNR (if provided)
    """
    metrics = {
        'nmse': calculate_nmse(pred, target),
        'nmse_db': calculate_nmse_db(pred, target),
        'per_port_nmse': calculate_per_port_nmse(pred, target),
        'per_port_nmse_db': calculate_per_port_nmse_db(pred, target)
    }
    
    if snr_db is not None:
        metrics['snr_db'] = snr_db
    
    return metrics


__all__ = [
    'calculate_nmse',
    'calculate_nmse_db',
    'calculate_per_port_nmse',
    'calculate_per_port_nmse_db',
    'evaluate_model'
]
