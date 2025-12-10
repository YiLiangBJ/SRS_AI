"""
Device management utilities.
"""

import torch


def get_device(device_arg: str = 'auto') -> torch.device:
    """
    Get torch device based on argument
    
    Args:
        device_arg: 'auto', 'cpu', 'cuda', or specific device like 'cuda:0'
    
    Returns:
        device: torch.device object
    
    Example:
        >>> device = get_device('auto')
        >>> print(device)
        cuda:0  # if GPU available, else cpu
    """
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_arg)


def print_device_info(device: torch.device = None):
    """
    Print device information
    
    Args:
        device: torch device (if None, auto-detect)
    """
    if device is None:
        device = get_device('auto')
    
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"  CUDA available: Yes")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Current device: {torch.cuda.current_device()}")
        print(f"  Device name: {torch.cuda.get_device_name(device)}")
        
        # Memory info
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            print(f"  Memory allocated: {mem_allocated:.2f} GB")
            print(f"  Memory reserved: {mem_reserved:.2f} GB")
    else:
        print(f"  CUDA available: No")
        print(f"  Using CPU")


__all__ = ['get_device', 'print_device_info']
