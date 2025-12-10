"""
Configuration parser with hyperparameter search space support.

Supports:
1. Single configuration (backward compatible)
2. Grid search with discrete values
3. Range-based search
4. Mixed mode (fixed + search parameters)
"""

from itertools import product
from typing import Dict, List, Any, Union
import numpy as np


def parse_search_space_value(value: Any, param_name: str = None) -> List[Any]:
    """
    Parse a single search space parameter value
    
    Args:
        value: Can be:
              - Single value: 64 -> [64]
              - List: [32, 64, 128] -> [32, 64, 128]
              - Dict with type: {type: 'range', min: 2, max: 5} -> [2, 3, 4, 5]
        param_name: Parameter name (for error messages)
    
    Returns:
        List of values to search
    
    Examples:
        >>> parse_search_space_value(64)
        [64]
        >>> parse_search_space_value([32, 64, 128])
        [32, 64, 128]
        >>> parse_search_space_value({'type': 'range', 'min': 2, 'max': 4})
        [2, 3, 4]
    """
    if isinstance(value, list):
        # Discrete values: [32, 64, 128]
        return value
    
    elif isinstance(value, dict):
        # Range-based search
        search_type = value.get('type')
        
        if search_type == 'choice':
            # Explicit choice: {type: 'choice', values: [32, 64, 128]}
            return value['values']
        
        elif search_type == 'range':
            # Integer range: {type: 'range', min: 2, max: 5, step: 1}
            min_val = value['min']
            max_val = value['max']
            step = value.get('step', 1)
            return list(range(min_val, max_val + 1, step))
        
        elif search_type == 'uniform':
            # Uniform sampling: {type: 'uniform', min: 0.001, max: 0.1, num_samples: 5}
            min_val = value['min']
            max_val = value['max']
            num_samples = value.get('num_samples', 5)
            return list(np.linspace(min_val, max_val, num_samples))
        
        elif search_type == 'loguniform':
            # Log-uniform sampling: {type: 'loguniform', min: 0.001, max: 0.1, num_samples: 5}
            min_val = value['min']
            max_val = value['max']
            num_samples = value.get('num_samples', 5)
            return list(np.logspace(np.log10(min_val), np.log10(max_val), num_samples))
        
        else:
            raise ValueError(
                f"Unknown search type '{search_type}' for parameter '{param_name}'. "
                f"Valid types: 'choice', 'range', 'uniform', 'loguniform'"
            )
    
    else:
        # Single value (not a search parameter)
        return [value]


def expand_search_space(search_space: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand search space into all combinations (Cartesian product)
    
    Args:
        search_space: Dictionary of parameter names to search values
    
    Returns:
        List of configuration dictionaries (one per combination)
    
    Examples:
        >>> search_space = {
        ...     'hidden_dim': [32, 64],
        ...     'num_stages': [2, 3]
        ... }
        >>> configs = expand_search_space(search_space)
        >>> len(configs)
        4
    """
    if not search_space:
        return [{}]
    
    # Parse each parameter's values
    param_names = []
    param_values = []
    
    for param_name, value in search_space.items():
        values = parse_search_space_value(value, param_name)
        param_names.append(param_name)
        param_values.append(values)
    
    # Generate Cartesian product
    configs = []
    for combination in product(*param_values):
        config = dict(zip(param_names, combination))
        configs.append(config)
    
    return configs


def parse_model_config(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse model configuration, supporting both single config and search space
    
    Args:
        config: Model configuration dictionary
    
    Returns:
        List of configurations (one for single config, multiple for search space)
    
    Configuration formats:
    
    1. Single configuration (backward compatible):
        {
            'model_type': 'separator1',
            'hidden_dim': 64,
            'num_stages': 3
        }
        -> Returns: [same config]
    
    2. Search space only:
        {
            'model_type': 'separator1',
            'search_space': {
                'hidden_dim': [32, 64, 128],
                'num_stages': [2, 3]
            }
        }
        -> Returns: 6 configs (3 x 2)
    
    3. Fixed + Search (recommended):
        {
            'model_type': 'separator1',
            'fixed_params': {
                'mlp_depth': 3
            },
            'search_space': {
                'hidden_dim': [32, 64, 128],
                'num_stages': [2, 3]
            }
        }
        -> Returns: 6 configs with mlp_depth=3
    """
    # Check if this is a search space configuration
    has_search_space = 'search_space' in config
    
    if not has_search_space:
        # Mode 1: Single configuration (backward compatible)
        # Return as-is
        return [config]
    
    # Mode 2/3: Search space configuration
    model_type = config.get('model_type')
    if not model_type:
        raise ValueError("model_type is required in configuration")
    
    # Get fixed parameters (if any)
    fixed_params = config.get('fixed_params', {})
    
    # Get search space
    search_space = config.get('search_space', {})
    
    # Expand search space
    search_configs = expand_search_space(search_space)
    
    # Merge with fixed params and model_type
    final_configs = []
    for search_config in search_configs:
        final_config = {
            'model_type': model_type,
            **fixed_params,  # Fixed parameters (not searched)
            **search_config  # Search parameters (this combination)
        }
        final_configs.append(final_config)
    
    return final_configs


def generate_config_name(config: Dict[str, Any], base_name: str = None) -> str:
    """
    Generate a descriptive name for a configuration
    
    Args:
        config: Configuration dictionary
        base_name: Base name (e.g., 'separator1_search')
    
    Returns:
        Descriptive name like 'separator1_hd64_s3'
    
    Examples:
        >>> config = {'model_type': 'separator1', 'hidden_dim': 64, 'num_stages': 3}
        >>> generate_config_name(config)
        'separator1_hd64_stages3'
    """
    if base_name:
        name_parts = [base_name]
    else:
        name_parts = [config.get('model_type', 'model')]
    
    # Key parameters to include in name (only the important ones)
    key_params = {
        'hidden_dim': 'hd',
        'num_stages': 'stages',
        'mlp_depth': 'depth',
        'share_weights_across_stages': 'share',
        'activation_type': 'act',
        'num_ports': 'ports',
    }
    
    # Only add key parameters to name
    for key, abbrev in key_params.items():
        if key in config:
            value = config[key]
            
            # Format value
            if isinstance(value, bool):
                value_str = '1' if value else '0'
            elif isinstance(value, float):
                value_str = f"{value:.4f}".rstrip('0').rstrip('.')
            else:
                value_str = str(value)
            
            name_parts.append(f"{abbrev}{value_str}")
    
    return '_'.join(name_parts)


def print_search_space_summary(configs: List[Dict[str, Any]], config_name: str = None):
    """
    Print a summary of the search space
    
    Args:
        configs: List of generated configurations
        config_name: Name of the search space configuration
    """
    if len(configs) == 1:
        print(f"📋 Single configuration")
        if config_name:
            print(f"   Name: {config_name}")
        print(f"   Parameters: {configs[0]}")
        return
    
    print(f"🔍 Search space: {len(configs)} configurations")
    if config_name:
        print(f"   Name: {config_name}")
    
    # Identify search parameters (parameters that vary)
    first_config = configs[0]
    search_params = {}
    fixed_params = {}
    
    for key in first_config.keys():
        if key == 'model_type':
            continue
        
        values = [cfg.get(key) for cfg in configs]
        unique_values = list(set(values))
        
        if len(unique_values) > 1:
            search_params[key] = unique_values
        else:
            fixed_params[key] = unique_values[0]
    
    print(f"\n   Fixed parameters:")
    for key, value in fixed_params.items():
        print(f"     {key}: {value}")
    
    print(f"\n   Search parameters:")
    for key, values in search_params.items():
        print(f"     {key}: {values} ({len(values)} values)")
    
    print(f"\n   Total combinations: {len(configs)}")


# Convenience function for training scripts
def load_and_parse_config(config_dict: Dict[str, Any], verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Load and parse a configuration with optional verbose output
    
    Args:
        config_dict: Configuration dictionary
        verbose: Print summary if True
    
    Returns:
        List of parsed configurations
    """
    configs = parse_model_config(config_dict)
    
    if verbose:
        print_search_space_summary(configs)
    
    return configs


__all__ = [
    'parse_model_config',
    'parse_search_space_value',
    'expand_search_space',
    'generate_config_name',
    'print_search_space_summary',
    'load_and_parse_config'
]
