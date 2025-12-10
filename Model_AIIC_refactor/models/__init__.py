"""
Model registry and factory functions.

Usage:
    from models import create_model, list_models, register_model
    
    # List available models
    models = list_models()
    
    # Create model from config
    config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 64}
    model = create_model('separator1', config)
    
    # Register new model
    register_model('my_model', MyModelClass)
"""

from .base_model import BaseSeparatorModel
from .separator1 import Separator1
from .separator2 import Separator2

__version__ = '2.0.0'
__all__ = ['BaseSeparatorModel', 'Separator1', 'Separator2', 
           'create_model', 'list_models', 'register_model']


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL_REGISTRY = {
    'separator1': Separator1,
    'separator2': Separator2,
    # Aliases for compatibility
    'dual_path': Separator1,
    'complex_linear': Separator2,
    'type1': Separator1,
    'type2': Separator2,
}


def register_model(name: str, model_class):
    """
    Register a new model to the registry
    
    Args:
        name: Model name (used in config)
        model_class: Model class (must inherit from BaseSeparatorModel)
    
    Raises:
        ValueError: If model_class doesn't inherit from BaseSeparatorModel
    
    Example:
        >>> from models import register_model
        >>> from models.base_model import BaseSeparatorModel
        >>> 
        >>> class MyModel(BaseSeparatorModel):
        >>>     ...
        >>> 
        >>> register_model('my_model', MyModel)
    """
    if not issubclass(model_class, BaseSeparatorModel):
        raise ValueError(
            f"{model_class.__name__} must inherit from BaseSeparatorModel"
        )
    
    MODEL_REGISTRY[name] = model_class
    print(f"✓ Registered model: '{name}' -> {model_class.__name__}")


def create_model(model_name: str, config: dict):
    """
    Factory function: Create model from name and config
    
    Args:
        model_name: Model name in MODEL_REGISTRY
        config: Configuration dictionary
    
    Returns:
        model: Model instance
    
    Raises:
        ValueError: If model_name not found in registry
    
    Example:
        >>> config = {'seq_len': 12, 'num_ports': 4, 'hidden_dim': 64}
        >>> model = create_model('separator1', config)
    """
    # Backward compatibility: Convert numeric model_type to string
    if isinstance(model_name, int):
        model_name = f'separator{model_name}'
    
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )
    
    model_class = MODEL_REGISTRY[model_name]
    return model_class.from_config(config)


def list_models() -> list:
    """
    List all available models in registry
    
    Returns:
        models: List of model names
    """
    return list(MODEL_REGISTRY.keys())


def get_model_class(model_name: str):
    """
    Get model class from name
    
    Args:
        model_name: Model name
    
    Returns:
        model_class: Model class
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: '{model_name}'")
    return MODEL_REGISTRY[model_name]
