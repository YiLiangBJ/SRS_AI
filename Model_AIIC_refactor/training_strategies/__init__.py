"""Training-strategy registry and factory helpers."""

from __future__ import annotations

from typing import Mapping

from .base import BaseTrainingStrategy
from .standard_supervised import StandardSupervisedStrategy


TRAINING_STRATEGY_REGISTRY = {
    StandardSupervisedStrategy.strategy_type: StandardSupervisedStrategy,
}


def register_training_strategy(name: str, strategy_class):
    if not issubclass(strategy_class, BaseTrainingStrategy):
        raise ValueError(f"{strategy_class.__name__} must inherit from BaseTrainingStrategy")
    TRAINING_STRATEGY_REGISTRY[name] = strategy_class


def get_training_strategy_class(strategy_type: str):
    if strategy_type not in TRAINING_STRATEGY_REGISTRY:
        available = ', '.join(sorted(TRAINING_STRATEGY_REGISTRY.keys()))
        raise ValueError(
            f"Unknown training strategy type '{strategy_type}'. Available strategies: {available}"
        )
    return TRAINING_STRATEGY_REGISTRY[strategy_type]


def create_training_strategy(spec: Mapping[str, object]):
    strategy_type = dict(spec or {}).get('type')
    strategy_class = get_training_strategy_class(strategy_type)
    return strategy_class(spec)


__all__ = [
    'BaseTrainingStrategy',
    'StandardSupervisedStrategy',
    'create_training_strategy',
    'get_training_strategy_class',
    'register_training_strategy',
]