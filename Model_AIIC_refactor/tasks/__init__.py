"""Task registry and factory helpers."""

from __future__ import annotations

from typing import Mapping

from .base import BaseTask
from .channel_separator import ChannelSeparatorTask


TASK_REGISTRY = {
    ChannelSeparatorTask.task_type: ChannelSeparatorTask,
}


def register_task(name: str, task_class):
    if not issubclass(task_class, BaseTask):
        raise ValueError(f"{task_class.__name__} must inherit from BaseTask")
    TASK_REGISTRY[name] = task_class


def get_task_class(task_type: str):
    if task_type not in TASK_REGISTRY:
        available = ', '.join(sorted(TASK_REGISTRY.keys()))
        raise ValueError(f"Unknown task type '{task_type}'. Available tasks: {available}")
    return TASK_REGISTRY[task_type]


def create_task(spec: Mapping[str, object]):
    task_type = dict(spec or {}).get('type')
    task_class = get_task_class(task_type)
    return task_class(spec)


__all__ = [
    'BaseTask',
    'ChannelSeparatorTask',
    'create_task',
    'get_task_class',
    'register_task',
]