"""Base training-strategy definitions for the component-based experiment platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping


class BaseTrainingStrategy(ABC):
    """Abstract training strategy adapter."""

    strategy_type = 'base'

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = deepcopy(dict(spec or {}))
        self.type = self.spec.get('type', self.strategy_type)
        self.params = deepcopy(self.spec.get('params', {}))

    @classmethod
    @abstractmethod
    def compile_runtime_spec(
        cls,
        task_spec: Mapping[str, Any],
        strategy_spec: Mapping[str, Any],
        default_training_config: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        """Compile one raw strategy recipe into runtime training spec."""

    @abstractmethod
    def create_trainer(self, model, training_spec: Mapping[str, Any], request, device, tensorboard_dir: Path):
        """Create the concrete trainer/backend used by this strategy."""

    @abstractmethod
    def run(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any], experiment_dir: Path, progress_tracker):
        """Execute the strategy-specific training loop."""

    @abstractmethod
    def final_evaluate(self, trainer, task, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Run the final post-training evaluation for one completed run."""