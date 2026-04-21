"""Base task definitions for the component-based experiment platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Mapping, Optional, Sequence

import torch


class BaseTask(ABC):
    """Abstract task adapter for model compilation, training, evaluation, and export."""

    task_type = 'base'

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = deepcopy(dict(spec or {}))
        self.type = self.spec.get('type', self.task_type)
        self.params = deepcopy(self.spec.get('params', {}))

    @classmethod
    @abstractmethod
    def compile_model_spec(cls, task_spec: Mapping[str, Any], model_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Compile one model recipe into runtime model spec using task context."""

    @abstractmethod
    def build_dummy_input(self, model_spec: Mapping[str, Any], batch_size: int = 1) -> torch.Tensor:
        """Create representative model input for export or smoke checks."""

    @abstractmethod
    def get_training_run_context(self, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Return task-specific kwargs consumed by the active training strategy."""

    @abstractmethod
    def evaluate_with_trainer(self, trainer, model_spec: Mapping[str, Any], training_spec: Mapping[str, Any]) -> Dict[str, Any]:
        """Run final evaluation after one training run using the active trainer."""

    @abstractmethod
    def evaluate_at_snr(
        self,
        model,
        model_spec: Mapping[str, Any],
        snr_db: float,
        tdl_config: str,
        num_batches: int = 100,
        batch_size: int = 2048,
        device: str | torch.device = 'cpu',
        use_amp: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate one trained model at a single SNR/TDL point."""

    def get_default_tdl_list(self) -> Sequence[str]:
        """Return default TDL values used by this task for evaluation sweeps."""
        tdl_config = self.params.get('tdl_config')
        if tdl_config is None:
            return []
        if isinstance(tdl_config, list):
            return list(tdl_config)
        return [tdl_config]

    def get_default_eval_batch_size(self) -> int:
        """Return the final single-run evaluation batch size."""
        return int(self.params.get('final_eval_batch_size', 200))

    def get_default_eval_num_batches(self) -> int:
        """Return the default number of evaluation batches for post-training checks."""
        return int(self.params.get('final_eval_num_batches', 1))

    @staticmethod
    def resolve_device(device: str | torch.device) -> torch.device:
        """Normalize device selection and set CUDA device when needed."""
        if isinstance(device, torch.device):
            resolved = device
        elif device == 'auto':
            resolved = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            resolved = torch.device(device)

        if resolved.type == 'cuda':
            torch.cuda.set_device(resolved if resolved.index is not None else 0)
        return resolved