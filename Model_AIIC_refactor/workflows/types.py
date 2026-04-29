"""Shared dataclasses for workflow orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _parse_csv_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def _parse_override_args(values: Optional[List[str]]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for raw in values or []:
        if '=' not in raw:
            raise ValueError(f'Override must use key=value form: {raw}')
        key, raw_value = raw.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f'Override key cannot be empty: {raw}')
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


@dataclass
class TrainRequest:
    """Normalized training CLI request."""

    experiment: str
    batch_size: Optional[int] = None
    num_batches: Optional[int] = None
    runs: List[str] = field(default_factory=list)
    model_overrides: Dict[str, Any] = field(default_factory=dict)
    training_overrides: Dict[str, Any] = field(default_factory=dict)
    init_checkpoint: Optional[str] = None
    device: str = 'auto'
    save_dir: str = ''
    use_amp: bool = True
    compile_model: Optional[bool] = None
    eval_after_train: bool = False
    eval_snr_range: str = '30:-3:0'
    eval_tdl: str = 'A-30,B-100,C-300'
    eval_num_batches: int = 100
    eval_batch_size: int = 2048
    plot_after_eval: bool = False
    export_onnx_after_train: bool = False
    onnx_export_selection: str = 'best'
    onnx_output_dir: Optional[str] = None
    onnx_opset: int = 13
    onnx_batch_size: int = 1
    onnx_dynamic_batch: bool = True
    onnx_validate: bool = False
    export_matlab_after_train: bool = False
    matlab_export_selection: str = 'best'
    matlab_output_dir: Optional[str] = None
    plan_only: bool = False

    @classmethod
    def from_namespace(cls, namespace):
        """Construct from argparse.Namespace."""
        payload = vars(namespace).copy()
        payload['runs'] = _parse_csv_arg(payload.get('runs'))
        payload['model_overrides'] = _parse_override_args(payload.pop('model_override', None))
        payload['training_overrides'] = _parse_override_args(payload.pop('training_override', None))
        if (payload['model_overrides'] or payload['training_overrides']) and not payload['runs']:
            raise ValueError('--model_override and --training_override require --runs to avoid mutating the full experiment sweep')
        return cls(**payload)


@dataclass
class PostprocessSummary:
    """Outputs created after training."""

    onnx_manifests: List[Dict[str, Any]] = field(default_factory=list)
    matlab_manifests: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_output_dir: Optional[Path] = None
    evaluation_results: Optional[Dict[str, Any]] = None
    evaluation_summary_path: Optional[Path] = None
    plot_output_dir: Optional[Path] = None
    generated_plots: List[Path] = field(default_factory=list)


@dataclass
class TrainingSummary:
    """Structured result of a training experiment run."""

    experiment_output_dir: Path
    experiment_name: str
    suite: Any
    device: Any
    request: TrainRequest
    plan_only: bool = False
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_duration: float = 0.0
    results: List[Dict[str, Any]] = field(default_factory=list)
    results_sorted: List[Dict[str, Any]] = field(default_factory=list)
    report_path: Optional[Path] = None
    postprocess: Optional[PostprocessSummary] = None
