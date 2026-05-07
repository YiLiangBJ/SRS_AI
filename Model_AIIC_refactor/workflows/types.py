"""Shared dataclasses for workflow orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import warnings

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


def merge_override_dicts(*override_dicts: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for override_dict in override_dicts:
        merged.update(override_dict or {})
    return merged


def parse_namespaced_overrides(values: Optional[List[str]], allowed_roots: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    allowed = {root.strip() for root in allowed_roots if root and root.strip()}
    parsed = {root: {} for root in allowed}
    for key, value in _parse_override_args(values).items():
        root, separator, path = key.partition('.')
        if not separator or not path:
            allowed_display = ', '.join(sorted(allowed))
            raise ValueError(
                f"Override '{key}' must use <scope>.<path>=value with scope in [{allowed_display}]"
            )
        if root not in allowed:
            allowed_display = ', '.join(sorted(allowed))
            raise ValueError(
                f"Unsupported override scope '{root}' in '{key}'. Expected one of [{allowed_display}]"
            )
        parsed[root][path] = value
    return parsed


def apply_namespace_overrides(namespace, override_values: Optional[List[str]], allowed_fields: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    overrides = _parse_override_args(override_values)
    if not overrides:
        return {}

    allowed = None if allowed_fields is None else {field for field in allowed_fields}
    unknown = []
    for key in overrides:
        if '.' in key:
            raise ValueError(f"Namespace override '{key}' must target a top-level argument name")
        if allowed is not None and key not in allowed:
            unknown.append(key)

    if unknown:
        allowed_display = ', '.join(sorted(allowed or []))
        raise ValueError(f"Unsupported override keys: {unknown}. Allowed keys: [{allowed_display}]")

    for key, value in overrides.items():
        setattr(namespace, key, value)
    return overrides


def warn_legacy_override_usage(option_name: str, replacement: str, values: Optional[List[str]]) -> None:
    if values:
        warnings.warn(
            f"{option_name} is deprecated; use {replacement} instead.",
            FutureWarning,
            stacklevel=3,
        )


@dataclass
class TrainRequest:
    """Normalized training CLI request."""

    experiment: str
    batch_size: Optional[int] = None
    num_batches: Optional[int] = None
    runs: List[str] = field(default_factory=list)
    task_overrides: Dict[str, Any] = field(default_factory=dict)
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
        warn_legacy_override_usage('--task_override', '--override task.<path>=value', payload.get('task_override'))
        warn_legacy_override_usage('--model_override', '--override model.<path>=value', payload.get('model_override'))
        warn_legacy_override_usage('--training_override', '--override training.<path>=value', payload.get('training_override'))
        namespaced_overrides = parse_namespaced_overrides(
            payload.pop('override', None),
            allowed_roots=('task', 'model', 'training'),
        )
        payload['task_overrides'] = merge_override_dicts(
            namespaced_overrides['task'],
            _parse_override_args(payload.pop('task_override', None)),
        )
        payload['model_overrides'] = merge_override_dicts(
            namespaced_overrides['model'],
            _parse_override_args(payload.pop('model_override', None)),
        )
        payload['training_overrides'] = merge_override_dicts(
            namespaced_overrides['training'],
            _parse_override_args(payload.pop('training_override', None)),
        )
        return cls(**payload)


@dataclass
class EvaluationRequest:
    """Normalized evaluation CLI request."""

    exp_dir: Optional[str] = None
    run_dir: Optional[str] = None
    run_dirs: Optional[str] = None
    runs: List[str] = field(default_factory=list)
    list_runs: bool = False
    snr_range: str = '30:-3:0'
    tdl: str = 'A-30,B-100,C-300'
    num_batches: int = 100
    batches_per_snr: Optional[int] = None
    batch_size: int = 2048
    device: str = 'auto'
    use_amp: bool = True
    compile: bool = True
    plot_after_eval: bool = True
    output: Optional[str] = None

    @classmethod
    def from_namespace(cls, namespace):
        """Construct from argparse.Namespace."""
        apply_namespace_overrides(namespace, getattr(namespace, 'override', None), allowed_fields=set(vars(namespace)) - {'override'})
        payload = vars(namespace).copy()
        payload.pop('override', None)
        payload['runs'] = _parse_csv_arg(payload.get('runs'))
        if payload.get('batches_per_snr') is not None:
            payload['num_batches'] = payload['batches_per_snr']
        return cls(**payload)


@dataclass
class LatencyBenchmarkRequest:
    """Normalized latency benchmark CLI request."""

    exp_dir: Optional[str] = None
    run_dir: Optional[str] = None
    run_dirs: Optional[str] = None
    runs: List[str] = field(default_factory=list)
    list_runs: bool = False
    device: str = 'cpu'
    runtime_backends: Optional[str] = None
    execution_modes: Optional[str] = None
    precision_profiles: Optional[str] = None
    batch_sizes: str = '1,2,4,8,16,32,64,128'
    batch_antennas: Optional[str] = None
    batch_rbgs: Optional[str] = None
    thread_counts: Optional[str] = None
    warmup_iters: int = 20
    measure_iters: int = 50
    output: Optional[str] = None

    @classmethod
    def from_namespace(cls, namespace):
        """Construct from argparse.Namespace."""
        apply_namespace_overrides(namespace, getattr(namespace, 'override', None), allowed_fields=set(vars(namespace)) - {'override'})
        payload = vars(namespace).copy()
        payload.pop('override', None)
        payload['runs'] = _parse_csv_arg(payload.get('runs'))
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
