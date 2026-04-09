"""Shared dataclasses for workflow orchestration."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainRequest:
    """Normalized training CLI request."""

    experiment: str
    batch_size: Optional[int] = None
    num_batches: Optional[int] = None
    device: str = 'auto'
    save_dir: str = './experiments_refactored'
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
    onnx_dynamic_batch: bool = False
    onnx_validate: bool = False
    plan_only: bool = False

    @classmethod
    def from_namespace(cls, namespace):
        """Construct from argparse.Namespace."""
        return cls(**vars(namespace))


@dataclass
class PostprocessSummary:
    """Outputs created after training."""

    onnx_manifests: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_output_dir: Optional[Path] = None
    evaluation_results: Optional[Dict[str, Any]] = None
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
