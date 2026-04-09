"""Workflow layer for thin CLI entrypoints."""

from .types import TrainRequest, TrainingSummary, PostprocessSummary
from .reporting import generate_training_report
from .evaluation_workflow import evaluate_models_programmatic
from .export_workflow import export_run_to_onnx, export_runs_to_onnx
from .plotting_workflow import generate_plots_programmatic
from .postprocess_workflow import run_post_training_pipeline
from .train_workflow import run_training_experiment

__all__ = [
    'TrainRequest',
    'TrainingSummary',
    'PostprocessSummary',
    'generate_training_report',
    'evaluate_models_programmatic',
    'export_run_to_onnx',
    'export_runs_to_onnx',
    'generate_plots_programmatic',
    'run_post_training_pipeline',
    'run_training_experiment',
]
