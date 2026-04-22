"""Standalone latency benchmarking package."""

from .plotting import generate_latency_comparison_plots
from .workflow import benchmark_latency_programmatic, build_latency_task_matrix

__all__ = [
    'generate_latency_comparison_plots',
    'benchmark_latency_programmatic',
    'build_latency_task_matrix',
]