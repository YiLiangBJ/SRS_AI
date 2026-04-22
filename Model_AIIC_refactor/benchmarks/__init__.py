"""Standalone latency benchmarking package."""

from .workflow import benchmark_latency_programmatic, build_latency_task_matrix

__all__ = [
    'benchmark_latency_programmatic',
    'build_latency_task_matrix',
]