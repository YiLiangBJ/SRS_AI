"""Post-processing plots for saved latency benchmark results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import (
    max_legend_items,
    panel_figure_size,
    place_legend_outside_right,
    resolve_existing_path,
    split_csv_arg,
    style_for_series,
)


def resolve_latency_plot_input(input_path) -> tuple[Path, Path]:
    """Resolve a latency plotting target into (json_path, output_dir)."""
    resolved = resolve_existing_path(input_path)
    if isinstance(resolved, tuple):
        _, candidates = resolved
        candidate_text = '\n'.join(str(path) for path in candidates)
        raise FileNotFoundError('Latency plotting input not found. Checked:\n' + candidate_text)

    resolved = Path(resolved)
    if resolved.is_file():
        if resolved.name != 'latency_results.json':
            raise ValueError('Latency plotting input file must be latency_results.json')
        return resolved, resolved.parent / 'plots_custom'

    json_path = resolved / 'latency_results.json'
    if not json_path.exists():
        raise FileNotFoundError('Expected latency_results.json inside the provided latency directory')
    return json_path, resolved / 'plots_custom'


def _parse_optional_csv(value: Optional[str]) -> Optional[List[str]]:
    parsed = split_csv_arg(value)
    return parsed or None


def _coerce_threads(value: Optional[str]) -> Optional[List[int]]:
    parsed = _parse_optional_csv(value)
    if parsed is None:
        return None
    return [int(item) for item in parsed]


def _filter_results(
    results: List[Dict[str, Any]],
    runs: Optional[List[str]] = None,
    execution_modes: Optional[List[str]] = None,
    precision_profiles: Optional[List[str]] = None,
    thread_counts: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    filtered = [item for item in results if item.get('status') == 'ok']
    if runs is not None:
        filtered = [item for item in filtered if item['run_name'] in runs]
    if execution_modes is not None:
        filtered = [item for item in filtered if item['execution_mode'] in execution_modes]
    if precision_profiles is not None:
        filtered = [item for item in filtered if item['requested_precision_profile'] in precision_profiles]
    if thread_counts is not None:
        filtered = [item for item in filtered if int(item['num_threads']) in thread_counts]
    return filtered


def _subplot_grid(count: int) -> tuple[int, int]:
    cols = min(2, max(1, count))
    rows = math.ceil(count / cols)
    return rows, cols


def _prepare_axes(count: int):
    rows, cols = _subplot_grid(count)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 5 * rows), squeeze=False)
    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes[count:]:
        axis.set_visible(False)
    return fig, flat_axes[:count]


def _metric_value(item: Dict[str, Any], metric: str) -> float:
    if metric == 'p50_latency_ms':
        return float(item['p50_latency_ms'])
    if metric == 'throughput_samples_per_sec':
        return float(item['throughput_samples_per_sec'])
    raise ValueError(f'Unsupported metric {metric!r}')


def _metric_label(metric: str) -> str:
    if metric == 'p50_latency_ms':
        return 'P50 Latency (ms)'
    if metric == 'throughput_samples_per_sec':
        return 'Throughput (samples/s)'
    raise ValueError(f'Unsupported metric {metric!r}')


def _save_mode_panel_figure(results: List[Dict[str, Any]], output_path: Path, metric: str, thread_count: int):
    execution_modes = sorted({item['execution_mode'] for item in results})
    if not execution_modes:
        return None
    grouped_counts = []
    for execution_mode in execution_modes:
        mode_results = [item for item in results if item['execution_mode'] == execution_mode]
        grouped_counts.append(sorted({(item['run_name'], item['requested_precision_profile']) for item in mode_results}))
    rows, cols = _subplot_grid(len(execution_modes))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=panel_figure_size(len(execution_modes), max_legend_items(grouped_counts), cols),
        squeeze=False,
    )
    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes[len(execution_modes):]:
        axis.set_visible(False)
    axes = flat_axes[:len(execution_modes)]
    for axis, execution_mode in zip(axes, execution_modes):
        mode_results = [item for item in results if item['execution_mode'] == execution_mode]
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for item in mode_results:
            grouped.setdefault((item['run_name'], item['requested_precision_profile']), []).append(item)
        for series_index, ((run_name, precision), items) in enumerate(sorted(grouped.items())):
            items = sorted(items, key=lambda entry: entry['batch_size'])
            plot_style = style_for_series(series_index)
            axis.plot(
                [entry['batch_size'] for entry in items],
                [_metric_value(entry, metric) for entry in items],
                label=f'{run_name} / {precision}',
                **plot_style,
            )
        axis.set_title(f'{execution_mode} (threads={thread_count})')
        axis.set_xlabel('Batch Size')
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, alpha=0.3)
        place_legend_outside_right(fig, axis, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def _save_model_precision_panel_figure(results: List[Dict[str, Any]], output_path: Path, metric: str, thread_count: int):
    model_precision_keys = sorted({(item['run_name'], item['requested_precision_profile']) for item in results})
    if not model_precision_keys:
        return None
    grouped_counts = []
    for run_name, precision in model_precision_keys:
        subset = [item for item in results if item['run_name'] == run_name and item['requested_precision_profile'] == precision]
        grouped_counts.append(sorted({item['execution_mode'] for item in subset}))
    rows, cols = _subplot_grid(len(model_precision_keys))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=panel_figure_size(len(model_precision_keys), max_legend_items(grouped_counts), cols),
        squeeze=False,
    )
    flat_axes = [axis for row in axes for axis in row]
    for axis in flat_axes[len(model_precision_keys):]:
        axis.set_visible(False)
    axes = flat_axes[:len(model_precision_keys)]
    for axis, (run_name, precision) in zip(axes, model_precision_keys):
        subset = [item for item in results if item['run_name'] == run_name and item['requested_precision_profile'] == precision]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for item in subset:
            grouped.setdefault(item['execution_mode'], []).append(item)
        for series_index, (execution_mode, items) in enumerate(sorted(grouped.items())):
            items = sorted(items, key=lambda entry: entry['batch_size'])
            plot_style = style_for_series(series_index)
            axis.plot(
                [entry['batch_size'] for entry in items],
                [_metric_value(entry, metric) for entry in items],
                label=execution_mode,
                **plot_style,
            )
        axis.set_title(f'{run_name} / {precision} (threads={thread_count})')
        axis.set_xlabel('Batch Size')
        axis.set_ylabel(_metric_label(metric))
        axis.grid(True, alpha=0.3)
        place_legend_outside_right(fig, axis, fontsize=9)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def generate_latency_comparison_plots(
    input_path,
    output_dir=None,
    metric: str = 'p50_latency_ms',
    runs: Optional[str] = None,
    execution_modes: Optional[str] = None,
    precision_profiles: Optional[str] = None,
    thread_counts: Optional[str] = None,
) -> List[Path]:
    json_path, default_output_dir = resolve_latency_plot_input(input_path)
    output_dir = Path(output_dir) if output_dir else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as input_file:
        payload = json.load(input_file)

    filtered = _filter_results(
        payload.get('results', []),
        runs=_parse_optional_csv(runs),
        execution_modes=_parse_optional_csv(execution_modes),
        precision_profiles=_parse_optional_csv(precision_profiles),
        thread_counts=_coerce_threads(thread_counts),
    )
    if not filtered:
        return []

    available_threads = sorted({int(item['num_threads']) for item in filtered})
    generated: List[Path] = []
    for thread_count in available_threads:
        thread_results = [item for item in filtered if int(item['num_threads']) == thread_count]
        mode_path = output_dir / f'mode_panels_{metric}_threads_{thread_count}.jpg'
        model_precision_path = output_dir / f'model_precision_panels_{metric}_threads_{thread_count}.jpg'
        mode_file = _save_mode_panel_figure(thread_results, mode_path, metric, thread_count)
        model_precision_file = _save_model_precision_panel_figure(thread_results, model_precision_path, metric, thread_count)
        if mode_file is not None:
            generated.append(mode_file)
        if model_precision_file is not None:
            generated.append(model_precision_file)
    return generated
