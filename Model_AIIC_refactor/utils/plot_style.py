"""Shared plotting style helpers for readable multi-series figures."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib.pyplot as plt


_LINESTYLES = [
    '-',
    '--',
    '-.',
    ':',
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (7, 2, 1, 2)),
    (0, (2, 1)),
]
_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _palette_colors() -> list:
    colors: list = []
    for cmap_name in ['tab20', 'Dark2', 'Set2', 'Set1']:
        cmap = plt.get_cmap(cmap_name)
        if hasattr(cmap, 'colors'):
            colors.extend(list(cmap.colors))
    return colors


_COLORS = _palette_colors()


def color_for_index(index: int):
    return _COLORS[index % len(_COLORS)]


def linestyle_for_index(index: int):
    return _LINESTYLES[index % len(_LINESTYLES)]


def marker_for_index(index: int):
    return _MARKERS[index % len(_MARKERS)]


def style_for_series(index: int, marker: bool = True) -> dict:
    style = {
        'color': color_for_index(index),
        'linestyle': linestyle_for_index(index),
        'linewidth': 2.2,
    }
    if marker:
        style['marker'] = marker_for_index(index)
        style['markersize'] = 5.5
    return style


def legend_column_count(item_count: int, max_rows_per_col: int = 18) -> int:
    return 1


def legend_row_count(item_count: int, ncol: int) -> int:
    return max(1, math.ceil(max(1, item_count) / max(1, ncol)))


def outside_legend_figure_size(
    legend_count: int,
    base_width: float = 10.0,
    base_height: float = 6.0,
    max_rows_per_col: int = 18,
) -> tuple[float, float, int, tuple[float, float, float, float]]:
    ncol = legend_column_count(legend_count, max_rows_per_col=max_rows_per_col)
    width = base_width + 4.2
    height = base_height
    usable_right = 0.74
    rect = (0.0, 0.0, usable_right, 1.0)
    return width, height, ncol, rect


def panel_figure_size(panel_count: int, max_legend_count: int, cols: int) -> tuple[float, float]:
    rows = math.ceil(max(1, panel_count) / max(1, cols))
    width = cols * 11.5
    height = rows * 5.2
    return width, min(26.0, height)


def legend_fontsize(item_count: int, base_fontsize: int = 10, base_height: float = 6.0) -> int:
    max_items_at_base = max(8, int(base_height * 3.0))
    if item_count <= max_items_at_base:
        return base_fontsize
    if item_count <= int(max_items_at_base * 1.35):
        return max(8, base_fontsize - 1)
    if item_count <= int(max_items_at_base * 1.75):
        return max(7, base_fontsize - 2)
    if item_count <= int(max_items_at_base * 2.2):
        return max(6, base_fontsize - 3)
    return max(5, base_fontsize - 4)


def bottom_legend_column_count(
    labels: Sequence[str],
    base_width: float,
    max_cols: int = 4,
    min_col_width: float = 2.8,
) -> int:
    item_count = max(1, len(labels))
    width_limited_cols = max(1, min(max_cols, int(base_width / min_col_width)))
    if not labels:
        return width_limited_cols

    avg_label_len = sum(len(label) for label in labels) / item_count
    max_label_len = max(len(label) for label in labels)
    # Estimate how many legend columns fit before text becomes cramped.
    estimated_chars_per_col = max(18.0, avg_label_len * 0.75 + max_label_len * 0.25)
    char_limited_cols = max(1, int((base_width * 11.0) / estimated_chars_per_col))
    return max(1, min(item_count, width_limited_cols, char_limited_cols))


def bottom_legend_figure_size(
    labels: Sequence[str],
    base_width: float = 10.0,
    base_height: float = 6.0,
    max_cols: int = 4,
) -> tuple[float, float, int, tuple[float, float, float, float]]:
    ncol = bottom_legend_column_count(labels, base_width=base_width, max_cols=max_cols)
    rows = legend_row_count(len(labels), ncol)
    legend_height = 0.9 + 0.36 * max(1, rows)
    bottom_margin = min(0.34, 0.11 + 0.065 * rows)
    width = base_width
    height = base_height + legend_height
    rect = (0.0, bottom_margin, 1.0, 1.0)
    return width, height, ncol, rect


def place_legend_outside_right(figure, axis, fontsize: int = 10, max_rows_per_col: int = 18):
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    _, figure_height, ncol, rect = outside_legend_figure_size(
        len(labels),
        base_width=figure.get_size_inches()[0],
        base_height=figure.get_size_inches()[1],
        max_rows_per_col=max_rows_per_col,
    )
    axis.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=legend_fontsize(len(labels), base_fontsize=fontsize, base_height=figure_height),
        ncol=ncol,
        columnspacing=1.2,
        handlelength=3.0,
        frameon=True,
    )
    figure.tight_layout(rect=rect)


def place_legend_below(figure, axis, fontsize: int = 10, max_cols: int = 4):
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return

    _, figure_height, ncol, rect = bottom_legend_figure_size(
        labels,
        base_width=figure.get_size_inches()[0],
        base_height=figure.get_size_inches()[1],
        max_cols=max_cols,
    )
    figure.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        borderaxespad=0.0,
        fontsize=legend_fontsize(len(labels), base_fontsize=fontsize, base_height=figure_height),
        ncol=ncol,
        columnspacing=1.4,
        handlelength=2.8,
        frameon=True,
    )
    figure.tight_layout(rect=rect)


def max_legend_items(series_groups: Iterable[Sequence]) -> int:
    return max((len(group) for group in series_groups), default=1)