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
    return max(1, min(5, math.ceil(max(1, item_count) / max_rows_per_col)))


def legend_row_count(item_count: int, ncol: int) -> int:
    return max(1, math.ceil(max(1, item_count) / max(1, ncol)))


def outside_legend_figure_size(
    legend_count: int,
    base_width: float = 10.0,
    base_height: float = 6.0,
    max_rows_per_col: int = 18,
) -> tuple[float, float, int, tuple[float, float, float, float]]:
    ncol = legend_column_count(legend_count, max_rows_per_col=max_rows_per_col)
    width = base_width + 2.4 + 1.8 * (ncol - 1)
    height = base_height
    usable_right = max(0.38, 0.80 - 0.08 * (ncol - 1))
    rect = (0.0, 0.0, usable_right, 1.0)
    return width, height, ncol, rect


def panel_figure_size(panel_count: int, max_legend_count: int, cols: int) -> tuple[float, float]:
    rows = math.ceil(max(1, panel_count) / max(1, cols))
    width = cols * 11.5
    height = rows * 5.2
    return width, min(26.0, height)


def legend_fontsize(item_count: int, base_fontsize: int = 10) -> int:
    if item_count <= 12:
        return base_fontsize
    if item_count <= 20:
        return max(8, base_fontsize - 1)
    if item_count <= 32:
        return max(7, base_fontsize - 2)
    return max(6, base_fontsize - 3)


def place_legend_outside_right(figure, axis, fontsize: int = 10, max_rows_per_col: int = 18):
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return
    _, _, ncol, rect = outside_legend_figure_size(
        len(labels),
        base_width=figure.get_size_inches()[0],
        base_height=figure.get_size_inches()[1],
        max_rows_per_col=max_rows_per_col,
    )
    axis.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=legend_fontsize(len(labels), base_fontsize=fontsize),
        ncol=ncol,
        columnspacing=1.2,
        handlelength=3.0,
        frameon=True,
    )
    figure.tight_layout(rect=rect)


def max_legend_items(series_groups: Iterable[Sequence]) -> int:
    return max((len(group) for group in series_groups), default=1)