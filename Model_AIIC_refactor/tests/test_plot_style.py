"""Tests for shared plotting style helpers."""

import unittest

from utils.plot_style import (
    bottom_legend_column_count,
    bottom_legend_figure_size,
    legend_fontsize,
    outside_legend_figure_size,
    style_for_series,
)


class TestPlotStyle(unittest.TestCase):
    def test_outside_legend_figure_size_prefers_width_over_height_growth(self):
        small_width, small_height, small_cols, small_rect = outside_legend_figure_size(4, base_width=10.0, base_height=6.0)
        large_width, large_height, large_cols, large_rect = outside_legend_figure_size(48, base_width=10.0, base_height=6.0)

        self.assertGreaterEqual(large_width, small_width)
        self.assertEqual(large_height, small_height)
        self.assertEqual(large_cols, 1)
        self.assertEqual(small_cols, 1)
        self.assertEqual(small_rect[2], large_rect[2])

    def test_bottom_legend_figure_size_prefers_height_over_width_growth(self):
        small_labels = ['model_a', 'model_b']
        large_labels = [f'model_{index}_with_a_longer_label' for index in range(12)]

        small_width, small_height, _, small_rect = bottom_legend_figure_size(small_labels, base_width=10.0, base_height=6.0)
        large_width, large_height, large_cols, large_rect = bottom_legend_figure_size(large_labels, base_width=10.0, base_height=6.0)

        self.assertEqual(large_width, small_width)
        self.assertGreater(large_height, small_height)
        self.assertGreaterEqual(large_cols, 1)
        self.assertGreater(large_rect[1], small_rect[1])

    def test_bottom_legend_column_count_shrinks_for_long_labels(self):
        short_labels = [f'm{index}' for index in range(8)]
        long_labels = [f'model_{index}_with_very_long_descriptor_text' for index in range(8)]

        short_cols = bottom_legend_column_count(short_labels, base_width=12.0, max_cols=4)
        long_cols = bottom_legend_column_count(long_labels, base_width=12.0, max_cols=4)

        self.assertGreaterEqual(short_cols, long_cols)

    def test_style_for_series_distinguishes_neighboring_series(self):
        first = style_for_series(0)
        second = style_for_series(1)
        third = style_for_series(2)

        self.assertNotEqual((first['color'], first['linestyle']), (second['color'], second['linestyle']))
        self.assertNotEqual((second['color'], second['linestyle']), (third['color'], third['linestyle']))

    def test_style_for_series_without_marker_omits_marker_fields(self):
        style = style_for_series(0, marker=False)
        self.assertNotIn('marker', style)
        self.assertNotIn('markersize', style)

    def test_legend_fontsize_shrinks_for_dense_legends(self):
        self.assertEqual(legend_fontsize(8, base_fontsize=10, base_height=6.0), 10)
        self.assertLess(legend_fontsize(24, base_fontsize=10, base_height=6.0), 10)
        self.assertLessEqual(
            legend_fontsize(48, base_fontsize=10, base_height=6.0),
            legend_fontsize(24, base_fontsize=10, base_height=6.0),
        )


if __name__ == '__main__':
    unittest.main()