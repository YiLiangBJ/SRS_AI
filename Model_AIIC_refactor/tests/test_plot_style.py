"""Tests for shared plotting style helpers."""

import unittest

from utils.plot_style import legend_fontsize, outside_legend_figure_size, style_for_series


class TestPlotStyle(unittest.TestCase):
    def test_outside_legend_figure_size_prefers_width_over_height_growth(self):
        small_width, small_height, small_cols, _ = outside_legend_figure_size(4, base_width=10.0, base_height=6.0)
        large_width, large_height, large_cols, _ = outside_legend_figure_size(48, base_width=10.0, base_height=6.0)

        self.assertGreaterEqual(large_width, small_width)
        self.assertEqual(large_height, small_height)
        self.assertGreaterEqual(large_cols, small_cols)

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
        self.assertEqual(legend_fontsize(8, base_fontsize=10), 10)
        self.assertLess(legend_fontsize(24, base_fontsize=10), 10)
        self.assertLessEqual(legend_fontsize(48, base_fontsize=10), legend_fontsize(24, base_fontsize=10))


if __name__ == '__main__':
    unittest.main()