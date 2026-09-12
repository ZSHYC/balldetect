"""一项手算检查：图像y向下、角度单位、无符号轴和距离保持。"""
import unittest

import numpy as np

from scripts.analyze_blurball_midpoint import components, describe


class MidpointAxisTest(unittest.TestCase):
    def test_image_axis_geometry(self):
        delta = np.array([[3., 4.], [3., 4.], [3., 4.], [4., 0.]])
        actual = components(delta, [0., 90., 180., 0.])
        np.testing.assert_allclose(actual, [[3., 4.], [4., -3.], [-3., -4.], [4., 0.]], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(delta, axis=1))
        self.assertEqual(describe(actual)['parallel_dominant_fraction'], .5)
        self.assertEqual(describe(actual)['perpendicular_lt4_fraction'], .5)
        self.assertEqual(describe(actual[:0]), {'n': 0})
