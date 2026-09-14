"""邻帧命中不能来自未知标签、重叠容差圆或边界上的非严格命中。"""
import unittest

import numpy as np

from scripts.analyze_blurball_temporal_locations import temporal_location_masks


class TemporalLocationTests(unittest.TestCase):
    def test_visibility_separation_and_strict_radius(self):
        target = np.zeros((5, 2))
        prediction = np.array([[8, 0], [8, 0], [8, 0], [7, 0], [12, 0]])
        support = np.array([[[8, 0]], [[8, 0]], [[8, 0]], [[7, 0]], [[8, 0]]])
        visible = np.array([True, False, True, True, True])
        support_visible = np.array([[True], [True], [False], [True], [True]])
        _, eligible, hits = temporal_location_masks(
            prediction, target, visible, support, support_visible)
        np.testing.assert_array_equal(eligible[:, 0], [True, False, False, False, True])
        np.testing.assert_array_equal(hits[:, 0], [True, False, False, False, False])

    def test_multiple_support_hits_keep_individual_slots(self):
        _, eligible, hits = temporal_location_masks(
            np.array([[10., 0.]]), np.array([[0., 0.]]), np.array([True]),
            np.array([[[9., 0.], [11., 0.], [-10., 0.], [0., 0.]]]),
            np.ones((1, 4), dtype=bool))
        np.testing.assert_array_equal(eligible, [[True, True, True, False]])
        np.testing.assert_array_equal(hits, [[True, True, False, False]])


if __name__ == '__main__':
    unittest.main()
