import unittest

import numpy as np

from scripts.probe_blurball_history_ball_evidence import history_predictions, supported_candidates


class HistoryBallEvidenceTest(unittest.TestCase):
    def test_source_identity_not_csv_adjacency_and_near_far_order(self):
        frames = [dict(game='match18', clip=clip, original_frame_id=f)
                  for clip, f in [('a', 10), ('a', 11), ('a', 12), ('b', 11)]]
        saved = [dict(frames[i], pred_x=x, pred_y=3., presence_probability=.9)
                 for i, x in [(3, 99.), (0, 10.)]]
        xy, q, available = history_predictions(frames, np.array([[0, 1, 2]]), saved)
        np.testing.assert_array_equal(available, [[False, True]])
        np.testing.assert_array_equal(q, [[0., .9]])
        np.testing.assert_array_equal(xy[0, 1], [10., 3.])

    def test_support_keeps_order_and_falls_back_on_rejection_or_missing(self):
        # Grid 2x4 over 80x20: cell5 center=(29.5,14.5), cell6=(49.5,14.5).
        cells = np.tile([[[0, 5, 6], [0, 5, 6]]], (4, 1, 1))
        xy = np.tile([[[39.5, 14.5], [39.5, 14.5]]], (4, 1, 1))
        xy[3] = [45.5, 14.5]  # cell5 exactly16 away must fail; cell6 passes.
        q = np.ones((4, 2))
        q[1, 1] = .49
        available = np.ones((4, 2), dtype=bool)
        available[2, 0] = False
        support, selected = supported_candidates(cells, (2, 4),
            np.tile([[[80, 20], [80, 20]]], (4, 1, 1)), xy, q, available)
        np.testing.assert_array_equal(selected, [1, 0, 0, 2])
        np.testing.assert_array_equal(support[0], [[False, True, True], [False, True, True]])


if __name__ == '__main__':
    unittest.main()
