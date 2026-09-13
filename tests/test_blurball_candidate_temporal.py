import unittest

import numpy as np

from scripts.rerank_blurball_candidates import history_indices, rerank, select_strength


class CandidateTemporalTest(unittest.TestCase):
    def test_training_selection_prioritizes_emitted_then_raw_then_smaller(self):
        rows = [dict(visibility_raw=1, x_raw=0., y_raw=0.) for _ in range(2)]
        xy = np.array([[[9., 0.], [0., 0.]], [[0., 0.], [9., 0.]]])
        choices = np.array([[0, 0], [1, 1], [1, 0], [1, 0]])
        chosen, records = select_strength(rows, xy, np.array([1., .1]), choices, [0., .25, 1., 4.])
        self.assertEqual(chosen, 1.)
        self.assertEqual([(r['tp4'], r['raw_correct4']) for r in records],
                         [(0, 1), (1, 1), (1, 2), (1, 2)])

    def setUp(self):
        self.rows = [dict(game='match00', clip='001', original_frame_id=i,
                          pts_seconds=i*.01) for i in range(3)]
        self.xy = np.array([[[0., 0.], [1000., 0.]],
                            [[10., 0.], [1000., 0.]],
                            [[9., 0.], [20., 0.]]])
        self.scores = np.array([[0., -100.], [0., -100.], [0., -.1]])
        self.q = np.ones(3)

    def test_constant_velocity_differs_from_static_and_zero_preserves(self):
        links = history_indices(self.rows)
        np.testing.assert_array_equal(links, [[-1, -1], [-1, -1], [0, 1]])
        velocity = rerank(self.xy, self.scores, self.q, self.rows, links, 'velocity', [0., 1.])
        stationary = rerank(self.xy, self.scores, self.q, self.rows, links, 'stationary', [1.])
        np.testing.assert_array_equal(velocity, [[0, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(stationary, [[0, 0, 0]])
        np.testing.assert_array_equal(
            rerank(self.xy, self.scores, self.q, self.rows, links, 'velocity', [0.]),
            [[0, 0, 0]])
        self.q[1] = .49
        np.testing.assert_array_equal(
            rerank(self.xy, self.scores, self.q, self.rows, links, 'velocity', [1.]),
            [[0, 0, 0]])

    def test_real_time_ratio_and_future_do_not_change_past(self):
        self.rows[2]['pts_seconds'] = .03
        self.xy[2, 1, 0] = 30.
        rows = self.rows + [dict(game='match00', clip='001', original_frame_id=3,
                                 pts_seconds=.04)]
        xy = np.concatenate((self.xy, [[[400., 600.], [2., 9.]]]))
        scores = np.concatenate((self.scores, [[0., -1.]]))
        q = np.ones(4)
        result = rerank(xy, scores, q, rows, history_indices(rows), 'velocity', [1.])
        np.testing.assert_array_equal(result[0, :3], [0, 0, 1])
        xy[3] = 5000.
        scores[3] = [-100., 0.]
        changed = rerank(xy, scores, q, rows, history_indices(rows), 'velocity', [1.])
        np.testing.assert_array_equal(changed[0, :3], result[0, :3])

    def test_missing_original_frames_and_clip_boundary_reset(self):
        rows = [dict(game='match00', clip='001', original_frame_id=i, pts_seconds=i*.01)
                for i in (0, 1, 4, 5, 6)]
        rows += [dict(game='match00', clip='002', original_frame_id=i, pts_seconds=i*.01)
                 for i in (5, 6, 7)]
        np.testing.assert_array_equal(history_indices(rows),
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [2, 3],
             [-1, -1], [-1, -1], [5, 6]])
        rows[-1]['pts_seconds'] = 0.
        np.testing.assert_array_equal(history_indices(rows)[-1], [-1, -1])


if __name__ == '__main__':
    unittest.main()
