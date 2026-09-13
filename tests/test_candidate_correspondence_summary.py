import unittest

import numpy as np
import torch

from scripts.probe_blurball_candidate_correspondence import summarize_matches, true_cell_ranks


class CorrespondenceSummaryTest(unittest.TestCase):
    def test_last_half_pixel_uses_existing_edge_cell_convention(self):
        frames = [dict(visibility_raw=1, x_raw=x, y_raw=0., width=2, height=1)
                  for x in (1.4, 1.4, 1.9)]
        arrays = dict(matched_cells=np.ones((1, 2, 1), dtype=int),
                      true_ranks=np.ones((1, 2, 1), dtype=int),
                      max_cosine=np.ones((1, 2, 1)))
        result = summarize_matches([frames[2]], frames, np.array([[0, 1, 2]]),
                                   np.array([[[1.9, 0.]]]), arrays, (1, 2),
                                   {'all': np.ones(1, dtype=bool)})
        self.assertEqual(result['1']['native_same']['both_visible_n'], 1)

    def test_stable_rank_counts_earlier_ties(self):
        costs = torch.tensor([[[[.5, .5, .8], [.2, .2, .2]],
                               [[.8, .5, .5], [.2, .3, .2]]]])
        cells = torch.tensor([[1, 2]])
        np.testing.assert_array_equal(true_cell_ranks(costs, cells).numpy(), [[[3, 2], [3, 3]]])

    def test_conditional_matching_does_not_hide_missing_current_candidate(self):
        frames = [dict(visibility_raw=1, x_raw=0., y_raw=0., width=2, height=1)
                  for _ in range(6)]
        rows = [frames[2], frames[5]]
        windows = np.array([[0, 1, 2], [3, 4, 5]])
        xy = np.array([[[20., 0.], [0., 0.]], [[20., 0.], [30., 0.]]])
        arrays = dict(matched_cells=np.zeros((2, 2, 2), dtype=int),
                      true_ranks=np.ones((2, 2, 2), dtype=int),
                      max_cosine=np.array([[[.9, .8], [.9, .8]], [[.7, .6], [.7, .6]]]))
        g = summarize_matches(rows, frames, windows, xy, arrays, (1, 2),
                              {'all': np.ones(2, dtype=bool)})['1']['all']
        self.assertEqual(g['both_visible_n'], 2)
        self.assertEqual(g['current_candidate_covered4_n'], 1)
        self.assertEqual(g['conditional_history_pck16'], 1.)
        self.assertEqual(g['joint_current4_history16_rate'], .5)
        self.assertEqual(g['far_wrong_top1_pair']['n'], 1)
        self.assertEqual(g['far_wrong_top1_pair']['correct_score_loses'], 1)


if __name__ == '__main__':
    unittest.main()
