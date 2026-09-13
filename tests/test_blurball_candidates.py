import unittest

import numpy as np
import torch

from scripts.probe_blurball_candidates import greedy_candidates, coverage


class CandidateTest(unittest.TestCase):
    def test_suppression_ties_edges_and_original_patch(self):
        spatial = torch.full((1, 32, 32), -20.)
        spatial[0, 0, 0], spatial[0, 0, 1] = 10., 9.
        spatial[0, 0, 8] = spatial[0, 8, 0] = 8.
        before = spatial.clone()
        centers, scores, patches = greedy_candidates(spatial, 3)
        torch.testing.assert_close(centers, torch.tensor([[[0, 0], [8, 0], [0, 8]]]))
        torch.testing.assert_close(scores, torch.tensor([[10., 8., 8.]]))
        torch.testing.assert_close(spatial, before)
        self.assertEqual(patches[0, 1, 7, 0].item(), 9.)
        self.assertTrue(torch.isneginf(patches[0, 0, :7]).all())

    def test_oracle_prefix_strict_radius_and_v0(self):
        rows = [dict(x_raw=0., y_raw=0., visibility_raw=v) for v in (1, 1, 0)]
        xy = np.array([[[4, 0], [3, 0]], [[20, 0], [16, 0]], [[0, 0], [0, 0]]])
        result = coverage(rows, xy, {'all': np.ones(3, dtype=bool)}, (1, 2))['all']
        self.assertEqual(result['n'], 2)
        self.assertEqual(result['budgets']['1']['4']['covered'], 0)
        self.assertEqual(result['budgets']['2']['4']['covered'], 1)
        self.assertEqual(result['budgets']['2']['16']['covered'], 1)


if __name__ == '__main__':
    unittest.main()
