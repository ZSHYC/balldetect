import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.probe_blurball_candidates import coverage, extract_candidate_arrays, greedy_candidates


class CandidateTest(unittest.TestCase):
    def test_extract_candidate_arrays_batches_and_preserves_export_contract(self):
        windows = np.array([[0, 1, 2], [3, 4, 5]])
        rows = [dict(width=512, height=288), dict(width=1024, height=576)]
        logits = []
        for x, y, presence in ((10, 20, 0.), (30, 40, 1.)):
            value = torch.full((1, 288 * 512 + 1), -20.)
            value[0, y * 512 + x] = 10.
            value[0, -1] = presence
            logits.append(value)

        with patch('scripts.probe_blurball_candidates.batch_logits', side_effect=logits) as forward:
            result = extract_candidate_arrays(object(), np.empty(0), windows, rows,
                                              torch.device('cpu'), batch_size=1)

        self.assertEqual(forward.call_count, 2)
        np.testing.assert_array_equal(result['grid_centers'][:, 0], [[10, 20], [30, 40]])
        np.testing.assert_array_equal(result['original_xy'][:, 0], [[10, 20], [60.5, 80.5]])
        np.testing.assert_array_equal(result['current_frame_ids'], [2, 5])
        self.assertEqual(result['grid_centers'].shape, (2, 16, 2))
        self.assertEqual(result['local_xy'].shape, (2, 16, 2))
        self.assertEqual(result['peak_logits'].shape, (2, 16))
        np.testing.assert_allclose(
            result['q'], [0.99995458, 0.99987662], rtol=0, atol=1e-7)

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
