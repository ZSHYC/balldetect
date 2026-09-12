import unittest

import numpy as np
import torch

from scripts.analyze_blurball_readout import barycenters, local_patches, paired_groups


class LocalReadoutTest(unittest.TestCase):
    def test_peak_edges_expectation_and_pairing(self):
        logits = torch.full((2, 3, 4), -torch.inf)
        logits[0, 0, 0] = logits[0, 0, 2] = 0
        logits[1, 2, 3] = 0
        center, patch = local_patches(logits)
        np.testing.assert_array_equal(center, [[0, 0], [3, 2]])
        self.assertEqual(torch.isfinite(patch[0]).sum(), 2)
        np.testing.assert_allclose(barycenters(center.numpy(), patch.numpy()), [[1, 0], [3, 2]])
        rows = [{'x_raw': 0, 'y_raw': 0, 'visibility_raw': v, 'l_raw': 11,
                 'game': 'match18'} for v in (1, 1, 0)]
        result = paired_groups(rows, np.array([[4, 0], [0, 0], [0, 0]]),
                               np.array([[3, 0], [5, 0], [9, 0]]))
        self.assertEqual(result['all/gt10']['4'],
                         dict(n=2, rescued=1, broken=1, both_correct=0, both_wrong=0))
        self.assertEqual(result['all/gt10/old_4_16']['4']['rescued'], 1)
        self.assertEqual(result['all/gt10/old_lt4']['4']['broken'], 1)
