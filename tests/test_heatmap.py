import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ballmotion.heatmap import disk_targets, quality_focal_loss, heatmap_predictions


class HeatmapTest(unittest.TestCase):
    def test_disk_absence_qfl_and_original_coordinates(self):
        targets = disk_targets(torch.tensor([4 * 9 + 4, 0, 81]), (9, 9))
        self.assertEqual(tuple(targets.shape), (3, 1, 9, 9))
        self.assertEqual(targets.flatten(1).sum(1).tolist(), [21., 8., 0.])
        self.assertEqual(targets[0, 0, 4, 6].item(), 1.)
        self.assertEqual(targets[0, 0, 6, 6].item(), 0.)
        logits = torch.zeros_like(targets, requires_grad=True)
        loss = quality_focal_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), math.log(2) / 4, places=6)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 4, 4].item(), 0)
        self.assertGreater(logits.grad[2, 0, 4, 4].item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        peaks = torch.full((2, 1, 9, 9), -4.)
        peaks[0, 0, 2, 3] = 2.
        peaks[1, 0, 7, 6] = -1.
        xy, probabilities = heatmap_predictions(peaks)
        expected = np.array([[(3.5)*1280/9-.5, (2.5)*720/9-.5],
                             [(6.5)*1280/9-.5, (7.5)*720/9-.5]])
        np.testing.assert_allclose(xy, expected)
        np.testing.assert_allclose(probabilities, torch.sigmoid(torch.tensor([2., -1.])).numpy())
        self.assertEqual((probabilities >= .5).tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
