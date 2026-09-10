"""Dense local correspondence keeps time, offset, and boundary semantics explicit."""
import unittest

import torch
from torch.nn import functional as F

from ballmotion.correspondence import cost_volume


class CorrespondenceCacheTest(unittest.TestCase):
    def test_known_offsets_boundary_self_control_and_scalar_cosine(self):
        features = torch.zeros(1, 3, 192, 3, 4, dtype=torch.float32)
        query_yx = (1, 1)
        features[:, :, 3] = 1
        features[0, 2, :, query_yx[0], query_yx[1]] = 0
        features[0, 2, 0, *query_yx] = 1
        features[0, 2, 1, *query_yx] = 2
        features[0, 1, :, 0, 2] = features[0, 2, :, *query_yx]
        features[0, 0, :, 2, 0] = 0
        features[0, 0, 0, 2, 0] = 1

        raw = cost_volume(features, "raw")
        centered = cost_volume(features, "centered")
        self_centered = cost_volume(features, "self_centered")

        delta1_match = (-1 + 2) * 5 + (1 + 2)
        delta2_match = 25 + (1 + 4) * 9 + (-1 + 4)
        self_delta1_center = 2 * 5 + 2
        self_delta2_center = 25 + 4 * 9 + 4
        self.assertEqual(raw.shape, (1, 106, 3, 4))
        self.assertEqual(raw.dtype, torch.float32)
        torch.testing.assert_close(raw[0, delta1_match, *query_yx], torch.tensor(1.0))
        torch.testing.assert_close(raw[0, delta2_match, *query_yx], torch.tensor(5 ** -.5))
        self.assertEqual(raw[0, 16, *query_yx], 0)  # delta1 cannot read delta2's key
        self.assertEqual(raw[0, 25 + 32, *query_yx], 0)  # delta2 cannot read delta1's key
        self.assertLess(raw[0, 12, *query_yx], 1)
        self.assertLess(raw[0, 25 + 40, *query_yx], 1)
        self.assertEqual(raw[0, 7, 0, 0], -2)  # delta1 dy=-1, dx=0
        self.assertTrue(torch.isfinite(raw).all())

        current = features[0, 2] - features[0, 2].mean((-2, -1), keepdim=True)
        previous = features[0, 1] - features[0, 1].mean((-2, -1), keepdim=True)
        expected = F.cosine_similarity(current[:, 1, 1], previous[:, 0, 2], dim=0)
        torch.testing.assert_close(centered[0, delta1_match, *query_yx], expected)
        self.assertLess(centered[0, self_delta1_center, *query_yx], 1)
        torch.testing.assert_close(self_centered[0, self_delta1_center, *query_yx],
                                   torch.tensor(1.0))
        torch.testing.assert_close(self_centered[0, self_delta2_center, *query_yx],
                                   torch.tensor(1.0))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
