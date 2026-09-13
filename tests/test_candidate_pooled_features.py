import unittest

import torch

from ballmotion.correspondence import candidate_pooled_features


class CandidatePooledFeaturesTest(unittest.TestCase):
    def test_displaced_correspondence_aggregates_before_selection(self):
        # Current object at left; near at right, far at left. Centered maps.
        current = torch.tensor([[[1., -1.]], [[0., 0.]]])
        frames = torch.stack((current, -current, current))[None]
        query, local, pooled = candidate_pooled_features(
            frames, torch.tensor([[[.5, .5]]]), torch.tensor([[4., 2.]]))
        torch.testing.assert_close(query, torch.tensor([[[1., 0.]]]))
        torch.testing.assert_close(local, torch.tensor([[[[-1., 0.]], [[1., 0.]]]]))
        # softmax(+1/.1,-1/.1), weighted signed unit keys.
        expected = torch.tensor(10.).tanh()
        torch.testing.assert_close(pooled[0, :, 0, 0], expected.expand(2))
        torch.testing.assert_close(pooled[0, :, 0, 1], torch.zeros(2))

    def test_zero_descriptors_remain_finite_and_do_not_invent_evidence(self):
        q, local, pooled = candidate_pooled_features(
            torch.ones(2, 3, 4, 2, 3), torch.zeros(2, 2, 2), torch.tensor([[6., 4.], [6., 4.]]))
        for value in (q, local, pooled):
            self.assertTrue(torch.isfinite(value).all())
            self.assertEqual(value.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
