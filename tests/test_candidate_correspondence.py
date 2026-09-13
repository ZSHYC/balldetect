"""候选query到两帧原生特征格的全局cosine对应。"""
import math
import unittest

import torch

from ballmotion.correspondence import candidate_costs


class CandidateCorrespondenceTest(unittest.TestCase):
    def test_original_coordinates_centering_interpolation_order_and_border(self):
        centered = torch.tensor([
            [[3., -1.], [-1., -1.]],
            [[-1., -1.], [-1., 3.]],
        ])
        features = torch.stack((-centered, centered, centered), dim=0)[None]
        features[:, 0] += torch.tensor([10., -7.])[None, :, None, None]
        features[:, 1] += torch.tensor([4., 9.])[None, :, None, None]
        features[:, 2] += torch.tensor([-5., 2.])[None, :, None, None]
        current_xy = torch.tensor([[[0., 0.], [7., 3.], [2.5, 1.], [-5., -3.]]])

        costs = candidate_costs(features, current_xy, torch.tensor([[8., 4.]]))

        self.assertEqual(costs.shape, (1, 2, 4, 4))
        self.assertEqual(costs.dtype, torch.float32)
        self.assertEqual(costs[0, 0, 0].argmax().item(), 0)
        self.assertEqual(costs[0, 0, 1].argmax().item(), 3)
        torch.testing.assert_close(costs[0, 0, 0, 0], torch.tensor(1.))
        torch.testing.assert_close(costs[0, 1, 0, 0], torch.tensor(-1.))
        torch.testing.assert_close(costs[0, 0, 1, 0], torch.tensor(-.6))
        torch.testing.assert_close(
            costs[0, 0, 2, 0], torch.tensor(4.5 / math.sqrt(21.25)))
        torch.testing.assert_close(costs[0, 0, 3, 0], torch.tensor(1.))


if __name__ == '__main__':
    unittest.main()
