import unittest

import torch
from torch.nn import functional as F

from ballmotion.candidate_readout import CandidateResidualReadout, candidate_targets


class CandidateReadoutTest(unittest.TestCase):
    def test_zero_initialization_keeps_unary_and_can_learn(self):
        torch.manual_seed(0)
        model = CandidateResidualReadout(channels=2, hidden=3)
        q = torch.tensor([[[1., 0.], [0., 1.]]])
        history = torch.zeros(1, 2, 2, 2)
        peaks = torch.tensor([[7., 6.]])
        initial = model(q, history, peaks)
        torch.testing.assert_close(initial, torch.tensor([[0., -1.]]), rtol=0, atol=0)
        loss = F.cross_entropy(initial, torch.tensor([1]))
        loss.backward()
        self.assertGreater(model.layers[-1].weight.grad.abs().sum().item(), 0.)
        torch.optim.SGD(model.parameters(), lr=.1).step()
        self.assertFalse(torch.equal(model(q, history, peaks), initial))

    def test_near_far_order_and_original_score_difference(self):
        model = CandidateResidualReadout(channels=1, hidden=1)
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.tensor([[1., 2., 3.]]))
            model.layers[0].bias.zero_()
            model.layers[-1].weight.fill_(1.)
        q = torch.tensor([[[1.], [5.]]])
        history = torch.tensor([[[[2.], [7.]], [[4.], [11.]]]])
        score = model(q, history, torch.tensor([[10., 8.]]))
        torch.testing.assert_close(score, torch.tensor([[17., 50.]]))

    def test_candidate_soft_labels_follow_original_distance_and_translation(self):
        xy = torch.tensor([[[0., 0.], [4., 0.], [8., 0.]]])
        gt = torch.tensor([[0., 0.]])
        target = candidate_targets(xy, gt)
        torch.testing.assert_close(target, torch.tensor([[.57409699, .34820743, .07769558]]))
        torch.testing.assert_close(target, candidate_targets(xy+1000, gt+1000), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
