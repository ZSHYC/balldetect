import unittest

import torch

from ballmotion.correspondence import top_candidate_matches
from ballmotion.candidate_readout import CandidateResidualReadout, candidate_match_inputs


class CandidateMatchAddressTest(unittest.TestCase):
    def test_score_and_cell_stay_bound_with_stable_ties(self):
        costs = torch.zeros(1, 2, 1, 20)
        costs[0, 0, 0, 17] = .8
        costs[0, 1, 0, 19] = .9
        score, cell = top_candidate_matches(costs)
        self.assertEqual(cell[0, 0, 0].tolist(), [17]+list(range(15)))
        self.assertEqual(cell[0, 1, 0].tolist(), [19]+list(range(15)))
        torch.testing.assert_close(score, costs.gather(-1, cell))

    def test_relative_offsets_use_cell_centers_and_keep_near_far_order(self):
        score = torch.tensor([[[[.8]], [[.6]]]])
        cells = torch.tensor([[[[5]], [[0]]]])
        # Grid2x4 on80x20: cell5=(29.5,14.5); query=(9.5,4.5).
        packed = candidate_match_inputs(score, cells, torch.tensor([[[9.5, 4.5]]]),
                                        torch.tensor([[80., 20.]]), (2, 4))
        torch.testing.assert_close(packed, torch.tensor([[[.8, .25, .5, .6, 0., 0.]]]))

    def test_match_scalars_are_not_scaled_as_visual_descriptors(self):
        model = CandidateResidualReadout(channels=4, hidden=1, matching_features=3)
        with torch.no_grad():
            model.layers[0].weight.zero_()
            model.layers[0].weight[0, 0] = 1.
            model.layers[0].weight[0, -3:] = torch.tensor([1., 2., 3.])
            model.layers[0].bias.zero_()
            model.layers[-1].weight.fill_(1.)
        q = torch.tensor([[[2., 0., 0., 0.]]])
        actual = model(q, torch.zeros(1, 2, 1, 4), torch.zeros(1, 1), torch.tensor([[[1., 2., 3.]]]))
        # sqrt(4)*2 + 1 + 2*2 + 3*3 = 18, GELU(18)=18 at this precision.
        torch.testing.assert_close(actual, torch.tensor([[18.]]))


if __name__ == '__main__':
    unittest.main()
