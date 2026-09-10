import importlib.util
from pathlib import Path
import unittest

import torch

torch.set_num_threads(2)

SCRIPT = Path(__file__).parents[1] / "scripts/analyze_tennis_readout.py"
spec = importlib.util.spec_from_file_location("analyze_tennis_readout", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReadoutConcentrationTest(unittest.TestCase):
    def test_conditional_concentration_uses_predicted_cell_and_valid_boundary(self):
        height, width = 72, 128
        cells = height * width
        center = 30 * width + 40
        far = 60 * width + 100

        single = torch.full((1, cells + 1), -100.)
        single[0, center] = 0
        single[0, -1] = torch.logsumexp(single[0, :-1], 0)
        dual = single.clone()
        dual[0, far] = 0
        dual[0, -1] = torch.logsumexp(dual[0, :-1], 0)
        single_scores = module.readout_scores(single, (height, width))
        dual_scores = module.readout_scores(dual, (height, width))

        self.assertAlmostEqual(single_scores["q"].item(), .5, places=6)
        self.assertAlmostEqual(dual_scores["q"].item(), .5, places=6)
        self.assertGreater(single_scores["maxc"].item(), dual_scores["maxc"].item())
        self.assertGreater(single_scores["m16"].item(), dual_scores["m16"].item())
        self.assertLess(single_scores["entropy_normalized"].item(),
                        dual_scores["entropy_normalized"].item())
        self.assertEqual(single_scores["predicted_cell"].item(), center)

        boundary = torch.zeros((1, cells + 1))
        boundary[0, -1] = torch.log(torch.tensor(float(cells)))
        boundary_scores = module.readout_scores(boundary, (height, width))
        self.assertEqual(boundary_scores["predicted_cell"].item(), 0)
        self.assertAlmostEqual(boundary_scores["m16"].item(), 4 / cells, places=8)

        joint = single.softmax(1)[0, :-1].reshape(1, 1, height, width)
        center_y, center_x = divmod(center, width)
        joint_neighborhood = joint[0, 0, center_y - 1:center_y + 2,
                                   center_x - 1:center_x + 2].sum()
        self.assertAlmostEqual(single_scores["q_times_m16"].item(),
                               joint_neighborhood.item(), places=7)
        self.assertEqual(module.auroc([1., 1.], [True, False]), .5)
        self.assertEqual(module.auroc([2., 1.], [True, False]), 1.)


if __name__ == "__main__":
    unittest.main()
