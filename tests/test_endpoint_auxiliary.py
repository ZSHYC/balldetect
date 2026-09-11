import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ballmotion import correspondence

torch.set_num_threads(2)


class EndpointAuxiliaryTest(unittest.TestCase):
    def test_local_endpoint_supervision_and_appearance_control(self):
        # 反向offset、把VC2/3当正对应、越界参与softmax或按pair总数平均会失败。
        frames = [dict(x_raw=19.5, y_raw=0., visibility_raw=v) for v in (1, 2, 3, 0)]
        cells = correspondence.native_endpoint_cells(frames, np.array([[0, 1, 2], [1, 2, 3]]))
        self.assertEqual(cells.tolist(), [[1, 2304, 2304], [2304, 2304, 2304]])

        features = torch.zeros(3, 3, 3, 7, 9)
        features[0, 2, 0, 0, 0] = 1
        features[0, 1, 0, 0, 1] = 1
        features[0, 0, 0, 1, 0] = 1
        features.requires_grad_()
        cells = torch.tensor([[9, 1, 0], [63, 1, 63], [0, 8, 0]])
        pairs = correspondence.endpoint_logits(features, cells)
        self.assertEqual(pairs[0]["batch_indices"].tolist(), [0])
        self.assertEqual(pairs[1]["batch_indices"].tolist(), [0, 2])
        self.assertEqual(pairs[0]["target"].tolist(), [13])
        self.assertEqual(pairs[1]["target"].tolist(), [49, 40])
        self.assertEqual(pairs[0]["logits"].argmax(1).tolist(), [13])
        self.assertEqual(pairs[1]["logits"][0].argmax().item(), 49)
        self.assertEqual(torch.isfinite(pairs[0]["logits"]).sum().item(), 9)
        self.assertEqual(torch.isfinite(pairs[1]["logits"]).sum(1).tolist(), [25, 25])
        self.assertEqual(pairs[1]["same_cell"].tolist(), [False, True])
        self.assertAlmostEqual(correspondence.endpoint_loss(pairs).item(), math.log(25) / 4, places=6)
        empty = correspondence.endpoint_logits(features[1:2], cells[1:2])
        zero = correspondence.endpoint_loss(empty)
        self.assertEqual(zero.item(), 0.)
        zero.backward()
        self.assertEqual(features.grad.abs().sum().item(), 0.)

        # 控制必须保留同样候选，但不能偷偷使用当前实例特征。
        features = torch.randn(3, 3, 3, 7, 9, generator=torch.Generator().manual_seed(7),
                               requires_grad=True)
        relation_loss = correspondence.endpoint_loss(correspondence.endpoint_logits(features, cells))
        relation_gradient = torch.autograd.grad(relation_loss, features)[0]
        self.assertGreater(relation_gradient[:, 2].abs().sum().item(), 0.)
        w = torch.tensor([1., 0., 0.], requires_grad=True)
        appearance = correspondence.endpoint_logits(features, cells, w)
        self.assertEqual(appearance[0]["target"].tolist(), [13])
        self.assertEqual(torch.isfinite(appearance[0]["logits"]).sum().item(), 9)
        appearance_loss = correspondence.endpoint_loss(appearance)
        feature_gradient, w_gradient = torch.autograd.grad(appearance_loss, (features, w))
        self.assertEqual(feature_gradient[:, 2].abs().sum().item(), 0.)
        self.assertGreater(feature_gradient[:, :2].abs().sum().item(), 0.)
        self.assertGreater(w_gradient.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(feature_gradient).all())


if __name__ == "__main__":
    unittest.main()
