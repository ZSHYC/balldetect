import copy
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from ballmotion.backbone_probe import BackboneProbe
from ballmotion.probe import SpatialProbe


class BackboneProbeTest(unittest.TestCase):
    def test_time_order_and_only_requested_backbone_gradients(self):
        torch.set_num_threads(2)
        torch.manual_seed(2)
        prefix = nn.Conv2d(3, 4, 8, stride=8)
        head = SpatialProbe(12, upscale=2, hidden_channels=5, num_frames=3)
        frozen = BackboneProbe(copy.deepcopy(prefix), copy.deepcopy(head), train_backbone=False)
        tuned = BackboneProbe(copy.deepcopy(prefix), copy.deepcopy(head), train_backbone=True)
        pixels = torch.randint(0, 256, (2, 3, 3, 16, 24), dtype=torch.uint8)
        mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        std = torch.tensor([.229, .224, .225])[None, :, None, None]
        by_frame = torch.cat([prefix((pixels[:, i].float() / 255 - mean) / std)
                              for i in range(3)], dim=1)
        expected = head(by_frame)
        torch.testing.assert_close(frozen(pixels), expected)
        torch.testing.assert_close(tuned(pixels), expected)
        targets = torch.tensor([1, 7])
        for model in (frozen, tuned):
            loss = nn.functional.cross_entropy(model(pixels), targets)
            loss.backward()
            self.assertTrue(all(p.grad is not None for p in model.head.parameters()))
        self.assertTrue(all(p.grad is None for p in frozen.prefix.parameters()))
        self.assertTrue(all(p.grad is not None and p.grad.abs().sum() > 0
                            for p in tuned.prefix.parameters()))


if __name__ == "__main__":
    unittest.main()
