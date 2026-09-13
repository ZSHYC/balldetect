"""批内帧复用必须保持三槽顺序，以及重复当前输入的完整读出。"""
import unittest

import numpy as np
import torch

from scripts.analyze_blurball_readout import batch_readout, local_patches


class ToyModel:
    def encode(self, pixels):
        return pixels[:, :1].float()

    def head(self, features):
        spatial = features[:, 0] + 2*features[:, 1] + 3*features[:, 2]
        return torch.cat((spatial.flatten(1), torch.zeros(len(features), 1)), dim=1)


class InputProbeTest(unittest.TestCase):
    def test_slot_order_repeated_current_and_partial_batch(self):
        rgb = np.zeros((4, 3, 288, 512), dtype=np.uint8)
        for i in range(4):
            rgb[i, 0, 10+i, 20+i] = 2+i
        windows = np.array([[2, 0, 1], [1, 2, 3]])
        model = ToyModel()
        for ids in (windows, np.repeat(windows[:, -1:], 3, axis=1), windows[-1:]):
            pixels = torch.from_numpy(rgb[ids]).float()
            expected = model.head(pixels[:, :, 0])
            center, patch = local_patches(expected[:, :-1].reshape(-1, 288, 512))
            actual = batch_readout(model, rgb, ids, torch.device('cpu'))
            torch.testing.assert_close(actual[0], center, rtol=0, atol=0)
            torch.testing.assert_close(actual[1], patch, rtol=0, atol=0)
            torch.testing.assert_close(actual[2], 1-expected.softmax(1)[:, -1], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
