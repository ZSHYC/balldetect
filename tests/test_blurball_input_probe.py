"""批内帧复用必须保持任意时间槽顺序，以及重复当前输入的完整读出。"""
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_blurball_readout import batch_logits, batch_readout, local_patches
from scripts.train_tennis_heatmap import build_dino_model


ROOT = Path(__file__).resolve().parents[1]


class ToyModel:
    def encode(self, pixels):
        return pixels[:, :1].float()

    def head(self, features):
        weights = torch.arange(1, features.shape[1] + 1, device=features.device)
        spatial = (features * weights[None, :, None, None]).sum(1)
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

    def test_five_slots_reuse_unique_frames_without_three_frame_reshape(self):
        rgb = np.zeros((7, 3, 288, 512), dtype=np.uint8)
        for i in range(7):
            rgb[i, 0, 20+i, 30+i] = 2+i
        windows = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
        model = ToyModel()
        pixels = torch.from_numpy(rgb[windows]).float()
        expected = model.head(pixels[:, :, 0])

        center, patch, q = batch_readout(model, rgb, windows, torch.device('cpu'))
        expected_center, expected_patch = local_patches(
            expected[:, :-1].reshape(-1, 288, 512))

        torch.testing.assert_close(center, expected_center, rtol=0, atol=0)
        torch.testing.assert_close(patch, expected_patch, rtol=0, atol=0)
        torch.testing.assert_close(q, 1-expected.softmax(1)[:, -1], rtol=0, atol=0)

    def test_real_five_frame_model_matches_direct_forward(self):
        weights = ROOT / ("models/pretrained/dinov3/lvd1689m/"
                          "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth")
        model = build_dino_model(
            weights, upscale=2, interaction="cross_address", num_frames=5).eval()
        rgb = np.random.default_rng(0).integers(
            0, 256, (7, 3, 32, 32), dtype=np.uint8)
        windows = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])

        with torch.inference_mode():
            direct = model(torch.from_numpy(rgb[windows]))
            reused = batch_logits(model, rgb, windows, torch.device('cpu'))

        torch.testing.assert_close(reused, direct)


if __name__ == '__main__':
    unittest.main()
