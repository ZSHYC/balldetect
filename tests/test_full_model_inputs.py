import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)

SCRIPT = Path(__file__).parents[1] / "scripts/train_tennis_heatmap.py"
spec = importlib.util.spec_from_file_location("train_tennis_heatmap", SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
model_input, predict = trainer.model_input, trainer.predict


class FakeModel(torch.nn.Module):
    def forward(self, pixels):
        assert pixels.dtype == torch.float32 and pixels.shape == (1, 9, 1, 1)
        logits = torch.zeros((1, 1, 2, 4))
        logits[0, 0, 1, 1] = 2
        return {0: logits}


class CountingPrefix(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frames_seen = 0

    def forward(self, pixels):
        self.frames_seen += len(pixels)
        return pixels[:, :1, :1, :1]


class OrderHead(torch.nn.Module):
    def forward(self, features):
        logits = torch.full((len(features), 9), -20., dtype=features.dtype)
        logits[:, :3] = features[:, :, 0, 0]
        logits[:, -1] = -1
        return logits


class FullModelInputTest(unittest.TestCase):
    def test_hrnet_and_dino_keep_distinct_input_and_prediction_contracts(self):
        rgb = np.asarray([
            [[[0]], [[0]], [[0]]],
            [[[255]], [[255]], [[255]]],
            [[[128]], [[64]], [[32]]],
        ], dtype=np.uint8)
        windows = np.asarray([[0, 1, 2]])
        ids = np.asarray([0])

        dino_input = model_input(rgb, windows, ids, torch.device("cpu"), "dino")
        self.assertEqual(dino_input.dtype, torch.uint8)
        self.assertEqual(tuple(dino_input.shape), (1, 3, 3, 1, 1))
        self.assertTrue(torch.equal(dino_input, torch.from_numpy(rgb)[None]))

        hrnet_input = model_input(rgb, windows, ids, torch.device("cpu"), "hrnet")
        self.assertEqual(tuple(hrnet_input.shape), (1, 9, 1, 1))
        expected = torch.tensor([[-2.117904, -2.035714, -1.804444,
                                   2.248908, 2.428571, 2.640000,
                                   0.074065, -0.915266, -1.246710]])
        self.assertTrue(torch.allclose(hrnet_input[:, :, 0, 0], expected, atol=1e-5))

        hrnet_xy, hrnet_presence = predict(
            FakeModel(), rgb, windows, ids, 1, torch.device("cpu"), "hrnet", (2, 4))
        np.testing.assert_allclose(hrnet_xy, [[479.5, 539.5]])
        np.testing.assert_allclose(hrnet_presence, [0.880797], atol=1e-6)

    def test_dino_predict_encodes_unique_frames_and_restores_window_order(self):
        rgb = np.zeros((4, 3, 1, 1), dtype=np.uint8)
        rgb[:, 0, 0, 0] = [10, 250, 100, 200]
        windows = np.asarray([[3, 0, 2], [2, 1, 3]])
        ids = np.asarray([0, 1])
        prefix = CountingPrefix()
        model = trainer.BackboneProbe(prefix, OrderHead(), train_backbone=False)

        direct_logits = model(model_input(rgb, windows, ids, torch.device("cpu"), "dino"))
        direct_xy = trainer.grid_to_original(direct_logits[:, :-1].argmax(1).numpy(), (2, 4))
        direct_presence = (1 - direct_logits.softmax(1)[:, -1]).numpy()
        self.assertEqual(prefix.frames_seen, 6)
        prefix.frames_seen = 0

        xy, presence = predict(
            model, rgb, windows, ids, 2, torch.device("cpu"), "dino", (2, 4))
        self.assertEqual(prefix.frames_seen, 4)
        np.testing.assert_allclose(xy, direct_xy)
        np.testing.assert_allclose(presence, direct_presence)
        np.testing.assert_allclose(xy, [[159.5, 179.5], [479.5, 179.5]])


if __name__ == "__main__":
    unittest.main()
