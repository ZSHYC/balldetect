import unittest
import importlib.util
from pathlib import Path

import numpy as np
import torch

SCRIPT = Path(__file__).parents[1] / "scripts/train_tennis_heatmap.py"
spec = importlib.util.spec_from_file_location("train_tennis_heatmap", SCRIPT)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
model_input, predict = trainer.model_input, trainer.predict


class FakeModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def forward(self, pixels):
        if self.model_name == "hrnet":
            assert pixels.dtype == torch.float32 and pixels.shape == (1, 9, 1, 1)
            logits = torch.zeros((1, 1, 2, 4))
            logits[0, 0, 1, 1] = 2
            return {0: logits}
        assert pixels.dtype == torch.uint8 and pixels.shape == (1, 3, 3, 1, 1)
        logits = torch.full((1, 9), -torch.inf)
        logits[0, 6] = 0
        logits[0, -1] = torch.log(torch.tensor(3.))
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
            FakeModel("hrnet"), rgb, windows, ids, 1, torch.device("cpu"), "hrnet", (2, 4))
        np.testing.assert_allclose(hrnet_xy, [[479.5, 539.5]])
        np.testing.assert_allclose(hrnet_presence, [0.880797], atol=1e-6)

        dino_xy, dino_presence = predict(
            FakeModel("dino"), rgb, windows, ids, 1, torch.device("cpu"), "dino", (2, 4))
        np.testing.assert_allclose(dino_xy, [[799.5, 539.5]])
        np.testing.assert_allclose(dino_presence, [.25], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
