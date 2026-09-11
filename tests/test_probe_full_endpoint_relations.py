from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from probe_full_endpoint_relations import aggregate, rows_from_pairs


class EndpointRelationAggregationTest(unittest.TestCase):
    def test_identity_equal_delta_macro_and_empty_group(self):
        # 批内合法pair不是连续项；错误使用position而非batch_indices会串掉目标身份。
        frames = [
            {"game": "game7", "clip": "Clip1", "original_frame_id": str(i)}
            for i in range(6)
        ]
        windows = torch.tensor([[0, 1, 2], [3, 4, 5]]).numpy()
        cells = torch.tensor([[0, 1, 2], [3, 4, 5]])
        pairs = [
            {"logits": torch.tensor([[2., 0.]]), "target": torch.tensor([1]),
             "batch_indices": torch.tensor([1]), "same_cell": torch.tensor([False])},
            {"logits": torch.tensor([[3., 0.], [0., 2.]]), "target": torch.tensor([0, 1]),
             "batch_indices": torch.tensor([0, 1]), "same_cell": torch.tensor([True, True])},
        ]
        rows = rows_from_pairs(pairs, torch.tensor([0, 1]).numpy(), windows, frames, cells)
        self.assertEqual((rows[0]["original_frame_id"], rows[0]["history_original_frame_id"]),
                         ("5", "4"))
        by_delta, macro = aggregate(rows)
        self.assertEqual(by_delta["1"]["all"]["count"], 1)
        self.assertEqual(by_delta["2"]["all"]["count"], 2)
        self.assertEqual(macro["all"]["r1"], .5)
        self.assertAlmostEqual(macro["all"]["nll"], 1.1073428462)
        self.assertIsNone(macro["moved"]["r1"])
        self.assertIsNone(macro["moved"]["nll"])
        with self.assertRaisesRegex(ValueError, "身份重复"):
            aggregate(rows + [rows[0]])


if __name__ == "__main__":
    unittest.main()
