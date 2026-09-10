"""Oracle correspondence probe: matching and search coverage stay distinct."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from probe_tennis_correspondence import analyze_pair, summarize_groups
from compare_predictions import paired_motion_groups


class CorrespondenceProbeTest(unittest.TestCase):
    def test_motion_groups_keep_rescues_breakages_and_missing_pairs_separate(self):
        rows = [dict(game="game7", clip="Clip1", original_frame_id=str(i),
                     visibility_raw=1, x_raw=0., y_raw=0.) for i in range(4)]
        old = np.array([[100., 0.], [0., 0.], [100., 0.], [0., 0.]])
        new = np.array([[0., 0.], [100., 0.], [100., 0.], [0., 0.]])
        pairs = [dict(game="game7", clip="Clip1", current_original_frame_id=i,
                      delta=d, native_displacement_chebyshev=distance)
                 for d in (1, 2) for i, distance in ((0, 0), (1, 2), (2, 3))]
        groups = paired_motion_groups(rows, old, new, pairs)
        self.assertEqual(groups["1"]["same_cell"]["16"]["rescued"], 1)
        self.assertEqual(groups["1"]["within_search_moved"]["16"]["broken"], 1)
        self.assertEqual(groups["1"]["outside_search"]["16"]["both_wrong"], 1)
        self.assertEqual(groups["2"]["outside_search"]["16"]["n"], 0)
        self.assertIsNone(groups["2"]["outside_search"]["16"]["net_pck_change"])
        self.assertEqual(sum(g["16"]["n"] for g in groups["1"].values()), 3)

    def test_unique_descriptor_separates_matching_from_local_coverage(self):
        grid_hw, original_hw = (3, 5), (30, 50)
        cells = grid_hw[0] * grid_hw[1]
        history = torch.eye(cells).reshape(cells, *grid_hw)
        current = torch.zeros_like(history)
        query_cell, true_cell = 0, 14
        current[:, 0, 0] = history[:, 2, 4]

        result = analyze_pair(current, history, query_cell, true_cell, (49, 29), grid_hw, original_hw)

        for mode in ("raw", "spatial_mean_centered"):
            self.assertEqual(result[mode]["global"]["predicted_cell"], true_cell)
            self.assertTrue(result[mode]["global"]["pck16"])
            self.assertEqual(result[mode]["global"]["optimistic_true_cell_rank"], 1)
            self.assertEqual(result[mode]["global"]["recall_at_1"], 1)
            self.assertGreater(result[mode]["global"]["true_minus_same_coordinate_score"], 0)
            self.assertFalse(result[mode]["radius2"]["geometric_coverage"])
            self.assertFalse(result[mode]["radius2"]["pck32"])
            self.assertTrue(result[mode]["radius4"]["geometric_coverage"])
            self.assertEqual(result[mode]["radius4"]["predicted_cell"], true_cell)

        tied = analyze_pair(torch.zeros_like(current), torch.zeros_like(history),
                            query_cell, true_cell, (49, 29), grid_hw, original_hw)
        self.assertEqual(tied["raw"]["global"]["optimistic_true_cell_rank"], 1)
        self.assertEqual(tied["raw"]["global"]["recall_at_1"], 0)

        groups = summarize_groups([{
            "game": "game7", "clip": "Clip1", "query_cell": query_cell,
            "true_history_cell": true_cell, "current_visibility": 1,
            "history_visibility": 1, "results": result,
        }])
        self.assertEqual(groups["different_native_cells"]["n"], 1)
        self.assertEqual(groups["different_native_cells"]["modes"]["raw"]["radius2"]
                         ["geometric_coverage_n"], 0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
