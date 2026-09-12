import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/compare_blurball_temporal.py"


class BlurBallTemporalComparisonTest(unittest.TestCase):
    def test_true_history_groups_strict_radii_and_decoder_q(self):
        self.assertTrue(SCRIPT.exists(), "BlurBall temporal comparison entry point is missing")
        spec = importlib.util.spec_from_file_location("blurball_temporal", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        frames = [
            {"game": "match18", "clip": "001", "original_frame_id": 0,
             "visibility_raw": 0, "x_raw": -1, "y_raw": -1, "l_raw": 0},
            {"game": "match18", "clip": "001", "original_frame_id": 1,
             "visibility_raw": 1, "x_raw": 0, "y_raw": 0, "l_raw": 1},
            {"game": "match18", "clip": "001", "original_frame_id": 2,
             "visibility_raw": 1, "x_raw": 12, "y_raw": 0, "l_raw": 11},
            {"game": "match18", "clip": "001", "original_frame_id": 3,
             "visibility_raw": 1, "x_raw": 16, "y_raw": 0, "l_raw": 6},
            {"game": "match18", "clip": "001", "original_frame_id": 4,
             "visibility_raw": 0, "x_raw": -1, "y_raw": -1, "l_raw": 20},
            {"game": "match19", "clip": "007", "original_frame_id": 0,
             "visibility_raw": 1, "x_raw": 1, "y_raw": 1, "l_raw": 3},
            {"game": "match19", "clip": "007", "original_frame_id": 1,
             "visibility_raw": 0, "x_raw": -1, "y_raw": -1, "l_raw": 0},
            {"game": "match19", "clip": "007", "original_frame_id": 2,
             "visibility_raw": 1, "x_raw": 2, "y_raw": 1, "l_raw": 0},
        ]
        windows = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7]])
        rows = [frames[window[-1]] for window in windows]
        history_xy = np.array([[8.1, 0], [20, 0], [100, 100], [2, 1]])
        repeat_xy = np.array([[8, 0], [19.9, 0], [200, 200], [2, 1]])
        history_q = np.array([.5, .9, .6, .49])
        repeat_q = np.array([.49, .8, .4, .7])
        decoders = {
            "argmax": {"history": (history_xy, history_q),
                       "repeat": (repeat_xy, repeat_q)},
            "local_readout": {"history": (history_xy + .1, history_q.copy()),
                              "repeat": (repeat_xy + .1, repeat_q.copy())},
        }

        result = module.compare(rows, frames, windows, decoders)

        groups = result["decoders"]["argmax"]["groups"]
        self.assertEqual(groups["visibility/V1_h01"]["n_targets"], 1)
        self.assertEqual(groups["visibility/V1_h10"]["n_targets"], 1)
        self.assertEqual(groups["visibility/V0_h11"]["n_targets"], 1)
        self.assertEqual(groups["visibility/V0_h11"]["paired_raw"]["4"]["n"], 0)
        self.assertEqual(groups["visibility/V0_h11"]["history"]["frame_counts4"]["fp2"], 1)
        self.assertEqual(groups["visibility/V0_h11"]["repeat"]["frame_counts4"]["tn"], 1)
        self.assertEqual(groups["displacement/d1_4_16"]["n_targets"], 2)
        self.assertEqual(groups["displacement/d2_ge16"]["n_targets"], 1)
        self.assertEqual(groups["all"]["paired_raw"]["4"], {
            "n": 3, "rescued_by_history": 1, "broken_by_history": 1,
            "both_correct": 1, "both_wrong": 0, "net_history_pck_change": 0.0,
        })
        self.assertEqual(groups["gt10_match/match18"]["n_targets"], 1)

        paired = groups["all"]["paired_decisions"]
        self.assertEqual(paired["matrix_axes"], {"rows": "repeat_current", "columns": "history"})
        self.assertEqual(paired["v1_state_order"], [
            "correct_emitted", "wrong_emitted", "correct_rejected", "wrong_rejected"])
        self.assertEqual(paired["v1"]["4"], [
            [0, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(paired["v1"]["8"], [
            [1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(paired["v0_state_order"], ["emitted", "rejected"])
        self.assertEqual(paired["v0"], [[0, 0], [1, 0]])
        invisible = groups["visibility/V0_h11"]["paired_decisions"]
        self.assertEqual(invisible["v1"]["4"], np.zeros((4, 4), dtype=int).tolist())
        for radius in (4, 8, 16):
            matrix = np.array(paired["v1"][str(radius)])
            for model, margin in (("history", matrix.sum(axis=0)),
                                  ("repeat", matrix.sum(axis=1))):
                counts = groups["all"][model][f"frame_counts{radius}"]
                self.assertEqual(margin[0], counts["tp"])
                self.assertEqual(margin[1], counts["fp1"])
                self.assertEqual(margin[2:].sum(), counts["fn_visible"])

        broken = {name: {model: (xy.copy(), q.copy()) for model, (xy, q) in values.items()}
                  for name, values in decoders.items()}
        broken["local_readout"]["history"][1][0] = .1
        with self.assertRaisesRegex(ValueError, "history.*q"):
            module.compare(rows, frames, windows, broken)
        wrong_rows = rows.copy()
        wrong_rows[0] = frames[3]
        with self.assertRaisesRegex(ValueError, "末帧"):
            module.compare(wrong_rows, frames, windows, decoders)


if __name__ == "__main__":
    unittest.main()
