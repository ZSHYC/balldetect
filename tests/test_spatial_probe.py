"""手算几何、标签和评价；阻止错位结果被算作定位成功。"""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch

from ballmotion.tennis import label_state, grid_targets, grid_to_original, read_frames, center_pairs
from ballmotion.probe import SpatialProbe, evaluate_predictions


class SpatialProbeTest(unittest.TestCase):
    def test_center_pairs_do_not_cross_clips_or_reindex_sparse_frames(self):
        rows = [dict(game="game1", clip=clip, original_frame_id=str(frame), label_state=state)
                for clip, frame, state in [("Clip1", 0, "located"), ("Clip1", 2, "located"),
                                           ("Clip2", 3, "located"), ("Clip2", 4, "absent"),
                                           ("Clip2", 5, "located")]]
        self.assertEqual(list(center_pairs(rows, 1)), [])
        pairs = list(center_pairs(rows, 2))
        self.assertEqual([(a["clip"], a["original_frame_id"], b["original_frame_id"])
                          for a, b in pairs], [("Clip1", "0", "2"), ("Clip2", "3", "5")])

    def test_published_blank_absence_coordinates_and_original_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "frames.csv").write_text(
                "game,clip,original_frame_id,visibility_raw,x_raw,y_raw,width,height\n"
                "game1,Clip1,0000,0,,,1280,720\n"
                "game1,Clip1,0001,1,10,20,1280,720\n"
                "game1,Clip2,0000,1,30,40,1280,720\n"
                "game7,Clip1,0000,1,30,40,1280,720\n")
            rows, excluded = read_frames(directory, {1}, frame_step=8)
        self.assertEqual(len(rows), 2)
        self.assertEqual(excluded, [])
        self.assertEqual(rows[0]["x_raw"], None)
        self.assertEqual(rows[0]["label_state"], "absent")
        self.assertEqual(rows[1]["clip"], "Clip2")
        self.assertEqual(rows[1]["original_frame_id"], "0000")
        np.testing.assert_array_equal(grid_targets(np.array([[np.nan, np.nan]]), [False], (2, 4)), [8])

    def test_half_pixel_grid_and_absence(self):
        # 原图 16x8 -> 网格 4x2，每个 cell 覆盖 4x4 原图像素。
        xy = np.array([[0, 0], [15, 7], [4, 4], [-1, -1]])
        target = grid_targets(xy, np.array([1, 1, 1, 0]), (2, 4), (8, 16))
        np.testing.assert_array_equal(target, [0, 7, 5, 8])
        np.testing.assert_allclose(grid_to_original(np.array([0, 7, 5]), (2, 4), (8, 16)),
                                   [[1.5, 1.5], [13.5, 5.5], [5.5, 5.5]])

    def test_unknown_and_out_of_bounds_are_not_absence(self):
        self.assertEqual(label_state(0, 0, 0, 1280, 720), "absent")
        self.assertEqual(label_state(3, 100, 200, 1280, 720), "located")
        self.assertEqual(label_state(1, 282, 720, 1280, 720), "invalid")
        self.assertEqual(label_state(1, -1, 10, 1280, 720), "invalid")
        self.assertEqual(label_state(None, None, None, 1280, 720), "unknown")

    def test_wrong_location_counts_as_fp_and_fn(self):
        rows = [dict(visibility_raw=v, x_raw=x, y_raw=y, clip="Clip1", game="game7")
                for v, x, y in [(1, 10, 10), (2, 10, 10), (1, 10, 10), (0, 0, 0)]]
        pred = np.array([[10, 10], [30, 10], [10, 10], [1, 1]])
        result = evaluate_predictions(rows, pred, np.array([.9, .9, .1, .9]))
        self.assertEqual(result["location"]["pck8"], 2 / 3)
        self.assertEqual(result["detection8"]["tp"], 1)
        self.assertEqual(result["detection8"]["fp"], 2)
        self.assertEqual(result["detection8"]["fn"], 2)
        self.assertAlmostEqual(result["detection8"]["f1"], 1 / 3)

    def test_small_feature_problem_can_be_learned(self):
        torch.manual_seed(0)
        x = torch.zeros(4, 2, 4, 4)
        x[0, 0, 1, 2] = 5
        x[1, 0, 3, 0] = 5
        x[2, 0, 0, 3] = 5
        x[3, 1] = 3  # 可区分的无球图像，不向模型泄露实际标签。
        targets = torch.tensor([6, 12, 3, 16])
        model = SpatialProbe(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        for _ in range(60):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.equal(model(x).argmax(1), targets))
        self.assertLess(float(loss.detach()), .05)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
