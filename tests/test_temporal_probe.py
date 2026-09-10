"""真实时间窗口、目标帧、未知标签上下文与重复帧对照。"""
import unittest

import numpy as np
import torch

from ballmotion.tennis import causal_windows
from ballmotion.probe import frame_batch, SpatialProbe, paired_location_changes


class TemporalProbeTest(unittest.TestCase):
    def test_paired_analysis_counts_rescues_and_new_errors(self):
        rows = [dict(game="game7", clip="Clip1", original_frame_id=str(i),
                     visibility_raw=1, x_raw=0., y_raw=0.) for i in range(4)]
        baseline = np.array([[0., 0.], [100., 0.], [0., 0.], [100., 0.]])
        challenger = np.array([[0., 0.], [0., 0.], [100., 0.], [100., 0.]])
        result = paired_location_changes(rows, baseline, challenger)["all"]["16"]
        self.assertEqual(result["rescued"], 1)
        self.assertEqual(result["broken"], 1)
        self.assertEqual(result["net_pck_change"], 0.)
        self.assertEqual(result["both_wrong"], 1)

    def test_causal_windows_keep_real_ids_and_do_not_use_context_labels(self):
        rows = [dict(game="game1", clip=clip, original_frame_id=str(frame), label_state=state)
                for clip, frame, state in [("Clip1", 0, "located"), ("Clip1", 1, "unknown"),
                                           ("Clip1", 2, "located"), ("Clip1", 4, "located"),
                                           ("Clip2", 0, "located"), ("Clip2", 1, "absent"),
                                           ("Clip2", 2, "located")]]
        windows, skipped = causal_windows(rows, target_step=2, history=2)
        np.testing.assert_array_equal(windows, [[0, 1, 2], [4, 5, 6]])
        self.assertEqual(skipped, 3)  # 两个 clip 开头和缺少真实 frame3 的目标4。

    def test_batch_layout_and_repeat_control(self):
        array = np.arange(12).reshape(6, 2, 1, 1)
        windows = np.array([[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(frame_batch(array, windows, "current")[:, :, 0, 0], [[4, 5], [10, 11]])
        np.testing.assert_array_equal(frame_batch(array, windows, "stack")[:, :, 0, 0],
                                      [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        np.testing.assert_array_equal(frame_batch(array, windows, "repeat")[:, :, 0, 0],
                                      [[4, 5, 4, 5, 4, 5], [10, 11, 10, 11, 10, 11]])

    def test_each_frame_normalizes_independently(self):
        # 历史帧的均值/尺度改变不应改变当前帧的归一化值。
        head = SpatialProbe(6, num_frames=3)
        x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 2, 2)
        changed = x.clone()
        changed[:, :2] = changed[:, :2] * 100 + 500
        torch.testing.assert_close(head.norm(x)[:, -2:], head.norm(changed)[:, -2:])
        logits = head(x)
        self.assertEqual(logits.shape, (1, 5))
        logits.sum().backward()
        self.assertTrue(all(p.grad is not None for p in head.parameters()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
