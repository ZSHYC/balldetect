"""多帧输入按原帧号构造，不能跨缺帧、rally或已知内部边界。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ballmotion.blurball import temporal_windows


def test_center_and_causal_windows_keep_target_identity():
    frames = [dict(match='00', rally='001', original_frame_id=i,
                   visibility_raw=0 if i == 11 else 1)
              for i in range(16) if i != 4]
    # 另一rally有frame4，也不能拿来填补当前rally的缺帧。
    frames += [dict(match='00', rally='002', original_frame_id=i,
                    visibility_raw=1) for i in range(16)]
    targets = np.array([2, 5, 9, 11])  # 原frame2,6,10,12。
    boundaries = [dict(match='00', rally='001', new_segment_start='8')]
    center, excluded = temporal_windows(frames, targets, [-2, -1, 0, 1, 2], boundaries)
    assert [[frames[i]['original_frame_id'] for i in w] for w in center] == [
        [8, 9, 10, 11, 12], [10, 11, 12, 13, 14]]
    np.testing.assert_array_equal(center[:, 2], [9, 11])
    np.testing.assert_array_equal(excluded, [2, 5])
    assert frames[center[0, 3]]['visibility_raw'] == 0
    causal, excluded = temporal_windows(frames, targets, [-4, -3, -2, -1, 0], boundaries)
    assert [[frames[i]['original_frame_id'] for i in w] for w in causal] == [
        [8, 9, 10, 11, 12]]
    np.testing.assert_array_equal(causal[:, 4], [11])
    np.testing.assert_array_equal(excluded, [2, 5, 9])


def test_missing_future_does_not_cross_rally_or_shorten_window():
    frames = [dict(match='00', rally=r, original_frame_id=i)
              for r in ('001', '002') for i in range(5)]
    windows, excluded = temporal_windows(frames, [4], [-2, -1, 0, 1, 2], [])
    assert windows.shape == (0, 5)
    np.testing.assert_array_equal(excluded, [4])


if __name__ == '__main__':
    test_center_and_causal_windows_keep_target_identity()
    test_missing_future_does_not_cross_rally_or_shorten_window()
    print('BlurBall temporal windows: passed')
