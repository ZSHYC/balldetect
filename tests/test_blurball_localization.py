"""可见中点的原图坐标、严格容差与拒绝分母。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ballmotion.blurball import continuous_windows, evaluate_blurball, source_coordinates
from ballmotion.tennis import grid_targets


def test_localization():
    rows = [dict(game='match18', clip='001', width=1280, height=720,
                 visibility_raw=v, x_raw=10., y_raw=10., l_raw=l)
            for v, l in ((1, 0), (1, 2), (1, 6), (0, -1), (0, -1), (1, 11))]
    xy = np.array([[10, 10], [14, 10], [10, 10], [0, 0], [0, 0], [13, 10.]])
    metrics = evaluate_blurball(rows, xy, np.array([.9, .9, .1, .9, .1, .9]))
    assert metrics['location']['n'] == 4
    assert metrics['location']['pck4'] == .75
    assert metrics['emitted_location']['n'] == 3
    assert metrics['emitted_location']['pck4'] == 2/3
    assert metrics['detection4']['f1'] == .5
    assert metrics['detection4']['fn'] == 2
    assert metrics['author_detection4']['fn'] == 1
    assert np.isclose(metrics['author_detection4']['f1'], 4/7)
    assert metrics['frame_counts4'] == dict(tp=2, fp1=1, fp2=1, fn_visible=1, tn=1)
    assert metrics['by_half_length']['l0']['n_frames'] == 1
    assert sum(v['n_frames'] for v in metrics['by_half_length'].values()) == 4
    assert metrics['by_half_length']['2_5']['location']['pck4'] is None
    assert metrics['by_match_half_length']['match18']['5_10']['emitted_location']['n'] == 0

    dimensions = [dict(width=1280, height=720), dict(width=1920, height=1080),
                  dict(width=1266, height=720)]
    source = source_coordinates(np.array([[49.5, 99.5]] * 3), dimensions)
    assert np.allclose(source, [[49.5, 99.5], [74.5, 149.5], [48.953125, 99.5]])
    cells = grid_targets(source, [True] * 3, (288, 512),
                         ([r['height'] for r in dimensions], [r['width'] for r in dimensions]))
    assert cells.tolist() == [40*512+20] * 3
    assert grid_targets([[9999., 9999.]], [False], (288, 512)).tolist() == [288*512]
    frames = [dict(match='17', rally=rally, original_frame_id=i)
              for rally in ('001', '002') for i in range(6)]
    windows = np.array([[i-2, i-1, i] for i in (2, 3, 4, 5, 8, 9, 10, 11)])
    kept, removed = continuous_windows(frames, windows,
        [dict(match='17', rally='001', new_segment_start='3')])
    assert removed.tolist() == [1, 2]
    assert kept.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [7, 8, 9],
                             [8, 9, 10], [9, 10, 11]]
    try:
        continuous_windows(frames, windows,
                           [dict(match='17', rally='001', new_segment_start='33')])
    except ValueError:
        pass
    else:
        raise AssertionError('不存在的分段起点不能静默忽略')
    print('BlurBall localization check passed')


if __name__ == '__main__':
    test_localization()
