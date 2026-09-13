"""轨迹诊断必须只用支撑位置，不能让目标GT进入拟合。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_blurball_context import fit_from_context


def test_irregular_time_fit_excludes_target_for_both_directions():
    for times, slot in (([-.4, -.1, 0., .2, .5], 2),
                        ([-.9, -.4, -.2, -.1, 0.], 4)):
        times = np.asarray([times])
        xy = np.stack((2 + 3 * times + 4 * times**2,
                       -1 + 2 * times - .5 * times**2), axis=-1)
        np.testing.assert_allclose(fit_from_context(xy, times, slot, 2), [[2., -1.]],
                                   atol=1e-10)
        xy[:, slot] = [1000000., -2000000.]
        np.testing.assert_allclose(fit_from_context(xy, times, slot, 2), [[2., -1.]],
                                   atol=1e-10)


if __name__ == '__main__':
    test_irregular_time_fit_excludes_target_for_both_directions()
    print('Target-excluded context fit: passed')
