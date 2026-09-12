"""空间配对方向须为same→cross，位置救回不自动等于输出救回。"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_blurball_spatial import compare_group


def check():
    rows = [dict(visibility_raw=v, x_raw=10., y_raw=10.) for v in (1, 1, 1, 0)]
    same_xy = np.array([[20., 10.], [10., 10.], [10., 10.], [0., 0.]])
    cross_xy = np.array([[10., 10.], [20., 10.], [10., 10.], [0., 0.]])
    same_q = np.ones(4)
    cross_q = np.array([0., 1., 1., 0.])
    result = compare_group(rows, same_xy, same_q, cross_xy, cross_q, np.ones(4, dtype=bool))
    raw = result['paired_raw']['4']
    assert (raw['n'], raw['rescued'], raw['broken'], raw['net_cross_pck_change']) == (3, 1, 1, 0.)
    decisions = result['paired_decisions']
    assert decisions['matrix_axes'] == {'rows': 'same_address', 'columns': 'cross_address'}
    assert decisions['v1']['4'] == [[1, 1, 0, 0], [0, 0, 1, 0], [0]*4, [0]*4]
    assert decisions['v0'] == [[0, 1], [0, 0]]
    assert result['same_address']['detection4']['tp'] == 2
    assert result['cross_address']['detection4']['tp'] == 1
    print('PASS: same→cross direction, rejected raw rescue, V0 suppression, full V1 denominator')


if __name__ == '__main__':
    check()
