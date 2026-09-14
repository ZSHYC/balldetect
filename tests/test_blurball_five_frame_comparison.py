"""旧困难组的身份交集及raw/发出位置的成对计数。"""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('five_compare', ROOT/'scripts/compare_blurball_five_frame.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_legacy_identity_intersection_and_temporal_visibility():
    frames = [dict(game='match21', clip='001', original_frame_id=i,
                   visibility_raw=int(i not in (0, 1, 4, 8, 9)),
                   x_raw=10.*i, y_raw=20., l_raw=0.) for i in range(10)]
    old_rows = [frames[i] for i in (2, 3, 5, 7)]
    gt = np.array([[r['x_raw'], r['y_raw']] for r in old_rows])
    old_xy = gt.copy()
    old_xy[:2, 0] += 20
    same_xy = gt.copy()
    same_xy[2, 0] += 20
    candidates = np.stack((old_xy, gt), axis=1)
    rows = [frames[i] for i in (7, 2, 5)]
    masks, report = module.diagnostic_groups(rows, frames, old_rows, old_xy, same_xy, candidates)
    key = 'legacy/cross_bad16_K16_good4'
    assert report['legacy_cohorts'][key] == {
        'original_n': 2, 'common_n': 1, 'excluded_identities': [('match21', '001', 3)]}
    np.testing.assert_array_equal(masks[key], [False, True, False])
    assert report['group_target_counts']['context_v1/past0_future1'] == 1
    assert report['group_target_counts']['context_v1/past1_future0'] == 1
    assert report['group_target_counts']['displacement/d1_4_to16'] == 1


def test_raw_rescue_remains_separate_from_emitted_rescue():
    rows = [dict(game='match21', clip='001', original_frame_id=i,
                 visibility_raw=1, x_raw=10.*i, y_raw=20., l_raw=0.) for i in range(3)]
    before = np.array([[20., 20.], [30., 20.], [20., 20.]])
    after = np.array([[0., 20.], [10., 20.], [40., 20.]])
    q = np.array([.9, .1, .9])
    masks = {'legacy/example': np.array([False, True, False])}
    paired = module.comparison_groups(rows, before, q, after, q, masks)
    assert paired['legacy/example']['raw_position']['4']['rescued_by_center5'] == 1
    assert paired['legacy/example']['correct_emitted']['4']['rescued'] == 0
    assert paired['all']['raw_position']['4']['broken_by_center5'] == 1
    assert paired['all']['paired_decisions']['matrix_axes'] == {'rows': 'causal5', 'columns': 'center5'}

    short = module.comparison_groups(rows, before, q, after, q, masks,
                                     before_name='center5', after_name='center3')
    assert short['all']['center5'] == paired['all']['causal5']
    assert short['all']['center3'] == paired['all']['center5']
    assert short['legacy/example']['raw_position']['4']['rescued_by_center3'] == 1
    assert short['all']['paired_decisions']['matrix_axes'] == {'rows': 'center5', 'columns': 'center3'}
