"""比较冻结候选残差三条件，复用已保存预测，不重跑网络。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from compare_blurball_temporal import (
    _group_masks, _validate_saved_rows, continuous_windows, read_predictions,
)
from rerank_blurball_candidates import groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT/'outputs/blurball/dino_cross_address_seed0/config.json').read_text())
    metadata = json.loads((ROOT/config['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    legal, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = np.array([w for w in legal if frames[w[-1]]['split'] == 'val'])
    rows = [frames[w[-1]] for w in windows]
    source = ROOT/'outputs/blurball/spatial_interaction/candidate_coverage'
    with np.load(source/'cross_address.npz') as f:
        np.testing.assert_array_equal(f['current_frame_ids'], windows[:, -1])
        xy = {'cross': f['local_xy'][:, 0]}
        q = f['q']
    epochs = {}
    for condition in ('current', 'stationary', 'correspondence'):
        path = args.root/condition
        result = json.loads((path/'results.json').read_text())
        assert result['completed_epochs'] == 30 and result['best_reproduced']
        assert result['config']['condition'] == condition and result['config']['source_epoch'] == 3
        saved, xy[condition], confidence = read_predictions(path/'val_predictions.csv')
        _validate_saved_rows(rows, saved, path/'val_predictions.csv')
        np.testing.assert_array_equal(confidence, q)
        epochs[condition] = result['best_epoch']
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    visible21 = np.array([r['game'] == 'match21' and r['visibility_raw'] == 1 for r in rows])
    with np.load(source/'same_address.npz') as f:
        np.testing.assert_array_equal(f['current_frame_ids'], windows[:, -1])
        same_error = np.linalg.norm(f['local_xy'][:, 0]-gt, axis=1)
    cross_error = np.linalg.norm(xy['cross']-gt, axis=1)
    masks = _group_masks(rows, frames, windows)
    masks['fixed/match21_lost16'] = visible21 & (same_error < 16) & (cross_error >= 16)
    masks['fixed/match21_rescued16'] = visible21 & (same_error >= 16) & (cross_error < 16)
    assert masks['fixed/match21_lost16'].sum() == 202 and masks['fixed/match21_rescued16'].sum() == 74
    result = {'protocol': 'blurball-candidate-residual-v1', 'targets': len(rows), 'best_epochs': epochs,
              'comparisons': {}}
    for before, after in [('cross', 'current'), ('cross', 'stationary'), ('cross', 'correspondence'),
                          ('current', 'correspondence'), ('stationary', 'correspondence')]:
        result['comparisons'][f'{before}_to_{after}'] = groups(
            rows, xy[after], q, xy[before], masks, before=before, after=after)
    (args.root/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(epochs))


if __name__ == '__main__':
    main()
