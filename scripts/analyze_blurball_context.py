"""合法多帧与前瞻支持诊断；仅训练GT拟合，排除目标位置，不解码或训练。"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ballmotion.blurball import position_summary, temporal_windows

CONTEXTS = {'causal3': (-2, -1, 0), 'center3': (-1, 0, 1),
            'causal5': (-4, -3, -2, -1, 0), 'center5': (-2, -1, 0, 1, 2),
            'center7': (-3, -2, -1, 0, 1, 2, 3),
            'center9': (-4, -3, -2, -1, 0, 1, 2, 3, 4)}


def fit_from_context(xy, times, target_slot, degree):
    """每个窗口仅以非目标时刻拟合二维多项式，在目标真实时间求值。"""
    support = np.arange(times.shape[1]) != target_slot
    if int(support.sum()) < degree + 1:
        raise ValueError('非目标观测不足以拟合指定阶数')
    relative = times[:, support] - times[:, target_slot:target_slot + 1]
    relative = relative / np.max(np.abs(relative), axis=1, keepdims=True)
    design = np.stack([relative**power for power in range(degree + 1)], axis=-1)
    coefficients = np.linalg.pinv(design) @ xy[:, support]
    return coefficients[:, 0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-cache', type=Path,
                        default=ROOT / 'data/cache/blurball/rgb_512x288_all_h2')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/blurball/context_feasibility')
    args = parser.parse_args()
    metadata = json.loads((args.rgb_cache / 'metadata.json').read_text())
    frames = metadata['frames']
    with (ROOT / 'configs/blurball_continuity_boundaries.csv').open(newline='') as f:
        boundaries = list(csv.DictReader(f))
    targets = np.asarray(metadata['windows'], dtype=np.int64)[:, -1]
    split = np.array([r['split'] for r in frames])
    visible = np.array([r['visibility_raw'] == 1 for r in frames])
    xy = np.array([[r['x_raw'], r['y_raw']] for r in frames], dtype=float)
    times = np.array([r['pts_seconds'] for r in frames], dtype=float)
    lengths = np.array([r['l_raw'] for r in frames], dtype=float)
    windows = {}
    report = {'scope': 'metadata coverage train/val; GT fits train only, target position excluded',
              'source': str(args.rgb_cache), 'contexts': {}, 'fits': {}}
    for name, offsets in CONTEXTS.items():
        current, excluded = temporal_windows(frames, targets, offsets, boundaries)
        slot = offsets.index(0)
        ids = current[:, slot]
        windows[name] = current
        report['contexts'][name] = {'offsets': offsets, 'target_slot': slot,
            'natural_targets': {s: int((split[ids] == s).sum()) for s in ('train', 'val')},
            'excluded_targets': {s: int((split[excluded] == s).sum()) for s in ('train', 'val')}}
    primary = np.intersect1d(windows['causal5'][:, -1], windows['center5'][:, 2])
    common = primary.copy()
    for name, offsets in CONTEXTS.items():
        common = np.intersect1d(common, windows[name][:, offsets.index(0)])
    report['five_frame_common_targets'] = {s: int((split[primary] == s).sum())
                                          for s in ('train', 'val')}
    report['all_context_common_targets'] = {s: int((split[common] == s).sum())
                                           for s in ('train', 'val')}
    saved = {'five_frame_common_targets': primary, 'all_context_common_targets': common}

    for cohort, ids, names in (('five_frame_pair', primary, ('causal5', 'center5')),
                              ('all_contexts', common, tuple(CONTEXTS))):
        ids = ids[split[ids] == 'train']
        selected = {}
        joint_visible = visible[ids].copy()
        availability = {}
        for name in names:
            offsets = CONTEXTS[name]
            slot = offsets.index(0)
            by_target = {int(w[slot]): w for w in windows[name]}
            current = np.asarray([by_target[int(i)] for i in ids])
            selected[name] = current
            support = np.arange(len(offsets)) != slot
            joint_visible &= visible[current].all(axis=1)
            v = visible[current[:, support]]
            target_v = visible[ids]
            span = times[current[:, -1]] - times[current[:, 0]]
            availability[name] = {
                'current_v1': int(target_v.sum()),
                'current_v1_no_v1_support': int((target_v & ~v.any(axis=1)).sum()),
                'current_v1_all_v1_support': int((target_v & v.all(axis=1)).sum()),
                'span_seconds_median': float(np.median(span)),
                'span_seconds_p90': float(np.percentile(span, 90))}
        fit_ids = ids[joint_visible]
        saved[f'{cohort}_fit_targets'] = fit_ids
        entries = {}
        for name, current in selected.items():
            current = current[joint_visible]
            slot = CONTEXTS[name].index(0)
            entries[name] = {}
            for degree in (1, 2):
                if current.shape[1] - 1 < degree + 1:
                    continue
                predicted = fit_from_context(xy[current], times[current], slot, degree)
                errors = np.linalg.norm(predicted - xy[fit_ids], axis=1)
                entries[name][str(degree)] = {
                    'all': position_summary(errors),
                    'l_gt10': position_summary(errors[lengths[fit_ids] > 10])}
                saved[f'{cohort}_{name}_degree{degree}'] = predicted
        report['fits'][cohort] = {'common_train_targets': len(ids),
            'joint_all_visible_targets': len(fit_ids), 'availability': availability,
            'target_excluded_gt_fits': entries}
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(args.output / 'gt_context_predictions.npz', **saved)
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
