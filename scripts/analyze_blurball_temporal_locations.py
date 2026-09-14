"""保存预测是否落在可区分的邻帧球位置；标签仅用于诊断，不修正输出。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_blurball_midpoint import WINDOW_OFFSETS, prepare_training_windows


def temporal_location_masks(xy, target_xy, target_visible, support_xy, support_visible):
    """严格4px命中；相隔至少8px的标签才有互不重叠的4px邻域。"""
    error = np.linalg.norm(xy - target_xy, axis=-1)
    separation = np.linalg.norm(support_xy - target_xy[:, None], axis=-1)
    support_error = np.linalg.norm(support_xy - xy[:, None], axis=-1)
    eligible = target_visible[:, None] & support_visible & (separation >= 8)
    hits = eligible & (support_error < 4)
    return error, eligible, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run / 'config.json').read_text())
    offsets = np.array(WINDOW_OFFSETS[config['window']])
    target_slot = config['target_slot']
    if offsets[target_slot] != 0 or config['temporal_input'] != 'history':
        raise ValueError('本诊断要求真实上下文和明确中心目标槽')
    metadata = json.loads((Path(config['rgb_cache']) / 'metadata.json').read_text())
    frames = metadata['frames']
    windows, rows, slot, _ = prepare_training_windows(
        frames, metadata['windows'], config['continuity_boundaries'],
        config['window'], config['temporal_input'])
    selected = np.array([r['split'] == 'val' for r in rows])
    windows = windows[selected]
    rows = [r for r, keep in zip(rows, selected) if keep]
    candidate_dir = args.run / 'candidate_coverage'
    with np.load(candidate_dir / 'candidates.npz') as a:
        np.testing.assert_array_equal(a['current_frame_ids'], windows[:, slot])
        candidates, q = a['local_xy'], a['q']
    target = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    support_slots = np.flatnonzero(offsets != 0)
    support_rows = [[frames[int(i)] for i in w[support_slots]] for w in windows]
    support = np.array([[[r['x_raw'], r['y_raw']] for r in rs] for rs in support_rows])
    support_visible = np.array([[r['visibility_raw'] == 1 for r in rs] for rs in support_rows])
    error, eligible, hits = temporal_location_masks(
        candidates[:, 0], target, visible, support, support_visible)
    best = np.linalg.norm(candidates - target[:, None], axis=-1).min(axis=1)
    wrong = visible & (error >= 4)
    matched = wrong & hits.any(axis=1)
    game = np.array([r['game'] for r in rows])
    masks = {'all': np.ones(len(rows), dtype=bool), 'emitted': q >= .5,
             'rejected': q < .5,
             'wrong4_K16_good4': wrong & (best < 4),
             'K16_best_4_to16': visible & (best >= 4) & (best < 16),
             'K16_best_ge16': visible & (best >= 16)}
    masks.update({g: game == g for g in sorted(set(game))})
    groups = {}
    for name, mask in masks.items():
        groups[name] = {
            'visible': int(np.sum(mask & visible)),
            'wrong4': int(np.sum(mask & wrong)),
            'wrong4_with_distinct_visible_support': int(np.sum(mask & wrong & eligible.any(1))),
            'wrong4_at_distinct_support': int(np.sum(mask & matched)),
            'by_offset': {str(int(offsets[s])): int(np.sum(mask & wrong & hits[:, j]))
                          for j, s in enumerate(support_slots)},
        }
    cases = []
    for i in np.flatnonzero(matched):
        cases.append({
            **{k: rows[i][k] for k in ('game', 'clip', 'original_frame_id')},
            'current_frame_index': int(windows[i, slot]),
            'window_frame_indices': windows[i].tolist(),
            'prediction': candidates[i, 0].tolist(), 'target': target[i].tolist(),
            'q': float(q[i]), 'error_px': float(error[i]), 'K16_best_error_px': float(best[i]),
            'matched_offsets': offsets[support_slots[hits[i]]].tolist(),
        })
    result = {
        'source_run': str(args.run.resolve()),
        'source_checkpoint_epoch': json.loads((candidate_dir / 'manifest.json').read_text())['checkpoint_epoch'],
        'input_offsets': offsets.tolist(), 'target_slot': slot,
        'target_count': len(rows),
        'definition': 'V1 target; V1 support; GT separation >=8 original px; predicted support error <4px',
        'interpretation': 'geometric association only; not proof of selecting support appearance or causation',
        'groups': groups, 'cases': cases,
    }
    destination = candidate_dir / 'temporal_locations.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(destination), 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
