"""复用原三帧对应与单帧预测，检验历史球身份支持，不重跑网络。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from compare_blurball_temporal import (
    _group_masks, _identity, _validate_saved_rows, continuous_windows, read_predictions,
)
from rerank_blurball_candidates import groups
from train_tennis_heatmap import write_predictions


def history_predictions(frames, windows, saved):
    """以合法窗口的源身份查单帧输出；槽顺序near, far，缺失不补帧。"""
    lookup = {_identity(row): row for row in saved}
    assert len(lookup) == len(saved), '单帧预测身份重复'
    xy = np.zeros((len(windows), 2, 2))
    q = np.zeros((len(windows), 2))
    available = np.zeros((len(windows), 2), dtype=bool)
    for i, window in enumerate(windows):
        for slot, frame in enumerate(window[-2::-1]):
            row = lookup.get(_identity(frames[frame]))
            if row is not None:
                xy[i, slot] = row['pred_x'], row['pred_y']
                q[i, slot] = row['presence_probability']
                available[i, slot] = True
    return xy, q, available


def supported_candidates(cells, grid_hw, dimensions, history_xy, history_q, available):
    """先取已有全局匹配，再核对历史单帧球；保留原候选优先级与缺失回退。"""
    height, width = grid_hw
    points = (np.stack((cells % width, cells // width), axis=-1)+.5)
    points = points * dimensions[:, :, None, :]/[width, height]-.5
    support = ((np.linalg.norm(points-history_xy[:, :, None, :], axis=-1) < 16)
               & (available & (history_q >= .5))[:, :, None])
    both = support.all(axis=1)
    # 无支持时argmax自然返回0；有多个支持时优先原排序最前者。
    return support, both.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有历史球证据诊断')
    started = time.perf_counter()
    base = ROOT/'outputs/blurball/spatial_interaction'
    cross_run = ROOT/'outputs/blurball/dino_cross_address_seed0'
    single_run = ROOT/'outputs/blurball/dino_repeat_current_seed0'
    config = json.loads((cross_run/'config.json').read_text())
    single_config = json.loads((single_run/'config.json').read_text())
    assert single_config['temporal_input'] == 'repeat_current'
    assert single_config['rgb_cache'] == config['rgb_cache']
    assert json.loads((single_run/'results.json').read_text())['best_epoch'] == 1
    manifest = json.loads((base/'candidate_correspondence/results.json').read_text())
    assert Path(manifest['source_run']).resolve() == cross_run and manifest['checkpoint_epoch'] == 3
    assert manifest['levels']['stage0']['grid_hw'] == [72, 128]
    metadata = json.loads((ROOT/config['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    legal, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = np.array([w for w in legal if frames[w[-1]]['split'] == 'val'])
    rows = [frames[w[-1]] for w in windows]
    saved, _, _ = read_predictions(single_run/'val_predictions.csv')
    _validate_saved_rows(rows, saved, single_run/'val_predictions.csv')
    assert len(rows) == 14192
    with np.load(base/'candidate_coverage/cross_address.npz') as data:
        np.testing.assert_array_equal(data['current_frame_ids'], windows[:, -1])
        xy, q, scores = data['local_xy'], data['q'], data['peak_logits']
    assert np.all(scores[:, :-1] >= scores[:, 1:]), '原候选必须按peak logit排序'
    with np.load(base/'candidate_correspondence/stage0.npz') as data:
        np.testing.assert_array_equal(data['current_frame_ids'], windows[:, -1])
        cells = data['matched_cells']
    history_xy, history_q, available = history_predictions(frames, windows, saved)
    old = [[frames[w[1]], frames[w[0]]] for w in windows]
    dimensions = np.array([[[r['width'], r['height']] for r in rs] for rs in old])
    support, selected = supported_candidates(cells, (72, 128), dimensions, history_xy, history_q, available)
    predicted = xy[np.arange(len(rows)), selected]
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    error = np.linalg.norm(xy-gt[:, None], axis=-1)
    ball = error.argmin(1)
    covered = visible & (error.min(1) < 4)
    old_gt = np.array([[[r['x_raw'], r['y_raw']] for r in rs] for rs in old])
    old_visible = np.array([[r['visibility_raw'] == 1 for r in rs] for rs in old])
    old_error = np.linalg.norm(history_xy-old_gt, axis=-1)
    masks = _group_masks(rows, frames, windows)
    with np.load(base/'candidate_coverage/same_address.npz') as data:
        np.testing.assert_array_equal(data['current_frame_ids'], windows[:, -1])
        same_error = np.linalg.norm(data['local_xy'][:, 0]-gt, axis=-1)
    match21 = np.array([r['game'] == 'match21' for r in rows]) & visible
    masks['fixed/match21_lost16'] = match21 & (same_error < 16) & (error[:, 0] >= 16)
    masks['fixed/match21_rescued16'] = match21 & (same_error >= 16) & (error[:, 0] < 16)
    assert masks['fixed/match21_lost16'].sum() == 202
    assert masks['fixed/match21_rescued16'].sum() == 74
    both_available = available.all(1)
    both_emitted = (available & (history_q >= .5)).all(1)
    masks.update({'history/both_available_emitted': both_emitted,
                  'history/prediction_missing': ~both_available,
                  'history/available_but_rejected': both_available & ~both_emitted})
    result = {'protocol': 'blurball-history-ball-evidence-v1',
        'sources': {'current': str(cross_run), 'current_epoch': 3,
                    'history': str(single_run/'val_predictions.csv'), 'history_epoch': 1,
                    'history_readout': 'original argmax',
                    'correspondence': str(base/'candidate_correspondence/stage0.npz')},
        'visual_support': ['t-2', 't-1', 't'], 'target_count': len(rows),
        'history_counts': {'both_available': int(both_available.sum()),
                           'both_available_emitted': int(both_emitted.sum())},
        'rule': 'earliest original-ranked candidate supported by both histories; else top1',
        'fixed_distance_px': 16, 'fixed_history_q': .5,
        'changed_targets': int(np.sum(selected != 0)),
        'any_joint_supported_targets': int(support.all(1).any(1).sum()),
        'groups': groups(rows, predicted, q, xy[:, 0], masks, after='history_ball'),
        'history_diagnostics': {},
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['scripts/probe_blurball_history_ball_evidence.py']}
    idx = np.arange(len(rows))
    for slot in range(2):
        diagnostic = {}
        for name, mask in masks.items():
            if not mask.any():
                continue
            hv = mask & old_visible[:, slot]
            usable = hv & available[:, slot]
            emitted = usable & (history_q[:, slot] >= .5)
            pair = mask & covered & (error[:, 0] >= 16) & old_visible[:, slot]
            correct_support = support[idx, slot, ball]
            wrong_support = support[:, slot, 0]
            # 矩阵行=球query是否支持，列=远错top1是否支持，0/1均按False/True。
            table = np.bincount((2*correct_support+wrong_support)[pair], minlength=4).reshape(2, 2)
            diagnostic[name] = {
                'window_weighted_history_visible_n': int(hv.sum()),
                'prediction_available_n': int((mask & available[:, slot]).sum()),
                'history_v1_prediction_available_n': int(usable.sum()),
                'history_v1_emitted_n': int(emitted.sum()),
                **{f'history_v1_available_correct{r}_n': int(np.sum(usable & (old_error[:, slot] < r)))
                   for r in (4, 16)},
                'history_v1_emitted_correct16_n': int(np.sum(emitted & (old_error[:, slot] < 16))),
                'pair_n': int(pair.sum()),
                'pair_prediction_available_n': int(np.sum(pair & available[:, slot])),
                'pair_prediction_emitted_n': int(np.sum(pair & available[:, slot] & (history_q[:, slot] >= .5))),
                'support_table_ball_rows_wrong_columns': table.tolist(),
                'pair_history_correct16_emitted_n': int(np.sum(pair & emitted & (old_error[:, slot] < 16)))}
        result['history_diagnostics'][str(slot+1)] = diagnostic
    args.output.mkdir(parents=True)
    np.savez(args.output/'evidence.npz', current_frame_ids=windows[:, -1],
             history_xy=history_xy, history_q=history_q, available=available,
             support=support, selected=selected.astype(np.uint8))
    write_predictions(args.output/'val_predictions.csv', rows, predicted, q)
    result['cpu_seconds_including_load_and_statistics'] = time.perf_counter()-started
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('changed_targets', 'history_counts', 'cpu_seconds_including_load_and_statistics')}, indent=2))


if __name__ == '__main__':
    main()
