"""固定候选与q上的因果短路径重排；仅训练侧选强度，不更新模型。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from compare_blurball_temporal import _group, _group_masks, continuous_windows
from train_tennis_heatmap import write_predictions

STRENGTHS = (0., 1/16, 1/4, 1., 4.)


def history_indices(rows):
    """仅连接同片段真实相邻原帧；输入已是通过内部边界过滤的合法目标。"""
    seen = {}
    links = np.full((len(rows), 2), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        game, clip, frame = row['game'], row['clip'], int(row['original_frame_id'])
        far = seen.get((game, clip, frame-2))
        near = seen.get((game, clip, frame-1))
        if (far is not None and near is not None
                and rows[far]['pts_seconds'] < rows[near]['pts_seconds'] < row['pts_seconds']):
            links[i] = far, near
        seen[game, clip, frame] = i
    return links


def rerank(xy, scores, q, rows, links, mode, strengths):
    """仅消费坐标、分数、q与真实时间；返回每个强度选中的候选序号。"""
    if mode not in ('stationary', 'velocity'):
        raise ValueError(mode)
    selected = np.zeros((len(strengths), len(rows)), dtype=np.int64)
    strengths = np.asarray(strengths, dtype=np.float64)
    active = np.flatnonzero(strengths != 0)
    if not len(active):
        return selected
    strengths = strengths[active]
    relative = np.asarray(scores, dtype=np.float64) - scores[:, :1]
    times = np.array([r['pts_seconds'] for r in rows])
    for i, (far, near) in enumerate(links):
        if far < 0 or near < 0 or q[far] < .5 or q[near] < .5:
            continue
        if not times[far] < times[near] < times[i]:
            raise ValueError('相邻原帧PTS时间必须递增')
        x, y, z = xy[i, :, None, None], xy[near, None, :, None], xy[far, None, None, :]
        if mode == 'velocity':
            ratio = (times[i]-times[near])/(times[near]-times[far])
            penalty = np.sum((x-y-ratio*(y-z))**2, axis=-1)/16
        else:
            penalty = (np.sum((x-y)**2, axis=-1)+np.sum((x-z)**2, axis=-1))/32
        past = relative[near, :, None] + relative[far, None, :]
        support = (past[None, None]-strengths[:, None, None, None]*penalty[None]).max(axis=(2, 3))
        selected[active, i] = (relative[i][None]+support).argmax(axis=1)
    return selected


def select_strength(rows, xy, q, choices, strengths):
    """训练侧TP4、全部可见correct4、较小强度，按顺序选优。"""
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    records = []
    for strength, ids in zip(strengths, choices):
        prediction = xy[np.arange(len(rows)), ids]
        correct = visible & (np.linalg.norm(prediction-gt, axis=1) < 4)
        records.append({'strength': float(strength), 'tp4': int(np.sum(correct & (q >= .5))),
                        'raw_correct4': int(correct.sum()), 'changed': int(np.sum(ids != 0))})
    best = max(range(len(records)), key=lambda j: (
        records[j]['tp4'], records[j]['raw_correct4'], -records[j]['strength']))
    return records[best]['strength'], records


def groups(rows, xy, q, baseline, masks, before='top1', after='reranked'):
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        group = _group(rows, xy, q, baseline, q, mask)
        group[after] = group.pop('history')
        group[before] = group.pop('repeat')
        group['paired_decisions']['matrix_axes'] = {'rows': before, 'columns': after}
        group['paired_raw'] = {radius: {
            key.replace('by_history', f'by_{after}').replace('net_history', f'net_{after}'): value
            for key, value in counts.items()} for radius, counts in group['paired_raw'].items()}
        result[name] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--train-candidates', type=Path, required=True)
    parser.add_argument('--val-candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有时序重排结果')
    config = json.loads((args.run/'config.json').read_text())
    metadata = json.loads((ROOT/config['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    all_windows, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = {split: np.array([w for w in all_windows if frames[w[-1]]['split'] == split])
               for split in ('train', 'val')}
    rows = {split: [frames[w[-1]] for w in ws] for split, ws in windows.items()}
    assert len(rows['train']) == 38854 and len(rows['val']) == 14192
    train_manifest = json.loads((args.train_candidates.parent/'manifest.json').read_text())
    val_manifest = json.loads((args.val_candidates.parent/'results.json').read_text())
    assert config['interaction'] == 'cross_address' and config['temporal_input'] == 'history'
    # 两份候选必须来自同一个固定模型；其余提取规则继承固定协议。
    assert val_manifest['models']['cross_address']['checkpoint_epoch'] == 3
    assert train_manifest['checkpoint_epoch'] == 3
    assert Path(train_manifest['source_run']).resolve() == args.run.resolve()
    assert Path(val_manifest['models']['cross_address']['source_run']).resolve() == args.run.resolve()
    assert Path(train_manifest['output']).resolve() == args.train_candidates.resolve()
    assert args.val_candidates.name == 'cross_address.npz', '验证候选必须来自cross'
    candidates = {}
    for split, path in (('train', args.train_candidates), ('val', args.val_candidates)):
        with np.load(path) as data:
            candidates[split] = {key: data[key] for key in
                                 ('local_xy', 'peak_logits', 'q', 'current_frame_ids')}
    for split, data in candidates.items():
        np.testing.assert_array_equal(data['current_frame_ids'], windows[split][:, -1])
        assert data['local_xy'].shape == (len(rows[split]), 16, 2)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    links = {split: history_indices(rs) for split, rs in rows.items()}
    train = candidates['train']
    selection = {'protocol': 'blurball-candidate-temporal-v1',
                 'visual_support': ['t-4', 't-3', 't-2', 't-1', 't'],
                 'source_run': str(args.run), 'train_export': train_manifest,
                 'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                 'code_changes': ['scripts/rerank_blurball_candidates.py'], 'rules': {}}
    for mode in ('stationary', 'velocity'):
        choices = rerank(train['local_xy'], train['peak_logits'], train['q'], rows['train'],
                         links['train'], mode, STRENGTHS)
        assert not choices[0].any(), '零强度必须保持top1'
        strength, trials = select_strength(rows['train'], train['local_xy'], train['q'], choices, STRENGTHS)
        selection['rules'][mode] = {'selected_strength': strength, 'train_trials': trials}
    # 在查看验证重排表现前持久化训练侧选择，验证不参与强度选择。
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2, allow_nan=False)+'\n')
    val = candidates['val']
    vr = rows['val']
    masks = _group_masks(vr, frames, windows['val'])
    eligible = np.all(links['val'] >= 0, axis=1)
    history_available = eligible.copy()
    eligible[history_available] &= np.all(val['q'][links['val'][history_available]] >= .5, axis=1)
    masks.update(history_used=eligible, history_missing=~history_available,
                 history_rejected=history_available & ~eligible)
    baseline = val['local_xy'][:, 0]
    # 固定202/74群体沿用上轮same/cross原预测，不接触候选选择。
    same = np.load(args.val_candidates.parent/'same_address.npz')
    np.testing.assert_array_equal(same['current_frame_ids'], val['current_frame_ids'])
    gt = np.array([[r['x_raw'], r['y_raw']] for r in vr])
    visible = np.array([r['visibility_raw'] == 1 for r in vr])
    covered = visible & (np.linalg.norm(val['local_xy']-gt[:, None], axis=2).min(axis=1) < 4)
    triple_covered = history_available.copy()
    triple_covered[history_available] &= covered[history_available] & np.all(
        covered[links['val'][history_available]], axis=1)
    masks.update(candidate_current_covered4=covered, candidate_triple_covered4=triple_covered)
    covered16 = visible & (np.linalg.norm(val['local_xy']-gt[:, None], axis=2).min(axis=1) < 16)
    triple_covered16 = history_available.copy()
    triple_covered16[history_available] &= covered16[history_available] & np.all(
        covered16[links['val'][history_available]], axis=1)
    masks.update(candidate_current_covered16=covered16, candidate_triple_covered16=triple_covered16)
    triple_visible = history_available.copy()
    triple_visible[history_available] &= visible[history_available] & np.all(
        visible[links['val'][history_available]], axis=1)
    residual = np.full(len(vr), np.nan)
    times = np.array([r['pts_seconds'] for r in vr])
    current = np.flatnonzero(triple_visible)
    far, near = links['val'][current].T
    ratio = (times[current]-times[near])/(times[near]-times[far])
    residual[current] = np.linalg.norm(gt[current]-gt[near]-ratio[:, None]*(gt[near]-gt[far]), axis=1)
    masks.update(gt_residual_lt4=triple_visible & (residual < 4),
                 gt_residual_4_16=triple_visible & (residual >= 4) & (residual < 16),
                 gt_residual_ge16=triple_visible & (residual >= 16))
    vis21 = np.array([r['visibility_raw'] == 1 and r['game'] == 'match21' for r in vr])
    same_error = np.linalg.norm(same['local_xy'][:, 0]-gt, axis=1)
    cross_error = np.linalg.norm(baseline-gt, axis=1)
    masks['fixed/match21_lost16'] = vis21 & (same_error < 16) & (cross_error >= 16)
    masks['fixed/match21_rescued16'] = vis21 & (same_error >= 16) & (cross_error < 16)
    assert masks['fixed/match21_lost16'].sum() == 202 and masks['fixed/match21_rescued16'].sum() == 74
    result = {'target_count': len(vr), 'history_available': int(history_available.sum()),
              'history_used': int(eligible.sum()), 'rules': {}}
    predictions = {}
    for mode, chosen in selection['rules'].items():
        ids = rerank(val['local_xy'], val['peak_logits'], val['q'], vr, links['val'],
                     mode, [chosen['selected_strength']])[0]
        xy = val['local_xy'][np.arange(len(vr)), ids]
        predictions[mode] = xy
        assert not ids[~eligible].any(), '历史不足时必须保持原预测'
        write_predictions(args.output/f'{mode}_val_predictions.csv', vr, xy, val['q'])
        result['rules'][mode] = {'strength': chosen['selected_strength'],
            'changed_targets': int(np.sum(ids != 0)),
            'groups': groups(vr, xy, val['q'], baseline, masks)}
    result['velocity_vs_stationary'] = groups(vr, predictions['velocity'], val['q'],
        predictions['stationary'], masks, before='stationary', after='velocity')
    result['cpu_selection_and_evaluation_seconds'] = time.perf_counter()-started
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {'strength': v['strength'], 'changed': v['changed_targets'],
                         'metrics': v['groups']['all']['reranked']} for k, v in result['rules'].items()}, indent=2))


if __name__ == '__main__':
    main()
