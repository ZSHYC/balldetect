"""在共同验证目标上比较五帧因果与双向模型的保存预测。"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from ballmotion.blurball import evaluate_blurball
from compare_blurball_temporal import (
    _identity,
    _validate_saved_rows,
    paired_decisions,
    paired_raw,
    read_predictions,
)
from train_blurball_midpoint import WINDOW_OFFSETS, prepare_training_windows

RADII = (4, 8, 16)


def emitted_pairs(rows, before_xy, before_q, after_xy, after_q, mask):
    """逐半径统计真正正确且已输出的位置救回/破坏。"""
    target = np.array([[row['x_raw'], row['y_raw']] for row in rows])
    visible = np.array([row['visibility_raw'] == 1 for row in rows])
    selected = np.asarray(mask, dtype=bool) & visible
    before_error = np.linalg.norm(before_xy-target, axis=1)
    after_error = np.linalg.norm(after_xy-target, axis=1)
    result = {}
    for radius in RADII:
        before = (before_q >= .5) & (before_error < radius)
        after = (after_q >= .5) & (after_error < radius)
        result[str(radius)] = {
            'n': int(selected.sum()),
            'rescued': int(np.sum(selected & ~before & after)),
            'broken': int(np.sum(selected & before & ~after)),
            'both_correct_emitted': int(np.sum(selected & before & after)),
            'both_not_correct_emitted': int(np.sum(selected & ~before & ~after)),
        }
    return result


def q_transitions(before_q, after_q, mask):
    selected = np.asarray(mask, dtype=bool)
    before = before_q >= .5
    after = after_q >= .5
    return {
        'n': int(selected.sum()),
        'both_emitted': int(np.sum(selected & before & after)),
        'before_emitted_after_rejected': int(np.sum(selected & before & ~after)),
        'before_rejected_after_emitted': int(np.sum(selected & ~before & after)),
        'both_rejected': int(np.sum(selected & ~before & ~after)),
        'max_probability_change': (float(np.max(np.abs(after_q[selected]-before_q[selected])))
                                   if selected.any() else None),
    }


def diagnostic_groups(rows, frames, old_rows, old_xy, same_xy, candidate_xy):
    """固定旧群体按身份取交集；新增时间分组仅用标签，不参与选优。"""
    old_keys = [_identity(row) for row in old_rows]
    target_keys = [_identity(row) for row in rows]
    target_set = set(target_keys)
    gt = np.array([[row['x_raw'], row['y_raw']] for row in old_rows])
    old_v1 = np.array([row['visibility_raw'] == 1 for row in old_rows])
    old_error = np.linalg.norm(old_xy-gt, axis=1)
    same_error = np.linalg.norm(same_xy-gt, axis=1)
    coverage = (np.linalg.norm(candidate_xy-gt[:, None], axis=-1) < 4).any(1)
    match21 = np.array([row['game'] == 'match21' for row in old_rows])
    definitions = {
        'legacy/cross_bad16_K16_good4': old_v1 & (old_error >= 16) & coverage,
        'legacy/match21_same_good16_cross_bad16': old_v1 & match21 & (same_error < 16) & (old_error >= 16),
        'legacy/match21_cross_good16_same_bad16': old_v1 & match21 & (old_error < 16) & (same_error >= 16),
    }
    masks, cohorts = {}, {}
    for name, selected in definitions.items():
        members = {key for key, keep in zip(old_keys, selected) if keep}
        masks[name] = np.array([key in members for key in target_keys])
        cohorts[name] = {'original_n': len(members), 'common_n': int(masks[name].sum()),
                         'excluded_identities': sorted(members-target_set)}
        games = sorted({key[0] for key in members})
        if len(games) > 1:
            for game in games:
                masks[f'{name}/{game}'] = masks[name] & np.array([r['game'] == game for r in rows])

    # rows已经由共同五帧合法窗口生成；可直接查询其±1/±2原帧身份。
    lookup = {_identity(row): row for row in frames}
    support = {d: [lookup[(r['game'], r['clip'], int(r['original_frame_id'])+d)]
                   for r in rows] for d in (-2, -1, 1, 2)}
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    target_xy = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    previous_v1 = np.array([r['visibility_raw'] == 1 for r in support[-1]])
    previous_xy = np.array([[r['x_raw'], r['y_raw']] for r in support[-1]])
    d1 = np.linalg.norm(target_xy-previous_xy, axis=1)
    masks.update({
        'displacement/d1_lt4': visible & previous_v1 & (d1 < 4),
        'displacement/d1_4_to16': visible & previous_v1 & (d1 >= 4) & (d1 < 16),
        'displacement/d1_ge16': visible & previous_v1 & (d1 >= 16),
    })
    past = np.array([r['visibility_raw'] == 1 for r in support[-2]]) | previous_v1
    future = np.array([r['visibility_raw'] == 1 for r in support[1]]) | np.array(
        [r['visibility_raw'] == 1 for r in support[2]])
    for p in (False, True):
        for f in (False, True):
            masks[f'context_v1/past{int(p)}_future{int(f)}'] = visible & (past == p) & (future == f)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return masks, {'scope': 'exploratory diagnostics fixed before final paired predictions; not selection rules',
                   'legacy_cohorts': cohorts, 'group_target_counts': counts,
                   'context_definition': 'current V1; any V1 in t-2/t-1 versus any V1 in t+1/t+2; V0 is not proof of no physical ball'}


def comparison_groups(rows, before_xy, before_q, after_xy, after_q, extra_masks=None,
                      *, before_name='causal5', after_name='center5'):
    games = np.array([row['game'] for row in rows])
    visible = np.array([row['visibility_raw'] == 1 for row in rows])
    lengths = np.array([row['l_raw'] for row in rows], dtype=float)
    valid_length = visible & np.isfinite(lengths) & (lengths >= 0)
    masks = {'all': np.ones(len(rows), dtype=bool)}
    masks.update({f'match/{game}': games == game for game in sorted(set(games))})
    masks.update({
        'half_length/l0': valid_length & (lengths == 0),
        'half_length/0_2': valid_length & (lengths > 0) & (lengths <= 2),
        'half_length/2_5': valid_length & (lengths > 2) & (lengths <= 5),
        'half_length/5_10': valid_length & (lengths > 5) & (lengths <= 10),
        'half_length/gt10': valid_length & (lengths > 10),
    })
    if extra_masks is not None:
        masks.update(extra_masks)
    result = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        ids = np.flatnonzero(mask)
        selected_rows = [rows[index] for index in ids]
        raw = paired_raw(rows, after_xy, before_xy, mask)
        raw = {radius: {
            key.replace('by_history', f'by_{after_name}').replace('net_history', f'net_{after_name}'): value
            for key, value in counts.items()
        } for radius, counts in raw.items()}
        decisions = paired_decisions(
            selected_rows, after_xy[ids], after_q[ids], before_xy[ids], before_q[ids])
        decisions['matrix_axes'] = {'rows': before_name, 'columns': after_name}
        result[name] = {
            'n_targets': len(ids),
            before_name: evaluate_blurball(
                selected_rows, before_xy[ids], before_q[ids], grouped=False),
            after_name: evaluate_blurball(
                selected_rows, after_xy[ids], after_q[ids], grouped=False),
            'raw_position': raw,
            'correct_emitted': emitted_pairs(
                rows, before_xy, before_q, after_xy, after_q, mask),
            'q_transitions': q_transitions(before_q, after_q, mask),
            'paired_decisions': decisions,
        }
    return result


def load_arm(path, metadata):
    config = json.loads((path/'config.json').read_text())
    results = json.loads((path/'results.json').read_text())
    windows, all_rows, target_slot, _ = prepare_training_windows(
        metadata['frames'], metadata['windows'], config['continuity_boundaries'],
        config['window'], config['temporal_input'])
    if target_slot != config['target_slot']:
        raise ValueError(f'{path} target_slot与窗口不一致')
    val_ids = np.flatnonzero([row['split'] == 'val' for row in all_rows])
    rows = [all_rows[index] for index in val_ids]
    if (len(rows) != config['val_targets']
            or windows.shape[1] != len(WINDOW_OFFSETS[config['window']])):
        raise ValueError(f'{path} 上下文共同目标计数或帧数不一致')

    history = [json.loads(line) for line in (path/'history.jsonl').read_text().splitlines()]
    completed = max(record['epoch'] for record in history)
    if sorted(record['epoch'] for record in history) != list(range(completed+1)):
        raise ValueError(f'{path} epoch记录不连续')
    if completed != config['epochs']:
        raise ValueError(f'{path} 尚未完成配置轮次')

    predictions = {}
    for decoder, prediction_path in (
            ('argmax', path/'val_predictions.csv'),
            ('fixed_local', path/'local_readout/val_predictions.csv')):
        saved_rows, xy, q = read_predictions(prediction_path)
        _validate_saved_rows(rows, saved_rows, prediction_path)
        predictions[decoder] = (xy, q)
    if not np.array_equal(predictions['argmax'][1], predictions['fixed_local'][1]):
        raise ValueError(f'{path} 固定局部读出改变了q')
    if evaluate_blurball(rows, *predictions['argmax']) != results['val']:
        raise ValueError(f'{path} 保存argmax预测未复现results')
    local_results = json.loads((path/'local_readout/results.json').read_text())
    if evaluate_blurball(rows, *predictions['fixed_local']) != local_results['local_barycenter']:
        raise ValueError(f'{path} 保存局部预测未复现local results')
    return config, results, rows, predictions, completed


def main():
    output = ROOT/'outputs/blurball/five_frame_context'
    paths = {'center5': output/'center5_seed0', 'causal5': output/'causal5_seed0'}
    center_config = json.loads((paths['center5']/'config.json').read_text())
    metadata = json.loads((ROOT/center_config['rgb_cache']/'metadata.json').read_text())
    loaded = {name: load_arm(path, metadata) for name, path in paths.items()}
    configs = {name: values[0] for name, values in loaded.items()}

    expected = {
        'center5': (['t-2', 't-1', 't', 't+1', 't+2'], 2),
        'causal5': (['t-4', 't-3', 't-2', 't-1', 't'], 4),
    }
    for name, (slots, target_slot) in expected.items():
        config = configs[name]
        if (config['protocol'] != 'blurball-five-frame-context-v1'
                or config['window'] != name or config['input_slots'] != slots
                or config['target_slot'] != target_slot):
            raise ValueError(f'{name} 输入协议不一致')
    allowed_differences = {'window', 'input_slots', 'target_slot', 'output', 'code_revision'}
    shared_keys = set(configs['center5']) | set(configs['causal5'])
    unexpected = [key for key in sorted(shared_keys-allowed_differences)
                  if configs['center5'].get(key) != configs['causal5'].get(key)]
    if unexpected:
        raise ValueError(f'五帧两臂关键配置不同: {unexpected}')
    for key, value in {'epochs': 12, 'batch_size': 4, 'seed': 0,
                       'train_targets': 37590, 'val_targets': 13912,
                       'total_parameters': 1279169}.items():
        if configs['center5'][key] != value:
            raise ValueError(f'共同训练条件不符合协议: {key}')

    rows = loaded['causal5'][2]
    if ([_identity(row) for row in rows] != [_identity(row) for row in loaded['center5'][2]]
            or [(row['visibility_raw'], row['x_raw'], row['y_raw']) for row in rows]
            != [(row['visibility_raw'], row['x_raw'], row['y_raw'])
                for row in loaded['center5'][2]]):
        raise ValueError('五帧两臂共同验证身份或GT不一致')

    old_path = ROOT/'outputs/blurball/dino_cross_address_seed0/local_readout/val_predictions.csv'
    old_rows, old_xy, old_q = read_predictions(old_path)
    lookup = {_identity(row): index for index, row in enumerate(old_rows)}
    if len(lookup) != len(old_rows) or any(_identity(row) not in lookup for row in rows):
        raise ValueError('旧cross验证预测不能覆盖五帧共同目标')
    old_ids = np.array([lookup[_identity(row)] for row in rows])
    selected_old_rows = [old_rows[index] for index in old_ids]
    _validate_saved_rows(rows, selected_old_rows, old_path)

    same_path = ROOT/'outputs/blurball/dino_same_address_seed0/local_readout/val_predictions.csv'
    same_rows, same_xy, _ = read_predictions(same_path)
    _validate_saved_rows(old_rows, same_rows, same_path)
    candidate_path = ROOT/'outputs/blurball/spatial_interaction/candidate_coverage/cross_address.npz'
    with np.load(candidate_path) as candidates:
        source_rows = [metadata['frames'][index] for index in candidates['current_frame_ids']]
        _validate_saved_rows(old_rows, source_rows, candidate_path)
        candidate_xy = candidates['local_xy']
    if not np.array_equal(candidate_xy[:, 0], old_xy):
        raise ValueError('旧困难组候选0与原cross局部预测不一致')
    masks, diagnostic_metadata = diagnostic_groups(
        rows, metadata['frames'], old_rows, old_xy, same_xy, candidate_xy)
    comparisons = {}
    for decoder in ('argmax', 'fixed_local'):
        center_xy, center_q = loaded['center5'][3][decoder]
        causal_xy, causal_q = loaded['causal5'][3][decoder]
        comparisons[decoder] = comparison_groups(
            rows, causal_xy, causal_q, center_xy, center_q, masks)

    result = {
        'protocol': 'blurball-five-frame-context-v1',
        'scope': 'common validation targets only; no final test',
        'direction': 'causal5 -> center5',
        'targets': len(rows),
        'training_status': {
            name: {'best_epoch': values[1]['best_epoch'],
                   'completed_epochs': values[4],
                   'planned_epochs': values[0]['epochs'],
                   'code_revision': values[0]['code_revision']}
            for name, values in loaded.items()
        },
        'configuration_check': {
            'equal_except': sorted(allowed_differences),
            'code_revision_note': ('Sequential launches may record different revisions after documentation '
                                   'or analysis changes; equal configs alone do not prove source equivalence'),
            'shared_train_targets': 37590,
            'shared_val_targets': 13912,
        },
        'comparisons': comparisons,
        'diagnostic_groups': diagnostic_metadata,
        'old_cross_reference': {
            'prediction_source': str(old_path.resolve()),
            'subset_targets': len(rows),
            'original_training_targets': 38854,
            'original_validation_targets': 14192,
            'cohort_note': ('identity-matched subset only; old cross used a different training and '
                            'validation cohort and is not a controlled five-frame arm'),
            'fixed_local': evaluate_blurball(
                rows, old_xy[old_ids], old_q[old_ids]),
            'diagnostic_group_metrics': {
                name: evaluate_blurball([row for row, keep in zip(rows, mask) if keep],
                                       old_xy[old_ids][mask], old_q[old_ids][mask], grouped=False)
                for name, mask in masks.items() if mask.any()},
        },
    }
    (output/'comparison.json').write_text(
        json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['training_status'], indent=2))


if __name__ == '__main__':
    main()
