"""比较固定五帧增强基线与目标/支持先激活控制，只读取保存预测和候选。"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_blurball_five_frame import ROOT, _validate_saved_rows, comparison_groups, load_arm
from compare_blurball_hflip import displacement_groups


BASELINE = ROOT / 'outputs/blurball/centered_hflip/center5_seed0'
EXPECTED_SLOTS = ['t-2', 't-1', 't', 't+1', 't+2']


def validate_configs(baseline, challenger):
    allowed = {'interaction', 'protocol', 'output', 'code_revision'}
    changed = [key for key in sorted((set(baseline) | set(challenger)) - allowed)
               if baseline.get(key) != challenger.get(key)]
    if changed:
        raise ValueError(f'激活顺序控制出现额外配置差异: {changed}')
    if (baseline.get('interaction') != 'cross_address'
            or baseline.get('protocol') != 'blurball-centered-hflip-v1'):
        raise ValueError('参考模型必须是center5同步翻转cross_address基线')
    if (challenger.get('interaction') != 'target_activation'
            or challenger.get('protocol') != 'blurball-temporal-activation-v1'):
        raise ValueError('挑战模型必须只改变为target_activation协议')
    expected = {
        'window': 'center5', 'input_slots': EXPECTED_SLOTS, 'target_slot': 2,
        'temporal_input': 'history', 'augmentation': 'hflip',
        'augmentation_probability': .5, 'seed': 0, 'batch_size': 4, 'epochs': 12,
        'train_targets': 37590, 'val_targets': 13912, 'total_parameters': 1279169,
        'head': {'input_channels': 960, 'hidden_channels': 32, 'upscale': 8,
                 'num_frames': 5, 'appearance_channels': 960},
    }
    mismatched = [key for key, value in expected.items() if baseline.get(key) != value]
    if mismatched:
        raise ValueError(f'训练条件不符合激活顺序协议: {mismatched}')


def validate_candidate_arrays(rows, frames, arrays, fixed_xy, fixed_q, path):
    required = {'local_xy', 'q', 'current_frame_ids'}
    if not required <= set(arrays):
        raise ValueError(f'{path} 缺少候选字段: {sorted(required-set(arrays))}')
    frame_ids = np.asarray(arrays['current_frame_ids'])
    if frame_ids.shape != (len(rows),) or np.any(frame_ids < 0) or np.any(frame_ids >= len(frames)):
        raise ValueError(f'{path} 候选帧索引无效')
    source_rows = [frames[int(index)] for index in frame_ids]
    _validate_saved_rows(rows, source_rows, path)
    xy, q = np.asarray(arrays['local_xy']), np.asarray(arrays['q'])
    if xy.shape != (len(rows), 16, 2) or q.shape != (len(rows),):
        raise ValueError(f'{path} 不是逐目标K16候选')
    if not np.isfinite(xy).all() or not np.isfinite(q).all():
        raise ValueError(f'{path} 候选含非有限值')
    np.testing.assert_allclose(xy[:, 0], fixed_xy, rtol=0, atol=1e-10,
                               err_msg=f'{path} 候选0未复现fixed_local')
    if float(np.max(np.abs(q-fixed_q), initial=0.)) > 1e-6:
        raise ValueError(f'{path} 候选q未复现保存预测')
    if not np.array_equal(q >= .5, fixed_q >= .5):
        raise ValueError(f'{path} 候选q改变了固定阈值决策')


def load_candidates(run, rows, frames, fixed_xy, fixed_q, best_epoch, interaction):
    directory = run / 'candidate_coverage'
    manifest = json.loads((directory / 'manifest.json').read_text())
    expected = {'candidate_count': 16, 'suppression_radius_cells': 7,
                'target_count': len(rows), 'target_slot': 2,
                'input_frame_offsets': [-2, -1, 0, 1, 2],
                'checkpoint_epoch': best_epoch, 'target_split': 'val',
                'interaction': interaction, 'source_run': str(run.resolve())}
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    if mismatched:
        raise ValueError(f'{directory} 候选导出不符合固定协议: {mismatched}')
    with np.load(directory / 'candidates.npz') as saved:
        arrays = {key: saved[key] for key in saved.files}
    validate_candidate_arrays(rows, frames, arrays, fixed_xy, fixed_q,
                              directory / 'candidates.npz')
    return arrays, manifest


def baseline_error_masks(rows, baseline_xy):
    target = np.array([[row['x_raw'], row['y_raw']] for row in rows])
    visible = np.array([row['visibility_raw'] == 1 for row in rows])
    error = np.linalg.norm(baseline_xy-target[:, None], axis=2)
    best = error.min(axis=1)
    return {
        'baseline/top1_good4': visible & (error[:, 0] < 4),
        'baseline_error/top1_wrong4_K16_good4': visible & (error[:, 0] >= 4) & (best < 4),
        'baseline_error/K16_best_4_to16': visible & (best >= 4) & (best < 16),
        'baseline_error/K16_best_ge16': visible & (best >= 16),
    }


def candidate_groups(rows, baseline_xy, challenger_xy, extra_masks):
    target = np.array([[row['x_raw'], row['y_raw']] for row in rows])
    visible = np.array([row['visibility_raw'] == 1 for row in rows])
    games = np.array([row['game'] for row in rows])
    masks = {'all': np.ones(len(rows), dtype=bool)}
    masks.update({f'match/{game}': games == game for game in sorted(set(games))})
    masks.update(extra_masks)
    before = np.linalg.norm(baseline_xy-target[:, None], axis=2)
    after = np.linalg.norm(challenger_xy-target[:, None], axis=2)
    result = {}
    for name, mask in masks.items():
        selected = mask & visible
        n = int(selected.sum())
        if not n:
            continue
        group = {'n_visible': n, 'coverage': {}, 'paired_K16': {}}
        for model, error in (('baseline', before), ('challenger', after)):
            group['coverage'][model] = {
                f'K{k}@{radius}': {
                    'covered': int(np.sum(error[selected, :k].min(axis=1) < radius)),
                    'coverage': float(np.mean(error[selected, :k].min(axis=1) < radius)),
                }
                for k in (1, 16) for radius in (4, 16)
            }
        for radius in (4, 16):
            before_ok = before[selected].min(axis=1) < radius
            after_ok = after[selected].min(axis=1) < radius
            group['paired_K16'][str(radius)] = {
                'rescued': int(np.sum(~before_ok & after_ok)),
                'broken': int(np.sum(before_ok & ~after_ok)),
                'net_covered': int(np.sum(after_ok)-np.sum(before_ok)),
                'both_covered': int(np.sum(before_ok & after_ok)),
                'both_uncovered': int(np.sum(~before_ok & ~after_ok)),
            }
        result[name] = group
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--challenger', type=Path, required=True)
    args = parser.parse_args()
    paths = {'baseline': BASELINE, 'challenger': args.challenger}
    configs = {name: json.loads((path / 'config.json').read_text())
               for name, path in paths.items()}
    validate_configs(configs['baseline'], configs['challenger'])
    metadata = json.loads((ROOT / configs['baseline']['rgb_cache'] / 'metadata.json').read_text())
    loaded = {name: load_arm(path, metadata) for name, path in paths.items()}
    rows = loaded['baseline'][2]
    _validate_saved_rows(rows, loaded['challenger'][2], paths['challenger'])
    for name, path in paths.items():
        local = json.loads((path / 'local_readout/config.json').read_text())
        if (local.get('radius_cells') != 7 or local.get('temperature') != 1
                or local.get('checkpoint_epoch') != loaded[name][1]['best_epoch']):
            raise ValueError(f'{path} 局部读出未使用radius7/T1或对应best')

    candidates = {}
    manifests = {}
    for name, path in paths.items():
        fixed_xy, fixed_q = loaded[name][3]['fixed_local']
        candidates[name], manifests[name] = load_candidates(
            path, rows, metadata['frames'], fixed_xy, fixed_q,
            loaded[name][1]['best_epoch'], configs[name]['interaction'])
    if not np.array_equal(candidates['baseline']['current_frame_ids'],
                          candidates['challenger']['current_frame_ids']):
        raise ValueError('两臂候选目标身份或顺序不同')

    error_masks = baseline_error_masks(rows, candidates['baseline']['local_xy'])
    actual_masks = displacement_groups(rows, metadata['frames'])
    actual_masks.update(error_masks)
    comparisons = {
        decoder: comparison_groups(
            rows, *loaded['baseline'][3][decoder], *loaded['challenger'][3][decoder],
            actual_masks, before_name='baseline', after_name='challenger')
        for decoder in ('argmax', 'fixed_local')
    }
    candidate_comparison = candidate_groups(
        rows, candidates['baseline']['local_xy'], candidates['challenger']['local_xy'],
        error_masks)
    all_candidates = candidate_comparison['all']
    net_k16_4 = all_candidates['paired_K16']['4']['net_covered']
    baseline_metrics = comparisons['fixed_local']['all']['baseline']
    challenger_metrics = comparisons['fixed_local']['all']['challenger']
    gates = {
        'K16_at_4_net_covered_at_least_123': net_k16_4 >= 123,
        'K1_PCK4_not_lower': (challenger_metrics['location']['pck4']
                              >= baseline_metrics['location']['pck4']),
        'actual_F1_at_4_not_lower': (challenger_metrics['detection4']['f1']
                                     >= baseline_metrics['detection4']['f1']),
    }
    result = {
        'protocol': 'blurball-temporal-activation-v1',
        'scope': 'same center5 validation targets; single seed development selection; no final test',
        'direction': 'cross_address baseline -> target_activation challenger',
        'targets': len(rows), 'threshold': .5,
        'training_status': {
            name: {'path': str(paths[name].resolve()), 'best_epoch': value[1]['best_epoch'],
                   'completed_epochs': value[4], 'code_revision': value[0]['code_revision']}
            for name, value in loaded.items()},
        'candidate_sources': {
            name: {'path': str((paths[name] / 'candidate_coverage/candidates.npz').resolve()),
                   'checkpoint_epoch': manifest['checkpoint_epoch']}
            for name, manifest in manifests.items()},
        'attribution': ('Only the target/support activation order changes; actual prediction metrics '
                        'are separate from GT-evaluated candidate oracle coverage.'),
        'comparisons': comparisons,
        'candidate_comparison': candidate_comparison,
        'development_gate': {
            'rule': ('recommend replacing the development baseline only if K16@4 gains at least '
                     '123 covered V1 targets while fixed-local K1 PCK@4 and actual F1@4 do not fall'),
            'origin': '123 is approximately 10% of the baseline 1228 K16@4-uncovered V1 targets',
            'single_seed_selection_not_significance': True,
            'observed_K16_at_4_net_covered': net_k16_4,
            'checks': gates,
            'recommend_replace_baseline': all(gates.values()),
        },
    }
    destination = args.challenger / 'comparison.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(destination),
                      'development_gate': result['development_gate']}, indent=2))


if __name__ == '__main__':
    main()
