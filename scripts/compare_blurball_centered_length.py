"""相同共同目标上比较center5→center3，不重跑模型或选择阈值。"""
import json

import numpy as np

from compare_blurball_five_frame import (
    ROOT, _validate_saved_rows, comparison_groups, diagnostic_groups, load_arm,
    read_predictions,
)


def validate_configs(configs):
    """允许协议预先声明的帧数/头形状变化，拒绝额外训练条件差异。"""
    differences = {'protocol', 'window', 'input_slots', 'target_slot', 'head',
                   'total_parameters', 'output', 'code_revision'}
    before, after = configs['center5'], configs['center3']
    changed = [key for key in sorted((set(before) | set(after)) - differences)
               if before.get(key) != after.get(key)]
    if changed:
        raise ValueError(f'中心长度控制出现额外配置差异: {changed}')
    for name, frames, slot, protocol, parameters in (
            ('center5', 5, 2, 'blurball-five-frame-context-v1', 1279169),
            ('center3', 3, 1, 'blurball-centered-length-v1', 1266497)):
        expected = {
            'protocol': protocol, 'window': name, 'target_slot': slot,
            'input_slots': [f't{i:+d}' if i else 't' for i in range(-slot, slot+1)],
            'head': {'input_channels': 192*frames, 'appearance_channels': 192*frames,
                     'hidden_channels': 32, 'upscale': 8, 'num_frames': frames},
            'total_parameters': parameters, 'epochs': 12, 'batch_size': 4, 'seed': 0,
            'train_targets': 37590, 'val_targets': 13912,
            'interaction': 'cross_address', 'temporal_input': 'history',
        }
        mismatched = [key for key, value in expected.items() if configs[name].get(key) != value]
        if mismatched:
            raise ValueError(f'{name}不符合长度协议: {mismatched}')


def main():
    output = ROOT/'outputs/blurball/centered_length'
    paths = {'center5': ROOT/'outputs/blurball/five_frame_context/center5_seed0',
             'center3': output/'center3_seed0'}
    configs = {name: json.loads((path/'config.json').read_text()) for name, path in paths.items()}
    validate_configs(configs)
    metadata = json.loads((ROOT/configs['center5']['rgb_cache']/'metadata.json').read_text())
    loaded = {name: load_arm(path, metadata) for name, path in paths.items()}
    rows = loaded['center5'][2]
    _validate_saved_rows(rows, loaded['center3'][2], paths['center3'])

    old_path = ROOT/'outputs/blurball/dino_cross_address_seed0/local_readout/val_predictions.csv'
    old_rows, old_xy, _ = read_predictions(old_path)
    same_path = ROOT/'outputs/blurball/dino_same_address_seed0/local_readout/val_predictions.csv'
    same_rows, same_xy, _ = read_predictions(same_path)
    _validate_saved_rows(old_rows, same_rows, same_path)
    candidate_path = ROOT/'outputs/blurball/spatial_interaction/candidate_coverage/cross_address.npz'
    with np.load(candidate_path) as candidates:
        source_rows = [metadata['frames'][i] for i in candidates['current_frame_ids']]
        _validate_saved_rows(old_rows, source_rows, candidate_path)
        candidate_xy = candidates['local_xy']
    if not np.array_equal(candidate_xy[:, 0], old_xy):
        raise ValueError('旧困难群体的候选0与保存预测不一致')
    masks, group_info = diagnostic_groups(rows, metadata['frames'], old_rows, old_xy, same_xy, candidate_xy)
    comparisons = {
        decoder: comparison_groups(rows, *loaded['center5'][3][decoder],
                                   *loaded['center3'][3][decoder], masks,
                                   before_name='center5', after_name='center3')
        for decoder in ('argmax', 'fixed_local')
    }
    result = {
        'protocol': 'blurball-centered-length-v1',
        'scope': 'common validation targets; single seed; no final test',
        'direction': 'center5 -> center3', 'targets': len(rows),
        'training_status': {
            name: {'best_epoch': value[1]['best_epoch'], 'completed_epochs': value[4],
                   'code_revision': value[0]['code_revision'],
                   'parameters': value[0]['total_parameters'], 'input_slots': value[0]['input_slots']}
            for name, value in loaded.items()},
        'attribution': 'Input length, head shape and parameter count differ; not a parameter-matched motion ablation',
        'diagnostic_groups': group_info, 'comparisons': comparisons,
    }
    (output/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['training_status'], indent=2))


if __name__ == '__main__':
    main()
