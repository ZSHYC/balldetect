"""相同中心窗口上比较无增强与同步翻转，只读取保存预测。"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_blurball_five_frame import ROOT, _validate_saved_rows, comparison_groups, load_arm
from compare_blurball_temporal import _identity


def validate_configs(baseline, augmented):
    allowed = {'protocol', 'augmentation', 'augmentation_probability', 'output', 'code_revision'}
    changed = [key for key in sorted((set(baseline) | set(augmented)) - allowed)
               if baseline.get(key) != augmented.get(key)]
    if changed:
        raise ValueError(f'同步翻转控制出现额外配置差异: {changed}')
    protocol = {'center3': 'blurball-centered-length-v1',
                'center5': 'blurball-five-frame-context-v1'}.get(baseline.get('window'))
    if protocol is None or baseline.get('protocol') != protocol:
        raise ValueError('无增强参考必须是原center3或center5共同目标协议')
    if baseline.get('augmentation') is not None:
        raise ValueError('参考模型必须无增强')
    if (augmented.get('protocol') != 'blurball-centered-hflip-v1'
            or augmented.get('augmentation') != 'hflip'
            or augmented.get('augmentation_probability') != .5):
        raise ValueError('增强模型必须只启用0.5概率同步水平翻转')
    expected = {'epochs': 12, 'batch_size': 4, 'seed': 0,
                'train_targets': 37590, 'val_targets': 13912,
                'interaction': 'cross_address', 'temporal_input': 'history'}
    mismatched = [key for key, value in expected.items() if baseline.get(key) != value]
    if mismatched:
        raise ValueError(f'训练条件不符合增强协议: {mismatched}')


def displacement_groups(rows, frames):
    lookup = {_identity(row): row for row in frames}
    previous = [lookup[(r['game'], r['clip'], int(r['original_frame_id']) - 1)] for r in rows]
    valid = np.array([r['visibility_raw'] == p['visibility_raw'] == 1
                      for r, p in zip(rows, previous)])
    xy = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    previous_xy = np.array([[r['x_raw'], r['y_raw']] for r in previous])
    distance = np.linalg.norm(xy - previous_xy, axis=1)
    return {'displacement/d1_lt4': valid & (distance < 4),
            'displacement/d1_4_to16': valid & (distance >= 4) & (distance < 16),
            'displacement/d1_ge16': valid & (distance >= 16)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--augmented', type=Path, required=True)
    args = parser.parse_args()
    paths = {'noaug': args.baseline, 'hflip': args.augmented}
    configs = {name: json.loads((path / 'config.json').read_text()) for name, path in paths.items()}
    validate_configs(configs['noaug'], configs['hflip'])
    metadata = json.loads((ROOT / configs['noaug']['rgb_cache'] / 'metadata.json').read_text())
    loaded = {name: load_arm(path, metadata) for name, path in paths.items()}
    rows = loaded['noaug'][2]
    _validate_saved_rows(rows, loaded['hflip'][2], paths['hflip'])
    # 同一套固定读出；错误半径、温度或best来源时，先重新生成对应预测再比较。
    for name, path in paths.items():
        local = json.loads((path / 'local_readout/config.json').read_text())
        if (local['radius_cells'] != 7 or local['temperature'] != 1
                or local['checkpoint_epoch'] != loaded[name][1]['best_epoch']):
            raise ValueError(f'{path} 局部读出未使用固定规则或对应best')
    masks = displacement_groups(rows, metadata['frames'])
    comparisons = {
        decoder: comparison_groups(rows, *loaded['noaug'][3][decoder],
                                   *loaded['hflip'][3][decoder], masks,
                                   before_name='noaug', after_name='hflip')
        for decoder in ('argmax', 'fixed_local')}
    result = {
        'protocol': 'blurball-centered-hflip-v1',
        'scope': 'same-window common validation targets; single seed; no final test',
        'window': configs['noaug']['window'], 'direction': 'noaug -> hflip',
        'targets': len(rows), 'threshold': .5,
        'training_status': {
            name: {'path': str(paths[name].resolve()), 'best_epoch': value[1]['best_epoch'],
                   'completed_epochs': value[4], 'code_revision': value[0]['code_revision']}
            for name, value in loaded.items()},
        'attribution': 'Training augmentation only; different code revisions require source review',
        'comparisons': comparisons,
    }
    destination = args.augmented / 'comparison.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(destination), 'training_status': result['training_status']}, indent=2))


if __name__ == '__main__':
    main()
