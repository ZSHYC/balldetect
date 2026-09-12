"""配对同预算空间交互的保存预测，复用既有指标及真实历史分组。"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_blurball_temporal import (
    ROOT, _group_masks, _load_run, continuous_windows, evaluate_blurball,
    paired_decisions, paired_raw,
)


def compare_group(rows, same_xy, same_q, cross_xy, cross_q, mask):
    ids = np.flatnonzero(mask)
    selected = [rows[i] for i in ids]
    raw = paired_raw(rows, cross_xy, same_xy, mask)
    for counts in raw.values():
        counts['rescued'] = counts.pop('rescued_by_history')
        counts['broken'] = counts.pop('broken_by_history')
        counts['net_cross_pck_change'] = counts.pop('net_history_pck_change')
    decisions = paired_decisions(selected, cross_xy[ids], cross_q[ids], same_xy[ids], same_q[ids])
    decisions['matrix_axes'] = {'rows': 'same_address', 'columns': 'cross_address'}
    return {'n_targets': len(ids),
            'same_address': evaluate_blurball(selected, same_xy[ids], same_q[ids], grouped=False),
            'cross_address': evaluate_blurball(selected, cross_xy[ids], cross_q[ids], grouped=False),
            'paired_raw': raw, 'paired_decisions': decisions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same', type=Path, required=True)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'不覆盖已有比较结果: {args.output}')
    runs = {'same_address': args.same, 'cross_address': args.cross}
    configs = {name: json.loads((run / 'config.json').read_text()) for name, run in runs.items()}
    common = ('rgb_cache', 'continuity_boundaries', 'seed', 'epochs', 'batch_size', 'optimizer',
              'weights', 'head', 'input_slots', 'train_targets', 'val_targets', 'precision',
              'augmentation', 'selection', 'total_parameters', 'protocol')
    if any(configs['same_address'][key] != configs['cross_address'][key] for key in common):
        raise ValueError('两臂共同实验条件不同，不能作为锁定的同预算比较')
    for name, config in configs.items():
        if (config['interaction'] != name or config['temporal_input'] != 'history'
                or config['protocol'] != 'blurball-spatial-interaction-v1'):
            raise ValueError(f'{name}不是本协议的真实历史空间交互运行')
    config = configs['same_address']
    metadata = json.loads((ROOT / config['rgb_cache'] / 'metadata.json').read_text())
    frames = metadata['frames']
    windows, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = np.array([w for w in windows if frames[w[-1]]['split'] == 'val'])
    rows = [frames[w[-1]] for w in windows]
    if len(rows) != config['val_targets']:
        raise ValueError('验证目标数与训练配置不同')
    predictions = {name: _load_run(run, rows) for name, run in runs.items()}
    for name, values in predictions.items():
        if not np.array_equal(values['argmax'][1], values['local_readout'][1]):
            raise ValueError(f'{name}局部读出改变了原q')
    histories = {name: [json.loads(s) for s in (run / 'history.jsonl').read_text().splitlines()]
                 for name, run in runs.items()}
    for name, records in histories.items():
        if [r['epoch'] for r in records] != list(range(config['epochs'] + 1)):
            raise ValueError(f'{name}尚未具有锁定预算的完整逐轮记录')
    masks = _group_masks(rows, frames, windows)
    games = np.array([r['game'] for r in rows])
    for game in sorted(set(games)):
        masks[f'd1_ge16_match/{game}'] = masks['displacement/d1_ge16'] & (games == game)
    result = {'protocol': config['protocol'], 'target_count': len(rows),
              'paired_direction': 'same_address -> cross_address',
              'runs': {name: str(run.resolve()) for name, run in runs.items()},
              'training': {name: {'epochs': records[-1]['epoch'],
                                 'best_epoch': json.loads((runs[name] / 'results.json').read_text())['best_epoch'],
                                 'fixed_final_argmax': records[-1]['val']}
                           for name, records in histories.items()}, 'decoders': {}}
    for decoder in ('argmax', 'local_readout'):
        same_xy, same_q = predictions['same_address'][decoder]
        cross_xy, cross_q = predictions['cross_address'][decoder]
        result['decoders'][decoder] = {
            name: compare_group(rows, same_xy, same_q, cross_xy, cross_q, mask)
            for name, mask in masks.items() if mask.any()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: groups['all']['paired_raw']
                      for name, groups in result['decoders'].items()}, indent=2))


if __name__ == '__main__':
    main()
