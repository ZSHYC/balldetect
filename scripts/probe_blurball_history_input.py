"""固定same/cross最佳权重，把历史槽替换为当前帧；完整验证，不训练。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from analyze_blurball_readout import barycenters, batch_readout
from compare_blurball_temporal import ROOT, _group, _group_masks, _load_run, continuous_windows
from train_tennis_heatmap import build_dino_model, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same', type=Path, required=True)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有输入干预结果')
    runs = {'same_address': args.same, 'cross_address': args.cross}
    configs = {k: json.loads((p / 'config.json').read_text()) for k, p in runs.items()}
    config = configs['same_address']
    for key in ('rgb_cache', 'continuity_boundaries', 'batch_size', 'val_targets'):
        assert config[key] == configs['cross_address'][key], key
    metadata = json.loads((ROOT / config['rgb_cache'] / 'metadata.json').read_text())
    frames = metadata['frames']
    windows, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = np.array([w for w in windows if frames[w[-1]]['split'] == 'val'])
    rows = [frames[w[-1]] for w in windows]
    assert len(rows) == config['val_targets'] == 14192
    original = {name: _load_run(run, rows) for name, run in runs.items()}
    masks = _group_masks(rows, frames, windows)
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    match21 = np.array([r['game'] == 'match21' for r in rows])
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    errors = {k: np.linalg.norm(v['local_readout'][0] - gt, axis=1) for k, v in original.items()}
    # 群体只由已保存的原输入预测定义，不随干预输出改变。
    masks['fixed/match21_lost16'] = match21 & visible & (errors['same_address'] < 16) & (errors['cross_address'] >= 16)
    masks['fixed/match21_rescued16'] = match21 & visible & (errors['same_address'] >= 16) & (errors['cross_address'] < 16)
    assert masks['fixed/match21_lost16'].sum() == 202
    assert masks['fixed/match21_rescued16'].sum() == 74
    repeated = np.repeat(windows[:, -1:], 3, axis=1)
    dimensions = np.array([[r['width'], r['height']] for r in rows])
    rgb = np.load(ROOT / config['rgb_cache'] / 'rgb.npy', mmap_mode='r')
    args.output.mkdir(parents=True)
    manifest = {'protocol': 'blurball-fixed-history-input-diagnostic-v1',
        'reference_input': ['t-2', 't-1', 't'], 'intervention_input': ['t', 't', 't'],
        'target_count': len(rows), 'target_slot': 2, 'batch_size': config['batch_size'],
        'scope': 'fixed trained weights; input distribution shift; not a retrained single-frame baseline',
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['scripts/analyze_blurball_readout.py: extracted batch_readout',
                         'scripts/probe_blurball_history_input.py: new diagnostic'],
        'precision': 'float32 model; float64 barycenter', 'models': {}}
    torch.set_num_threads(4)
    device = torch.device('cuda')
    for name, run in runs.items():
        started = time.perf_counter()
        c = configs[name]
        assert c['interaction'] == name and c['temporal_input'] == 'history'
        checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
        assert checkpoint['epoch'] == json.loads((run / 'results.json').read_text())['best_epoch']
        model = build_dino_model(c['weights'], upscale=8, interaction=name).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        centers, patches, probabilities = [], [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            # 原路径真实batch确认读取函数抽取未改变坐标/q，然后才运行干预。
            center, _, q = batch_readout(model, rgb, windows[:8], device)
            xy = (center.cpu().numpy()+.5) * dimensions[:8] / [512, 288] - .5
            assert np.array_equal(xy, original[name]['argmax'][0][:8])
            q_difference = float(np.abs(q.cpu().numpy()-original[name]['argmax'][1][:8]).max())
            assert q_difference <= 1e-6
            for start in range(0, len(rows), c['batch_size']):
                center, patch, q = batch_readout(model, rgb, repeated[start:start+c['batch_size']], device)
                centers.append(center.cpu().numpy())
                patches.append(patch.cpu().numpy())
                probabilities.append(q.cpu().numpy())
                if start % 1600 == 0:
                    print(f'{name}: {start+len(center)}/{len(rows)}', flush=True)
        centers, patches, q = np.concatenate(centers), np.concatenate(patches), np.concatenate(probabilities)
        forward_seconds = time.perf_counter()-started
        destination = args.output / name
        destination.mkdir()
        np.savez(destination / 'local_logits.npz', centers=centers, patches=patches, q=q,
                 current_frame_ids=windows[:, -1])
        info = {'source_run': str(run), 'checkpoint_epoch': checkpoint['epoch'],
                'original_batch_q_max_difference': q_difference, 'forward_elapsed_seconds': forward_seconds,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'decoders': {}}
        for decoder, grid_xy in (('argmax', centers), ('local_readout', barycenters(centers, patches))):
            xy = (grid_xy+.5) * dimensions / [512, 288] - .5
            write_predictions(destination / f'{decoder}_predictions.csv', rows, xy, q)
            old_xy, old_q = original[name][decoder]
            groups = {}
            for group_name, mask in masks.items():
                if not mask.any():
                    continue
                group = _group(rows, xy, q, old_xy, old_q, mask)
                group['original_history'] = group.pop('repeat')
                group['repeated_current'] = group.pop('history')
                group['paired_decisions']['matrix_axes'] = {'rows': 'original_history', 'columns': 'repeated_current'}
                for v in group['paired_raw'].values():
                    v['rescued_by_intervention'] = v.pop('rescued_by_history')
                    v['broken_by_intervention'] = v.pop('broken_by_history')
                    v['net_intervention_pck_change'] = v.pop('net_history_pck_change')
                groups[group_name] = group
            info['decoders'][decoder] = groups
        manifest['models'][name] = info
        (args.output / 'results.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
        print(name, 'complete', flush=True)
        del model, checkpoint


if __name__ == '__main__':
    main()
