"""固定已选best，以一次验证forward比较argmax和15×15局部重心。"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ballmotion.blurball import evaluate_blurball, source_coordinates
from ballmotion.tennis import grid_to_original
from train_blurball_midpoint import WINDOW_OFFSETS, prepare_training_windows
from train_tennis_heatmap import build_dino_model, write_predictions


def local_patches(spatial):
    """空间argmax附近固定半径7；图像外不贡献质量。"""
    width = spatial.shape[-1]
    peak = spatial.flatten(1).argmax(1)
    center = torch.stack((peak % width, peak // width), dim=1)
    return center, patches_at_centers(spatial, center[:, None])[:, 0]


def patches_at_centers(spatial, centers):
    """从原logit图提取B×K个15×15窗口，图像外不参与重心。"""
    batch, height, width = spatial.shape
    offset = torch.arange(-7, 8, device=spatial.device)
    y = centers[:, :, 1, None, None] + offset[None, None, :, None]
    x = centers[:, :, 0, None, None] + offset[None, None, None, :]
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    patch = spatial[torch.arange(batch, device=spatial.device)[:, None, None, None],
                    y.clamp(0, height - 1), x.clamp(0, width - 1)]
    return patch.masked_fill(~valid, -torch.inf)


def barycenters(center, patches):
    values = patches.astype(np.float64)
    mass = np.exp(values - values.max(axis=(1, 2), keepdims=True))
    mass /= mass.sum(axis=(1, 2), keepdims=True)
    offset = np.arange(-7, 8)
    return center + np.column_stack(((mass * offset[None, None, :]).sum(axis=(1, 2)),
                                     (mass * offset[None, :, None]).sum(axis=(1, 2))))


def batch_logits(model, rgb, batch_windows, device):
    """复用批内源帧编码，按原时间槽顺序返回完整logits。"""
    unique, inverse = np.unique(batch_windows, return_inverse=True)
    features = model.encode(torch.from_numpy(rgb[unique]).to(device))
    num_frames = batch_windows.shape[1]
    features = features[torch.from_numpy(inverse).to(device)].reshape(
        len(batch_windows), num_frames * features.shape[1], *features.shape[-2:])
    return model.head(features)


def batch_readout(model, rgb, batch_windows, device):
    """返回argmax邻域和原q，供固定读出与输入干预共用。"""
    logits = batch_logits(model, rgb, batch_windows, device)
    center, patch = local_patches(logits[:, :-1].reshape(-1, 288, 512))
    return center, patch, 1 - logits.softmax(1)[:, -1]


def paired_groups(rows, old, new):
    target = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    old_error, new_error = np.linalg.norm(old-target, axis=1), np.linalg.norm(new-target, axis=1)
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    lengths = np.array([r['l_raw'] for r in rows])
    games = np.array([r['game'] for r in rows])
    bins = {'all': np.ones(len(rows), dtype=bool), 'l0': lengths == 0,
            '0_2': (lengths > 0) & (lengths <= 2), '2_5': (lengths > 2) & (lengths <= 5),
            '5_10': (lengths > 5) & (lengths <= 10), 'gt10': lengths > 10}
    result = {}
    bands = {'': np.ones(len(rows), dtype=bool), '/old_lt4': old_error < 4,
             '/old_4_16': (old_error >= 4) & (old_error < 16), '/old_ge16': old_error >= 16}
    for game in ('all', *sorted(set(games))):
        for label, length_mask in bins.items():
            base = visible & length_mask & ((games == game) if game != 'all' else True)
            for band, band_mask in bands.items():
                mask = base & band_mask
                counts = {}
                for radius in (4, 8, 16):
                    a, b = old_error[mask] < radius, new_error[mask] < radius
                    counts[str(radius)] = {'n': int(mask.sum()),
                        'rescued': int(np.sum(~a & b)), 'broken': int(np.sum(a & ~b)),
                        'both_correct': int(np.sum(a & b)), 'both_wrong': int(np.sum(~a & ~b))}
                result[f'{game}/{label}{band}'] = counts
    return result


def main(run):
    started = time.perf_counter()
    config = json.loads((run / 'config.json').read_text())
    run_results = json.loads((run / 'results.json').read_text())
    metadata = json.loads((Path(config['rgb_cache']) / 'metadata.json').read_text())
    frames = metadata['frames']
    # 较早history运行的配置没有temporal_input；它们均使用真实三帧。
    temporal_input = config.get('temporal_input', 'history')
    window = config.get('window')
    windows, all_rows, target_slot, _ = prepare_training_windows(
        frames, metadata['windows'], config['continuity_boundaries'], window, temporal_input)
    if config.get('target_slot', 2) != target_slot:
        raise ValueError('训练配置的target_slot与重建窗口不一致')
    offsets = WINDOW_OFFSETS[window] if window is not None else (-2, -1, 0)
    input_slots = (["t"] * len(offsets) if temporal_input == 'repeat_current' else
                   [f't{offset:+d}' if offset else 't' for offset in offsets])
    if config.get('input_slots', input_slots) != input_slots:
        raise ValueError('训练配置的input_slots与重建窗口不一致')
    indices = np.flatnonzero([row['split'] == 'val' for row in all_rows])
    rows = [all_rows[i] for i in indices]
    with (run / 'val_predictions.csv').open(newline='') as handle:
        original = list(csv.DictReader(handle))
    key = lambda r: (r['game'], r['clip'], int(r['original_frame_id']))
    assert [key(r) for r in rows] == [key(r) for r in original], 'Forward target order differs'
    original_xy = np.array([[float(r['pred_x']), float(r['pred_y'])] for r in original])
    original_q = np.array([float(r['presence_probability']) for r in original])
    output = run / 'local_readout'
    output.mkdir(exist_ok=True)
    cache_path = output / 'local_logits.npz'
    if cache_path.exists():
        saved = np.load(cache_path)
        assert np.array_equal(saved['window_ids'], indices), 'Cached target order differs'
        centers, patches, q = saved['centers'], saved['patches'], saved['q']
    else:
        torch.set_num_threads(4)
        device = torch.device('cuda')
        num_frames = windows.shape[1]
        batch_size = config['batch_size'] if window is not None else 8
        model = build_dino_model(config['weights'], upscale=8,
                                 interaction=config.get('interaction', 'baseline'),
                                 num_frames=num_frames, target_slot=target_slot).to(device)
        checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
        selected_epoch = run_results['best_epoch']
        assert checkpoint['epoch'] == selected_epoch, 'Checkpoint must match saved argmax selection'
        model.load_state_dict(checkpoint['model'])
        model.eval()
        rgb = np.load(Path(config['rgb_cache']) / 'rgb.npy', mmap_mode='r')
        centers, patches, q = [], [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                batch_windows = windows[indices[start:start+batch_size]]
                center, patch, probability = batch_readout(model, rgb, batch_windows, device)
                centers.append(center.cpu().numpy())
                patches.append(patch.cpu().numpy())
                q.append(probability.cpu().numpy())
                if start % 800 == 0:
                    print(f'validation {start+len(batch_windows)}/{len(indices)}', flush=True)
        centers, patches, q = np.concatenate(centers), np.concatenate(patches), np.concatenate(q)
        xy = source_coordinates(grid_to_original(centers[:, 1] * 512 + centers[:, 0], (288, 512)), rows)
        assert np.array_equal(xy, original_xy), 'Original argmax was not reproduced'
        difference = float(np.max(np.abs(q-original_q)))
        assert difference <= 1e-6 and np.array_equal(q >= .5, original_q >= .5), 'Original q/output decisions differ'
        np.savez(cache_path, window_ids=indices, centers=centers, patches=patches, q=q)
        info = {'protocol': (config['protocol'] if config.get('interaction', 'baseline') != 'baseline' else
                             'blurball-full-temporal-control-v1' if temporal_input == 'repeat_current'
                             else 'blurball-local-readout-v1'), 'source_run': str(run),
                'training_complete': run_results.get('training_complete', True),
                'temporal_input': temporal_input,
                'interaction': config.get('interaction', 'baseline'),
                'checkpoint_epoch': checkpoint['epoch'], 'radius_cells': 7, 'temperature': 1,
                'batch_size': batch_size, 'precision': 'float32 model/logits; float64 CPU expectation',
                'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'argmax_exact': True, 'max_q_difference': difference, 'output_decisions_equal': True,
                'forward_elapsed_seconds': time.perf_counter()-started,
                'timing_scope': 'metadata/model load + RGB/H2D/forward + local extraction/verification/cache write',
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
        if window is not None:
            info.update(window=window, input_slots=input_slots, target_slot=target_slot)
        if not info['training_complete']:
            info['reference_protocol'] = info['protocol']
            info['protocol'] = ('blurball-spatial-interaction-development'
                                if config.get('interaction', 'baseline') != 'baseline'
                                else 'exploratory-interrupted-temporal-control')
            info['completed_epochs'] = run_results['completed_epochs']
            if 'stop_reason' in run_results:
                info['stop_reason'] = run_results['stop_reason']
        (output / 'config.json').write_text(json.dumps(info, indent=2) + '\n')
    grid_xy = barycenters(centers, patches)
    dimensions = np.array([[r['width'], r['height']] for r in rows])
    new_xy = (grid_xy + .5) * dimensions / [512, 288] - .5
    # 沿用保存的原q，确保本次只替换位置。
    write_predictions(output / 'val_predictions.csv', rows, new_xy, original_q)
    result = {'baseline': evaluate_blurball(rows, original_xy, original_q),
              'local_barycenter': evaluate_blurball(rows, new_xy, original_q),
              'paired': paired_groups(rows, original_xy, new_xy)}
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {k: result[name][k] for k in ('location', 'detection4', 'detection16')}
                      for name in ('baseline', 'local_barycenter')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    main(parser.parse_args().run)
