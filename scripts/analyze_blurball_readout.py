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
from ballmotion.blurball import continuous_windows, evaluate_blurball, source_coordinates
from ballmotion.tennis import grid_to_original
from train_tennis_heatmap import build_dino_model, write_predictions


def local_patches(spatial):
    """空间argmax附近固定半径7；图像外不贡献质量。"""
    batch, height, width = spatial.shape
    peak = spatial.flatten(1).argmax(1)
    center = torch.stack((peak % width, peak // width), dim=1)
    offset = torch.arange(-7, 8, device=spatial.device)
    y = center[:, 1, None, None] + offset[None, :, None]
    x = center[:, 0, None, None] + offset[None, None, :]
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    patch = spatial[torch.arange(batch, device=spatial.device)[:, None, None],
                    y.clamp(0, height - 1), x.clamp(0, width - 1)]
    return center, patch.masked_fill(~valid, -torch.inf)


def barycenters(center, patches):
    values = patches.astype(np.float64)
    mass = np.exp(values - values.max(axis=(1, 2), keepdims=True))
    mass /= mass.sum(axis=(1, 2), keepdims=True)
    offset = np.arange(-7, 8)
    return center + np.column_stack(((mass * offset[None, None, :]).sum(axis=(1, 2)),
                                     (mass * offset[None, :, None]).sum(axis=(1, 2))))


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
    metadata = json.loads((Path(config['rgb_cache']) / 'metadata.json').read_text())
    frames = metadata['frames']
    windows, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    # 较早history运行的配置没有temporal_input；它们均使用真实三帧。
    temporal_input = config.get('temporal_input', 'history')
    if temporal_input == 'repeat_current':
        windows = np.repeat(windows[:, -1:], 3, axis=1)
    indices = np.array([i for i, window in enumerate(windows) if frames[window[-1]]['split'] == 'val'])
    rows = [frames[windows[i, -1]] for i in indices]
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
        model = build_dino_model(config['weights'], upscale=8).to(device)
        checkpoint = torch.load(run / 'best.pt', map_location=device, weights_only=True)
        selected_epoch = json.loads((run / 'results.json').read_text())['best_epoch']
        assert checkpoint['epoch'] == selected_epoch, 'Checkpoint must match saved argmax selection'
        model.load_state_dict(checkpoint['model'])
        model.eval()
        rgb = np.load(Path(config['rgb_cache']) / 'rgb.npy', mmap_mode='r')
        centers, patches, q = [], [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(indices), 8):
                batch_windows = windows[indices[start:start+8]]
                unique, inverse = np.unique(batch_windows, return_inverse=True)
                features = model.encode(torch.from_numpy(rgb[unique]).to(device))
                features = features[torch.from_numpy(inverse).to(device)].reshape(
                    len(batch_windows), 3 * features.shape[1], *features.shape[-2:])
                logits = model.head(features)
                center, patch = local_patches(logits[:, :-1].reshape(-1, 288, 512))
                centers.append(center.cpu().numpy())
                patches.append(patch.cpu().numpy())
                q.append((1 - logits.softmax(1)[:, -1]).cpu().numpy())
                if start % 800 == 0:
                    print(f'validation {start+len(batch_windows)}/{len(indices)}', flush=True)
        centers, patches, q = np.concatenate(centers), np.concatenate(patches), np.concatenate(q)
        xy = source_coordinates(grid_to_original(centers[:, 1] * 512 + centers[:, 0], (288, 512)), rows)
        assert np.array_equal(xy, original_xy), 'Original argmax was not reproduced'
        difference = float(np.max(np.abs(q-original_q)))
        assert difference <= 1e-6 and np.array_equal(q >= .5, original_q >= .5), 'Original q/output decisions differ'
        np.savez(cache_path, window_ids=indices, centers=centers, patches=patches, q=q)
        info = {'protocol': ('blurball-full-temporal-control-v1' if temporal_input == 'repeat_current'
                             else 'blurball-local-readout-v1'), 'source_run': str(run),
                'temporal_input': temporal_input,
                'checkpoint_epoch': checkpoint['epoch'], 'radius_cells': 7, 'temperature': 1,
                'batch_size': 8, 'precision': 'float32 model/logits; float64 CPU expectation',
                'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'argmax_exact': True, 'max_q_difference': difference, 'output_decisions_equal': True,
                'forward_elapsed_seconds': time.perf_counter()-started,
                'timing_scope': 'metadata/model load + RGB/H2D/forward + local extraction/verification/cache write',
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
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
