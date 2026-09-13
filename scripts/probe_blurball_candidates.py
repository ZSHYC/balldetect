"""真实三帧下固定贪心空间候选的oracle覆盖，不训练或用GT选候选。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_blurball_readout import barycenters, batch_logits, patches_at_centers
from compare_blurball_temporal import _group_masks, _load_run, continuous_windows
from train_tennis_heatmap import build_dino_model


def greedy_candidates(spatial, count=16):
    """抑制半径7格方窗；分数/重心仍取原图。并列沿用argmax首索引。"""
    work = spatial.clone()
    height, width = spatial.shape[-2:]
    yy = torch.arange(height, device=spatial.device)[None, :, None]
    xx = torch.arange(width, device=spatial.device)[None, None, :]
    centers, scores = [], []
    for _ in range(count):
        peak = work.flatten(1).argmax(1)
        x, y = peak % width, peak // width
        centers.append(torch.stack((x, y), dim=1))
        scores.append(spatial.flatten(1).gather(1, peak[:, None])[:, 0])
        suppressed = (abs(xx-x[:, None, None]) <= 7) & (abs(yy-y[:, None, None]) <= 7)
        work.masked_fill_(suppressed, -torch.inf)
    centers = torch.stack(centers, dim=1)
    return centers, torch.stack(scores, dim=1), patches_at_centers(spatial, centers)


def coverage(rows, xy, masks, budgets=(1, 2, 4, 8, 16)):
    """GT仅评价已有候选，记录oracle覆盖，不选择实际输出。"""
    target = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    error = np.linalg.norm(xy-target[:, None], axis=2)
    best_error = np.minimum.accumulate(error, axis=1)
    result = {}
    for name, mask in masks.items():
        selected = mask & visible
        n = int(selected.sum())
        if not n:
            continue
        result[name] = {'n': n, 'budgets': {str(k): {str(radius): {
            'covered': int(np.sum(best_error[selected, k-1] < radius)),
            'coverage': float(np.mean(best_error[selected, k-1] < radius))}
            for radius in (4, 8, 16)} for k in budgets}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--same', type=Path, required=True)
    parser.add_argument('--cross', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有候选诊断')
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
    errors = {k: np.linalg.norm(v['local_readout'][0]-gt, axis=1) for k, v in original.items()}
    masks['fixed/match21_lost16'] = match21 & visible & (errors['same_address'] < 16) & (errors['cross_address'] >= 16)
    masks['fixed/match21_rescued16'] = match21 & visible & (errors['same_address'] >= 16) & (errors['cross_address'] < 16)
    assert masks['fixed/match21_lost16'].sum() == 202 and masks['fixed/match21_rescued16'].sum() == 74
    dimensions = np.array([[r['width'], r['height']] for r in rows])
    rgb = np.load(ROOT / config['rgb_cache'] / 'rgb.npy', mmap_mode='r')
    args.output.mkdir(parents=True)
    result = {'protocol': 'blurball-candidate-coverage-v1', 'target_count': len(rows),
        'budgets': [1, 2, 4, 8, 16], 'suppression_radius_cells': 7,
        'scope': 'GT-free proposals; oracle coverage only, not an automatic localization metric',
        'precision': 'float32 logits; float64 barycenter', 'input': ['t-2', 't-1', 't'],
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['scripts/analyze_blurball_readout.py: shared full logits and patch extraction',
                         'scripts/probe_blurball_candidates.py: candidate diagnostic'], 'models': {}}
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
        centers, local, scores, probabilities = [], [], [], []
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            for start in range(0, len(rows), c['batch_size']):
                batch_windows = windows[start:start+c['batch_size']]
                logits = batch_logits(model, rgb, batch_windows, device)
                center, score, patch = greedy_candidates(logits[:, :-1].reshape(-1, 288, 512))
                center = center.cpu().numpy()
                local.append(barycenters(center.reshape(-1, 2), patch.cpu().numpy().reshape(-1, 15, 15)).reshape(len(center), 16, 2))
                centers.append(center)
                scores.append(score.cpu().numpy())
                probabilities.append((1-logits.softmax(1)[:, -1]).cpu().numpy())
                if start % 1600 == 0:
                    print(f'{name}: {start+len(center)}/{len(rows)}', flush=True)
        centers, local, scores, q = map(np.concatenate, (centers, local, scores, probabilities))
        xy = (centers+.5)*dimensions[:, None]/[512, 288]-.5
        local_xy = (local+.5)*dimensions[:, None]/[512, 288]-.5
        assert np.array_equal(xy[:, 0], original[name]['argmax'][0]), 'first candidate argmax drift'
        np.testing.assert_allclose(local_xy[:, 0], original[name]['local_readout'][0], rtol=0, atol=1e-10)
        max_q_difference = float(np.abs(q-original[name]['argmax'][1]).max())
        assert max_q_difference <= 1e-6
        assert np.array_equal(q >= .5, original[name]['argmax'][1] >= .5)
        forward_seconds = time.perf_counter()-started
        np.savez(args.output / f'{name}.npz', grid_centers=centers, original_xy=xy,
                 local_xy=local_xy, peak_logits=scores, q=q, current_frame_ids=windows[:, -1])
        model_masks = dict(masks, original_emitted=q >= .5, original_rejected=q < .5)
        result['models'][name] = {'source_run': str(run), 'checkpoint_epoch': checkpoint['epoch'],
            'first_candidate_reproduced': True, 'max_q_difference': max_q_difference,
            'forward_and_extraction_seconds': forward_seconds,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'grid_centers': coverage(rows, xy, model_masks),
            'local_readout': coverage(rows, local_xy, model_masks)}
        (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print(name, 'complete', flush=True)
        del model, checkpoint


if __name__ == '__main__':
    main()
