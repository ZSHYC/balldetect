"""原三帧固定前缀的候选视觉对应：条件匹配与自动选择分开。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'src'))
from ballmotion.correspondence import candidate_costs
from ballmotion.tennis import grid_targets
from compare_blurball_temporal import _group_masks, continuous_windows
from rerank_blurball_candidates import groups
from train_tennis_heatmap import build_dino_model, write_predictions


def true_cell_ranks(costs, cells):
    """按分数降序、展平索引升序计算真实格排名，避免并列的乐观排名。"""
    targets = cells[:, :, None, None].expand(*costs.shape[:3], 1)
    truth = costs.gather(-1, targets)
    earlier = torch.arange(costs.shape[-1], device=costs.device) < targets
    return 1 + ((costs > truth) | ((costs == truth) & earlier)).sum(-1)


def summarize_matches(rows, frames, windows, xy, arrays, grid_hw, masks):
    height, width = grid_hw
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    error = np.linalg.norm(xy-gt[:, None], axis=-1)
    best_query = error.argmin(1)
    covered = visible & (error.min(1) < 4)
    result = {}
    for delta in (1, 2):
        slot = delta-1
        old = [frames[w[2-delta]] for w in windows]
        both = visible & np.array([r['visibility_raw'] == 1 for r in old])
        old_gt = np.array([[r['x_raw'], r['y_raw']] for r in old])
        dimensions = np.array([[r['width'], r['height']] for r in old])
        idx = np.arange(len(rows))
        cell = arrays['matched_cells'][idx, slot, best_query]
        point = (np.stack((cell % width, cell // width), axis=1)+.5)*dimensions/[width, height]-.5
        hit16 = np.linalg.norm(point-old_gt, axis=1) < 16
        ranks = arrays['true_ranks'][idx, slot, best_query]
        difference = arrays['max_cosine'][idx, slot, best_query]-arrays['max_cosine'][:, slot, 0]
        current_dimensions = np.array([[r['width'], r['height']] for r in rows])
        current_native = grid_targets(gt, visible, grid_hw,
                                      (current_dimensions[:, 1], current_dimensions[:, 0]))
        history_native = grid_targets(old_gt, both, grid_hw, (dimensions[:, 1], dimensions[:, 0]))
        same_cell = current_native == history_native
        delta_masks = dict(masks, native_same=both & same_cell, native_different=both & ~same_cell)
        delta_groups = {}
        for name, mask in delta_masks.items():
            endpoints = mask & both
            valid = endpoints & covered
            pair = valid & (error[:, 0] >= 16)
            n, nc, npair = int(endpoints.sum()), int(valid.sum()), int(pair.sum())
            if not n:
                continue
            delta_groups[name] = {
                'both_visible_n': n, 'current_candidate_covered4_n': nc,
                'conditional_history_pck16': float(hit16[valid].mean()) if nc else None,
                'joint_current4_history16_n': int(np.sum(valid & hit16)),
                'joint_current4_history16_rate': float(np.sum(valid & hit16)/n),
                **{f'conditional_history_r{k}': float(np.mean(ranks[valid] <= k)) if nc else None
                   for k in (1, 5, 10)},
                'far_wrong_top1_pair': {'n': npair,
                    'correct_score_wins': int(np.sum(difference[pair] > 0)),
                    'ties': int(np.sum(difference[pair] == 0)),
                    'correct_score_loses': int(np.sum(difference[pair] < 0)),
                    'mean_difference': float(difference[pair].mean()) if npair else None}}
        result[str(delta)] = delta_groups
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有对应诊断')
    config = json.loads((args.run/'config.json').read_text())
    candidate_manifest = json.loads((args.candidates.parent/'results.json').read_text())
    assert args.candidates.name == 'cross_address.npz'
    source = candidate_manifest['models']['cross_address']
    assert Path(source['source_run']).resolve() == args.run.resolve() and source['checkpoint_epoch'] == 3
    assert config['interaction'] == 'cross_address' and config['temporal_input'] == 'history'
    metadata = json.loads((ROOT/config['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    all_windows, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = np.array([w for w in all_windows if frames[w[-1]]['split'] == 'val'])
    rows = [frames[w[-1]] for w in windows]
    assert len(rows) == 14192
    with np.load(args.candidates) as data:
        xy, q = data['local_xy'], data['q']
        np.testing.assert_array_equal(data['current_frame_ids'], windows[:, -1])
    assert xy.shape == (len(rows), 16, 2)
    rgb = np.load(ROOT/config['rgb_cache']/'rgb.npy', mmap_mode='r')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    checkpoint = torch.load(args.run/'best.pt', map_location=device, weights_only=True)
    assert checkpoint['epoch'] == 3
    model = build_dino_model(config['weights'], upscale=8, interaction='cross_address').to(device).eval()
    model.load_state_dict(checkpoint['model'])
    args.output.mkdir(parents=True)
    chunks = {level: {'max_cosine': [], 'matched_cells': [], 'true_ranks': []} for level in ('stage0', 'stage1')}
    sizes = {'stage0': (72, 128), 'stage1': (36, 64)}
    cached = {}
    encoded_frames = 0
    reference_frames = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for start in range(0, len(rows), config['batch_size']):
            ws = windows[start:start+config['batch_size']]
            batch_rows = rows[start:start+len(ws)]
            new = np.array([i for i in np.unique(ws) if int(i) not in cached])
            if len(new):
                pixels = torch.from_numpy(rgb[new]).to(device)
                normalized = (pixels.float()/255-model.mean)/model.std
                shallow = model.prefix[1](model.prefix[0](normalized))
                deep = model.prefix[3](model.prefix[2](shallow))
                if start == 0:
                    torch.testing.assert_close(deep, model.encode(pixels), rtol=0, atol=0)
                    reference_frames = len(new)
                encoded_frames += len(new)
                cached.update({int(frame): (shallow[j], deep[j]) for j, frame in enumerate(new)})
            query_xy = torch.as_tensor(xy[start:start+len(ws)], device=device, dtype=torch.float32)
            dimensions = torch.tensor([[r['width'], r['height']] for r in batch_rows], device=device)
            history_rows = [[frames[w[1]], frames[w[0]]] for w in ws]
            history_xy = np.array([[[r['x_raw'], r['y_raw']] for r in rs] for rs in history_rows])
            history_wh = np.array([[[r['width'], r['height']] for r in rs] for rs in history_rows])
            history_visible = np.array([[r['visibility_raw'] == 1 for r in rs] for rs in history_rows])
            for level_index, (level, (height, width)) in enumerate(sizes.items()):
                features = torch.stack([torch.stack([cached[int(i)][level_index] for i in w]) for w in ws])
                costs = candidate_costs(features, query_xy, dimensions)
                value, cell = costs.max(-1)
                native = grid_targets(history_xy.reshape(-1, 2), history_visible.ravel(),
                    (height, width), (history_wh[..., 1].ravel(), history_wh[..., 0].ravel())).reshape(len(ws), 2)
                target_cell = np.where(history_visible, native, 0)
                ranks = true_cell_ranks(costs, torch.as_tensor(target_cell, device=device))
                ranks.masked_fill_(~torch.as_tensor(history_visible, device=device)[:, :, None], -1)
                for name, tensor in (('max_cosine', value), ('matched_cells', cell), ('true_ranks', ranks)):
                    chunks[level][name].append(tensor.cpu().numpy())
                del features, costs
            # 独立clone使保留两帧不会间接保留上一整批特征的storage。
            cached = {int(i): tuple(f.clone() for f in cached[int(i)]) for i in ws[-1, -2:]}
            if start % 1600 == 0:
                print(f'candidates: {start+len(ws)}/{len(rows)}; unique encoded: {encoded_frames}', flush=True)
    assert encoded_frames == len(np.unique(windows)), '生产路径重复提取了源帧'
    arrays = {level: {key: np.concatenate(value) for key, value in values.items()}
              for level, values in chunks.items()}
    forward_seconds = time.perf_counter()-started
    result = {'protocol': 'blurball-candidate-correspondence-v1', 'source_run': str(args.run),
        'checkpoint_epoch': 3, 'visual_support': ['t-2', 't-1', 't'], 'target_count': len(rows),
        'unique_encoded_frames': encoded_frames, 'first_batch_reference_frames': reference_frames,
        'first_batch_encode_exact': True, 'prefix_and_matching_seconds': forward_seconds,
        'device': torch.cuda.get_device_name(device), 'precision': 'float32 prefix and cosine',
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['src/ballmotion/correspondence.py: candidate_costs',
                         'scripts/probe_blurball_candidate_correspondence.py'], 'levels': {}}
    masks = _group_masks(rows, frames, windows)
    with np.load(args.candidates.parent/'same_address.npz') as same:
        np.testing.assert_array_equal(same['current_frame_ids'], windows[:, -1])
        same_xy = same['local_xy'][:, 0]
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows])
    vis21 = np.array([r['visibility_raw'] == 1 and r['game'] == 'match21' for r in rows])
    same_error = np.linalg.norm(same_xy-gt, axis=1)
    old_error = np.linalg.norm(xy[:, 0]-gt, axis=1)
    masks['fixed/match21_lost16'] = vis21 & (same_error < 16) & (old_error >= 16)
    masks['fixed/match21_rescued16'] = vis21 & (same_error >= 16) & (old_error < 16)
    assert masks['fixed/match21_lost16'].sum() == 202 and masks['fixed/match21_rescued16'].sum() == 74
    for level, values in arrays.items():
        np.savez(args.output/f'{level}.npz', current_frame_ids=windows[:, -1], **values)
        selected = values['max_cosine'].mean(axis=1).argmax(axis=1)
        predicted = xy[np.arange(len(rows)), selected]
        write_predictions(args.output/f'{level}_val_predictions.csv', rows, predicted, q)
        result['levels'][level] = {'grid_hw': list(sizes[level]),
            'conditional_matching': summarize_matches(rows, frames, windows, xy, values, sizes[level], masks),
            'support_readout': {'changed_targets': int(np.sum(selected != 0)),
                'groups': groups(rows, predicted, q, xy[:, 0], masks, after='cosine_readout')}}
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('unique_encoded_frames', 'prefix_and_matching_seconds',
                                                  'peak_allocated_mib')}, indent=2))


if __name__ == '__main__':
    main()
