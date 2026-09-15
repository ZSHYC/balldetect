"""同一目标图像的官方stage1/2与已适配stage1点特征；不保存整幅特征。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_tennis_heatmap import ConvNeXt, build_dino_model
from train_blurball_midpoint import prepare_training_windows


def sample_layer_points(features, xy, image_wh):
    """逐帧无仿射GroupNorm，按源像素中心坐标双线性采样。"""
    normalized = F.group_norm(features, 1)
    grid = 2 * (xy + .5) / image_wh[:, None] - 1
    values = F.grid_sample(normalized, grid[:, :, None], mode='bilinear',
                           padding_mode='border', align_corners=False)
    return values[..., 0].transpose(1, 2)


def official_layers(model, pixels):
    """原生stride8/16，不调用会继续计算未使用stage3的全模型接口。"""
    features = pixels
    result = {}
    for stage in range(3):
        features = model.stages[stage](model.downsample_layers[stage](features))
        if stage in (1, 2):
            result[f'pretrained_stage{stage}'] = features
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有点特征目录')
    run = ROOT / 'outputs/blurball/centered_hflip/center5_seed0'
    config = json.loads((run / 'config.json').read_text())
    metadata = json.loads((Path(config['rgb_cache']) / 'metadata.json').read_text())
    frames = metadata['frames']
    windows, rows, slot, _ = prepare_training_windows(
        frames, metadata['windows'], config['continuity_boundaries'], 'center5', 'history')
    sources = {'train': run / 'train_candidates', 'val': run / 'candidate_coverage'}
    torch.set_num_threads(4)
    device = torch.device('cuda')
    official = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
    official.load_state_dict(torch.load(config['weights'], map_location='cpu',
                                        weights_only=True, mmap=True), strict=True)
    official = official.to(device).eval()
    adapted = build_dino_model(config['weights'], upscale=8, interaction='cross_address',
                               num_frames=5, target_slot=2)
    checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['epoch'] != 8:
        raise ValueError('本层间诊断固定现有增强best8')
    adapted.load_state_dict(checkpoint['model'])
    adapted = adapted.to(device).eval()
    del checkpoint
    rgb = np.load(Path(config['rgb_cache']) / 'rgb.npy', mmap_mode='r')
    args.output.mkdir(parents=True)
    result = {
        'source_run': str(run), 'adapted_checkpoint_epoch': 8,
        'official_weights': config['weights'], 'input_hw': [288, 512],
        'normalization': 'ImageNet RGB; per-frame per-layer GroupNorm(1,C,affine=False)',
        'descriptors': {'pretrained_stage1': 192, 'pretrained_stage2': 384, 'adapted_stage1': 192},
        'stride': {'pretrained_stage1': 8, 'pretrained_stage2': 16, 'adapted_stage1': 8},
        'point_order': 'column0=GT anchor (valid only for V1); columns1:17=unchanged automatic K16',
        'scope': 'target RGB only; full official prefix for official layers; no hybrid prefix',
        'dtype': 'float32 forward and storage', 'batch_size': args.batch_size,
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'device': torch.cuda.get_device_name(), 'splits': {},
    }
    with torch.inference_mode():
        for split, source in sources.items():
            mask = np.array([r['split'] == split for r in rows])
            frame_ids = windows[mask, slot]
            selected_rows = [r for r, keep in zip(rows, mask) if keep]
            source_info = json.loads((source / 'manifest.json').read_text())
            if (source_info['checkpoint_epoch'] != 8 or source_info['source_run'] != str(run)
                    or source_info['target_split'] != split or source_info['target_slot'] != 2):
                raise ValueError(f'{source}不是固定增强best8的对应目标')
            with np.load(source / 'candidates.npz') as a:
                np.testing.assert_array_equal(a['current_frame_ids'], frame_ids)
                candidates = a['local_xy']
            gt = np.array([[r['x_raw'], r['y_raw']] for r in selected_rows])
            points = np.concatenate((gt[:, None], candidates), axis=1).astype(np.float32)
            wh = np.array([[r['width'], r['height']] for r in selected_rows], dtype=np.float32)
            dest = args.output / split
            dest.mkdir()
            np.save(dest / 'frame_ids.npy', frame_ids)
            arrays = {name: np.lib.format.open_memmap(
                dest / f'{name}.npy', mode='w+', dtype=np.float32, shape=(len(frame_ids), 17, channels))
                for name, channels in result['descriptors'].items()}
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            for start in range(0, len(frame_ids), args.batch_size):
                end = min(start + args.batch_size, len(frame_ids))
                pixels = torch.from_numpy(rgb[frame_ids[start:end]]).to(device)
                pixels = (pixels.float() / 255 - adapted.mean) / adapted.std
                features = official_layers(official, pixels)
                features['adapted_stage1'] = adapted.prefix(pixels)
                xy = torch.from_numpy(points[start:end]).to(device)
                image_wh = torch.from_numpy(wh[start:end]).to(device)
                for name, value in features.items():
                    arrays[name][start:end] = sample_layer_points(value, xy, image_wh).cpu().numpy()
                if start % 1600 == 0:
                    print(f'{split}: {end}/{len(frame_ids)} target frames; three descriptors', flush=True)
            for value in arrays.values():
                value.flush()
            result['splits'][split] = {
                'targets': len(frame_ids), 'unique_target_frames': len(np.unique(frame_ids)),
                'candidate_source': str(source), 'seconds': time.perf_counter() - started,
                'array_bytes': sum(p.stat().st_size for p in dest.glob('*.npy')),
                'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            }
    (args.output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['splits'], indent=2))


if __name__ == '__main__':
    main()
