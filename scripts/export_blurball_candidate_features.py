"""冻结cross stage0，为同地址/软对应候选读出共享一次逐帧提取。"""
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
from ballmotion.correspondence import (
    candidate_costs,
    candidate_pooled_features,
    top_candidate_matches,
)
from compare_blurball_temporal import continuous_windows
from train_tennis_heatmap import build_dino_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--representation', choices=('pooled', 'top_matches'), default='pooled')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有冻结候选特征')
    run = ROOT/'outputs/blurball/dino_cross_address_seed0'
    base = ROOT/'outputs/blurball/spatial_interaction'
    config = json.loads((run/'config.json').read_text())
    metadata = json.loads((ROOT/config['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    legal, _ = continuous_windows(frames, metadata['windows'], config['continuity_boundaries'])
    windows = {split: np.array([w for w in legal if frames[w[-1]]['split'] == split])
               for split in ('train', 'val')}
    assert len(windows['train']) == 38854 and len(windows['val']) == 14192
    paths = {'train': base/'candidate_temporal/train_candidates/candidates.npz',
             'val': base/'candidate_coverage/cross_address.npz'}
    train_manifest = json.loads((paths['train'].parent/'manifest.json').read_text())
    val_manifest = json.loads((paths['val'].parent/'results.json').read_text())['models']['cross_address']
    for manifest in (train_manifest, val_manifest):
        assert Path(manifest['source_run']).resolve() == run and manifest['checkpoint_epoch'] == 3
    candidates = {}
    for split, path in paths.items():
        with np.load(path) as data:
            np.testing.assert_array_equal(data['current_frame_ids'], windows[split][:, -1])
            candidates[split] = data['local_xy']
    rgb = np.load(ROOT/config['rgb_cache']/'rgb.npy', mmap_mode='r')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    model = build_dino_model(config['weights'], upscale=8, interaction='cross_address').to(device).eval()
    checkpoint = torch.load(run/'best.pt', map_location=device, weights_only=True)
    assert checkpoint['epoch'] == 3
    model.load_state_dict(checkpoint['model'])
    args.output.mkdir(parents=True)
    result = {'protocol': 'blurball-candidate-residual-v1', 'source_run': str(run), 'checkpoint_epoch': 3,
              'rgb_cache': config['rgb_cache'], 'candidate_sources': {k: str(v) for k, v in paths.items()},
              'stage': 0, 'channels': 96, 'grid_hw': [72, 128],
              'grid_cell_order': 'row-major: cell=y*128+x',
              'representation': args.representation,
              'source_feature': 'cross best3 DINOv3 ConvNeXt-Tiny stage0',
              'dtype': ('float32' if args.representation == 'pooled'
                        else {'scores': 'float32', 'cells': 'int16'}),
              'device': torch.cuda.get_device_name(),
              'float32_matmul_precision': torch.get_float32_matmul_precision(),
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'code_changes': ['src/ballmotion/correspondence.py: candidate feature representations',
                               'scripts/export_blurball_candidate_features.py'], 'splits': {}}
    if args.representation == 'pooled':
        result['temperature'] = .1
    else:
        result['protocol'] = 'blurball-candidate-addresses-v1'
        result['top_matches'] = {'count': 16, 'score': 'centered channel-normalized cosine',
                                 'cells': 'native stage0 grid addresses'}
    with torch.inference_mode():
        for split, ws in windows.items():
            dest = args.output/split
            dest.mkdir()
            if args.representation == 'pooled':
                specs = {'query': ((len(ws), 16, 96), np.float32),
                         'stationary': ((len(ws), 2, 16, 96), np.float32),
                         'correspondence': ((len(ws), 2, 16, 96), np.float32)}
            else:
                shape = (len(ws), 2, 16, 16)
                specs = {'scores': (shape, np.float32), 'cells': (shape, np.int16)}
            arrays = {key: np.lib.format.open_memmap(
                dest/f'{key}.npy', mode='w+', dtype=dtype, shape=shape)
                for key, (shape, dtype) in specs.items()}
            np.save(dest/'current_frame_ids.npy', ws[:, -1])
            cache = {}
            encoded = 0
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            for start in range(0, len(ws), config['batch_size']):
                batch = ws[start:start+config['batch_size']]
                new = np.array([f for f in np.unique(batch) if int(f) not in cache])
                pixels = torch.from_numpy(rgb[new]).to(device)
                shallow = model.prefix[1](model.prefix[0]((pixels.float()/255-model.mean)/model.std))
                cache.update({int(f): shallow[j] for j, f in enumerate(new)})
                encoded += len(new)
                features = torch.stack([torch.stack([cache[int(i)] for i in w]) for w in batch])
                xy = torch.as_tensor(candidates[split][start:start+len(batch)], device=device, dtype=torch.float32)
                wh = torch.tensor([[frames[w[-1]]['width'], frames[w[-1]]['height']] for w in batch], device=device)
                if args.representation == 'pooled':
                    values = candidate_pooled_features(features, xy, wh)
                else:
                    values = top_candidate_matches(candidate_costs(features, xy, wh))
                if split == 'val' and start == 0:
                    if args.representation == 'top_matches':
                        scores, cells = values[0][..., 0], values[1][..., 0]
                    else:
                        scores, cells = candidate_costs(features, xy, wh).max(-1)
                    with np.load(base/'candidate_correspondence/stage0.npz') as previous:
                        np.testing.assert_array_equal(cells.cpu().numpy(), previous['matched_cells'][:len(batch)])
                        difference = float(np.max(np.abs(scores.cpu().numpy()-previous['max_cosine'][:len(batch)])))
                    assert difference < 1e-6, '已有验证对应未被新提取复现'
                    result['first_val_batch_reference'] = {'targets': len(batch), 'matched_cells_exact': True,
                                                          'max_cosine_difference': difference}
                for (key, array), value in zip(arrays.items(), values):
                    array[start:start+len(batch)] = value.cpu().numpy()
                cache = {int(i): cache[int(i)].clone() for i in batch[-1, -2:]}
                if start % 1600 == 0:
                    print(f'{split}: {start+len(batch)}/{len(ws)}; unique stage0 frames {encoded}', flush=True)
            assert encoded == len(np.unique(ws)), '重复提取源帧'
            for array in arrays.values():
                array.flush()
            seconds_key = ('seconds_prefix_pooling_and_write' if args.representation == 'pooled'
                           else 'seconds_prefix_top_matches_and_write')
            result['splits'][split] = {'targets': len(ws), 'unique_encoded_frames': encoded,
                seconds_key: time.perf_counter()-started,
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'array_bytes': sum(p.stat().st_size for p in dest.glob('*.npy'))}
    (args.output/'manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['splits'], indent=2))


if __name__ == '__main__':
    main()
