"""用cross-address固定best在指定BlurBall划分上一次性导出K16候选。"""
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
from compare_blurball_temporal import _validate_saved_rows, read_predictions
from probe_blurball_candidates import extract_candidate_arrays
from train_blurball_midpoint import WINDOW_OFFSETS, prepare_training_windows
from train_tennis_heatmap import build_dino_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'不覆盖已有候选导出: {args.output}')

    started = time.perf_counter()
    config = json.loads((args.run / 'config.json').read_text())
    run_results = json.loads((args.run / 'results.json').read_text())
    if config.get('interaction') not in ('cross_address', 'target_activation'):
        raise ValueError('--run必须是cross_address或target_activation运行')
    if config['temporal_input'] != 'history':
        raise ValueError('--run必须使用真实上下文输入')

    cache = ROOT / config['rgb_cache']
    metadata = json.loads((cache / 'metadata.json').read_text())
    frames = metadata['frames']
    all_windows, all_rows, target_slot, _ = prepare_training_windows(
        frames, metadata['windows'], config['continuity_boundaries'],
        config.get('window'), config['temporal_input'])
    selected = np.array([r['split'] == args.split for r in all_rows])
    windows = all_windows[selected]
    rows = [r for r, keep in zip(all_rows, selected) if keep]
    if len(rows) != int(config[f'{args.split}_targets']):
        raise ValueError('导出目标数与源运行不一致')
    if target_slot != config.get('target_slot', 2):
        raise ValueError('候选目标槽与源运行不一致')
    offsets = WINDOW_OFFSETS[config['window']] if config.get('window') else (-2, -1, 0)

    prediction_path = args.run / f'{args.split}_predictions.csv'
    saved_rows, original_xy, original_q = read_predictions(prediction_path)
    _validate_saved_rows(rows, saved_rows, prediction_path)
    local_xy = None
    if args.split == 'val':
        local_path = args.run/'local_readout/val_predictions.csv'
        local_rows, local_xy, local_q = read_predictions(local_path)
        _validate_saved_rows(rows, local_rows, local_path)
        np.testing.assert_array_equal(original_q, local_q)

    torch.set_num_threads(4)
    device = torch.device('cuda')
    checkpoint = torch.load(args.run / 'best.pt', map_location=device, weights_only=True)
    if checkpoint['epoch'] != run_results['best_epoch']:
        raise ValueError('best.pt与运行记录中的best epoch不一致')
    model = build_dino_model(
        config['weights'], upscale=8, interaction=config['interaction'],
        num_frames=windows.shape[1], target_slot=target_slot).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    rgb = np.load(cache / 'rgb.npy', mmap_mode='r')

    torch.cuda.reset_peak_memory_stats()
    forward_started = time.perf_counter()
    with torch.inference_mode():
        arrays = extract_candidate_arrays(
            model, rgb, windows, rows, device, config['batch_size'], progress_label=args.split,
            target_slot=target_slot)
    forward_seconds = time.perf_counter() - forward_started

    if not np.array_equal(arrays['original_xy'][:, 0], original_xy):
        raise ValueError('K1 argmax没有复现全部保存预测')
    max_q_difference = float(np.max(np.abs(arrays['q'] - original_q)))
    if max_q_difference > 1e-6:
        raise ValueError(f'q复现误差超限: {max_q_difference}')
    if not np.array_equal(arrays['q'] >= .5, original_q >= .5):
        raise ValueError('q阈值输出没有复现全部保存预测')

    local_reproduced = None
    if local_xy is not None:
        np.testing.assert_allclose(arrays['local_xy'][:, 0], local_xy, rtol=0, atol=1e-10)
        local_reproduced = True

    args.output.mkdir(parents=True)
    artifact = args.output / 'candidates.npz'
    np.savez(artifact, **arrays)
    first, last = rows[0], rows[-1]
    manifest = {
        'protocol': ('blurball-cross-train-candidates-v1'
                     if config.get('window') is None and args.split == 'train'
                     else 'blurball-context-candidates-v1'),
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['shared extraction preserves explicit target slot and source temporal window'],
        'source_run': str(args.run.resolve()),
        'source_checkpoint': str((args.run / 'best.pt').resolve()),
        'checkpoint_epoch': checkpoint['epoch'],
        'interaction': config['interaction'],
        'input': [f't{d:+d}' if d else 't' for d in offsets],
        'input_frame_offsets': list(offsets),
        'target_slot': target_slot,
        'temporal_scope': 'source continuous windows within one clip/rally; target is t',
        'target_split': args.split,
        'target_count': len(rows),
        'target_range': {
            'games': sorted({row['game'] for row in rows}),
            'first': {key: first[key] for key in ('game', 'clip', 'original_frame_id')},
            'last': {key: last[key] for key in ('game', 'clip', 'original_frame_id')},
        },
        'rgb_cache': str(cache.resolve()),
        'rgb_array': str((cache / 'rgb.npy').resolve()),
        'output': str(artifact.resolve()),
        'candidate_count': 16,
        'suppression_radius_cells': 7,
        'precision': 'float32 model/logits; float64 barycenter',
        'identity_and_labels_equal': True,
        'k1_argmax_exact': True,
        'max_q_difference': max_q_difference,
        'q_threshold_decisions_equal': True,
        'local_reference_available': args.split == 'val',
        'local_reproduced': local_reproduced,
        'forward_and_extraction_seconds': forward_seconds,
        'total_elapsed_seconds': time.perf_counter() - started,
        'timing_scope': 'load metadata/predictions/model + RGB mmap/H2D/forward/K16 extraction/verification/write',
        'device': torch.cuda.get_device_name(device),
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
    }
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: manifest[key] for key in (
        'target_count', 'checkpoint_epoch', 'max_q_difference',
        'forward_and_extraction_seconds', 'peak_allocated_mib')}, indent=2))


if __name__ == '__main__':
    main()
