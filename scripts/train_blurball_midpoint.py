"""BlurBall原生因果三帧的DINOv3中点定位基线，不使用blur标签训练。"""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ballmotion.blurball import continuous_windows, evaluate_blurball, source_coordinates
from ballmotion.tennis import grid_targets
from train_tennis_heatmap import build_dino_model, model_input, predict, write_predictions

GRID_HW = (288, 512)


def compact(metrics):
    return {k: metrics[k] for k in ('location', 'detection4', 'author_detection4',
                                    'detection8', 'visibility')}


def save_training_state(path, model, optimizer, rng, progress):
    state = {**progress, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def restore_training_state(path, model, optimizer, rng):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--boundaries', type=Path,
                        default=ROOT / 'configs/blurball_continuity_boundaries.csv')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--temporal-input', choices=('history', 'repeat_current'), default='history')
    parser.add_argument('--resume', action='store_true', help='从输出目录的last.pt恢复完整轮次状态')
    args = parser.parse_args()
    if args.resume:
        if not (args.output / 'last.pt').exists() or not (args.output / 'config.json').exists():
            raise ValueError('续训需要原config.json和完整last.pt；仅best.pt不能无缝恢复')
        if (args.output / 'results.json').exists():
            raise ValueError('该实验已经完成，不重复续训')
    elif (args.output / 'config.json').exists():
        raise ValueError('输出已有实验配置，不覆盖现存运行')
    metadata = json.loads((args.rgb_cache / 'metadata.json').read_text())
    expected = {'dataset': 'blurball', 'matches': list(range(22)), 'input_hw': [288, 512],
                'history': 2, 'target_step': 1, 'resize': 'FFmpeg bilinear RGB',
                'annotation': 'streak midpoint'}
    if any(metadata['config'].get(k) != v for k, v in expected.items()):
        raise ValueError('需要协议v1的BlurBall完整00–21中点RGB缓存')
    frames = metadata['frames']
    if {r['match'] for r in frames} != {f'{i:02d}' for i in range(22)}:
        raise ValueError('开发源帧范围需要精确为match00–21')
    with args.boundaries.open(newline='') as handle:
        boundaries = list(csv.DictReader(handle))
    windows, removed = continuous_windows(frames, metadata['windows'], boundaries)
    removed_rows = [frames[metadata['windows'][i][-1]] for i in removed]
    if args.temporal_input == 'repeat_current':
        windows = np.repeat(windows[:, -1:], 3, axis=1)
    rows = [frames[i] for i in windows[:, -1]]
    train_idx = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val_idx = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    if (len(train_idx) + sum(r['split'] == 'train' for r in removed_rows) != metadata['train_targets']
            or len(val_idx) + sum(r['split'] == 'val' for r in removed_rows) != metadata['val_targets']
            or not len(train_idx) or not len(val_idx)):
        raise ValueError('目标划分与缓存不一致')
    rgb = np.load(args.rgb_cache / 'rgb.npy', mmap_mode='r')
    if rgb.shape != (len(frames), 3, 288, 512) or rgb.dtype != np.uint8:
        raise ValueError('RGB缓存尺寸或类型不符')
    targets = grid_targets([[r['x_raw'], r['y_raw']] for r in rows],
                           [r['visibility_raw'] == 1 for r in rows], GRID_HW,
                           ([r['height'] for r in rows], [r['width'] for r in rows]))

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda')
    weights = ROOT / ('models/pretrained/dinov3/lvd1689m/'
                      'dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth')
    model = build_dino_model(weights, upscale=8).to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.head.parameters(), 'lr': 3e-4, 'name': 'head'},
        {'params': model.prefix.parameters(), 'lr': 1e-5, 'name': 'prefix'},
    ], weight_decay=.01)
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                 if k != 'resume'},
              'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                        text=True).strip(),
              'weights': str(weights), 'cache_config': metadata['config'],
              'protocol': ('blurball-full-temporal-control-v1' if args.temporal_input == 'repeat_current'
                           else 'blurball-causal-midpoint-v2'),
              'continuity_boundaries': boundaries,
              'excluded_boundary_targets': [{k: r[k] for k in ('game', 'clip', 'original_frame_id')}
                                            for r in removed_rows],
              'model': 'DINOv3 ConvNeXt-Tiny stages0–1 + random SpatialProbe',
              'head': {'input_channels': 576, 'hidden_channels': 32, 'upscale': 8,
                       'num_frames': 3, 'appearance_channels': 576},
              'output_grid_hw': GRID_HW, 'classes': 288*512+1,
              'input_slots': (['t', 't', 't'] if args.temporal_input == 'repeat_current'
                              else ['t-2', 't-1', 't']), 'target_slot': 2,
              'train_targets': len(train_idx), 'val_targets': len(val_idx),
              'loss': 'cross_entropy; V0=no valid visible midpoint; no theta/l supervision',
              'optimizer': {'name': 'AdamW', 'head_lr': 3e-4, 'prefix_lr': 1e-5,
                            'weight_decay': .01, 'schedule': 'constant'},
              'selection': 'maximum local detection F1@4; then F1@8; first ties including epoch0',
              'local_f1': 'FP=FP1+FP2; FN=FN_visible+FP1; strict original pixel radii',
              'author_f1': 'FP=FP1+FP2; FN=FN_visible; reported separately, not used to select',
              'presence_probability_column': 'probability of a valid visible midpoint; threshold 0.5',
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'device': torch.cuda.get_device_name(), 'torch_version': str(torch.__version__),
              'precision': 'float32; no AMP', 'augmentation': None,
              'timing_scope': 'RGB mmap + H2D + online model train/evaluation; excludes RGB cache creation'}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        saved_config = json.loads((args.output / 'config.json').read_text())
        current_config = json.loads(json.dumps(config))
        changed = [k for k, v in current_config.items()
                   if k != 'code_revision' and saved_config.get(k) != v]
        if changed:
            raise ValueError(f'续训配置与原实验不同：{changed}')
    else:
        (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config), flush=True)

    def predictions(ids):
        reference_xy, confidence = predict(model, rgb, windows, ids, args.batch_size,
                                            device, 'dino', GRID_HW)
        selected_rows = [rows[i] for i in ids]
        return source_coordinates(reference_xy, selected_rows), confidence

    val_rows = [rows[i] for i in val_idx]
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    elapsed_before = peak_before = 0.
    history = []

    def log(record):
        history.append(record)
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)

    def save_progress(epoch):
        save_training_state(args.output / 'last.pt', model, optimizer, rng,
                            {'epoch': epoch, 'history': history, 'initial': initial,
                             'best': best, 'best_checkpoint': best_checkpoint,
                             'elapsed_seconds': elapsed_before + time.perf_counter() - started,
                             'peak_allocated_mib': max(peak_before,
                                                      torch.cuda.max_memory_allocated() / 2**20)})

    if args.resume:
        state = restore_training_state(args.output / 'last.pt', model, optimizer, rng)
        initial, history = state['initial'], state['history']
        best, best_checkpoint = tuple(state['best']), state['best_checkpoint']
        best_epoch = best_checkpoint['epoch']
        elapsed_before, peak_before = state['elapsed_seconds'], state['peak_allocated_mib']
        start_epoch = state['epoch'] + 1
        # last.pt是完整轮次边界；撤回尚未提交进该快照的日志或best更新。
        (args.output / 'history.jsonl').write_text(
            ''.join(json.dumps(r, allow_nan=False) + '\n' for r in history))
        torch.save(best_checkpoint, args.output / 'best.pt')
        print(json.dumps({'resumed_after_epoch': state['epoch'],
                          'code_revision': config['code_revision']}), flush=True)
        del state
    else:
        xy, confidence = predictions(val_idx)
        initial = evaluate_blurball(val_rows, xy, confidence, grouped=False)
        best = (initial['detection4']['f1'], initial['detection8']['f1'])
        best_epoch = 0
        best_checkpoint = {'model': copy.deepcopy(model.state_dict()), 'epoch': 0}
        torch.save(best_checkpoint, args.output / 'best.pt')
        log({'epoch': 0, 'train_loss': None, 'val': compact(initial),
             'elapsed_seconds': time.perf_counter() - started})
        save_progress(0)
        start_epoch = 1

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        model.prefix.eval()
        order = rng.permutation(train_idx)
        loss_sum = 0.
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start + args.batch_size]
            logits = model(model_input(rgb, windows, ids, device, 'dino'))
            loss = F.cross_entropy(logits, torch.from_numpy(targets[ids]).to(device))
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            processed = start + len(ids)
            if (start // args.batch_size + 1) % 400 == 0 or processed == len(order):
                print(json.dumps({'epoch': epoch, 'targets': processed, 'total': len(order),
                                  'loss_so_far': loss_sum / processed,
                                  'epoch_seconds': time.perf_counter() - epoch_started}), flush=True)
        xy, confidence = predictions(val_idx)
        metrics = evaluate_blurball(val_rows, xy, confidence, grouped=False)
        score = (metrics['detection4']['f1'], metrics['detection8']['f1'])
        if score > best:
            best, best_epoch = score, epoch
            best_checkpoint = {'model': copy.deepcopy(model.state_dict()), 'epoch': epoch}
            torch.save(best_checkpoint, args.output / 'best.pt')
        log({'epoch': epoch, 'train_loss': loss_sum / len(order), 'val': compact(metrics),
             'epoch_seconds': time.perf_counter() - epoch_started,
             'elapsed_seconds': elapsed_before + time.perf_counter() - started})
        save_progress(epoch)

    checkpoint = torch.load(args.output / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    results = {'initial_val': initial, 'best_epoch': best_epoch}
    for split, ids in (('train', train_idx), ('val', val_idx)):
        xy, confidence = predictions(ids)
        selected_rows = [rows[i] for i in ids]
        results[split] = evaluate_blurball(selected_rows, xy, confidence)
        write_predictions(args.output / f'{split}_predictions.csv', selected_rows, xy, confidence)
    results.update(elapsed_seconds=elapsed_before + time.perf_counter() - started,
                   peak_allocated_mib=max(peak_before, torch.cuda.max_memory_allocated() / 2**20))
    (args.output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'best_epoch': best_epoch, 'val': compact(results['val']),
                      'elapsed_seconds': results['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
