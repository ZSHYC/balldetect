"""用冻结层点特征学习固定K16候选的单帧分数残差。"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from ballmotion.blurball import evaluate_blurball
from compare_blurball_five_frame import comparison_groups
from compare_blurball_temporal import _validate_saved_rows
from compare_blurball_temporal_activation import baseline_error_masks
from train_blurball_midpoint import prepare_training_windows
from train_tennis_heatmap import write_predictions

LAYERS = ('pretrained_stage1', 'pretrained_stage2', 'adapted_stage1')


class LayerPointResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(channels, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, descriptors):
        return self.layers(descriptors).squeeze(-1)


def candidate_scores(model, descriptors, peak_logits):
    """只接受自动K16描述符；GT锚点不能进入读出。"""
    if descriptors.ndim != 3 or descriptors.shape[1] != 16:
        raise ValueError('读出必须且只能接收自动K16描述符')
    if peak_logits.shape != descriptors.shape[:2]:
        raise ValueError('点描述符与候选分数形状不一致')
    return peak_logits + model(descriptors)


def candidate_set_losses(logits, positive, allowed):
    """逐帧正例集合损失；ambiguous候选从分母和梯度中排除。"""
    masked = logits.masked_fill(~allowed, -torch.inf)
    return (torch.logsumexp(masked, dim=1)
            - torch.logsumexp(masked.masked_fill(~positive, -torch.inf), dim=1))


def eligible_frame_weights(preserve, eligible):
    """每个可监督帧等权；分组只用于报告，不改变训练分布。"""
    preserve = np.asarray(preserve, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    groups = {'preserve': eligible & preserve, 'recovery': eligible & ~preserve}
    counts = {name: int(mask.sum()) for name, mask in groups.items()}
    if min(counts.values()) == 0:
        raise ValueError(f'监督目标缺少训练组: {counts}')
    weights = np.zeros(len(eligible), dtype=np.float32)
    weights[eligible] = 1 / int(eligible.sum())
    return weights, counts


def supervision(rows, xy):
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows], dtype=np.float64)
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    lengths = np.array([r['l_raw'] for r in rows], dtype=np.float64)
    distance = np.linalg.norm(xy - gt[:, None], axis=2)
    positive = distance < 4
    far_threshold = np.maximum(16., lengths + 4.)
    far = distance >= far_threshold[:, None]
    eligible = visible & positive.any(axis=1) & far.any(axis=1)
    preserve = positive[:, 0]
    k1_far_recoverable = visible & positive.any(axis=1) & far[:, 0]
    stats = {
        'visible': int(visible.sum()),
        'eligible': int(eligible.sum()),
        'eligible_preserve': int(np.sum(eligible & preserve)),
        'eligible_recovery': int(np.sum(eligible & ~preserve)),
        'k1_far_recoverable': int(k1_far_recoverable.sum()),
        'k1_ambiguous_recoverable': int(np.sum(
            eligible & ~preserve & ~far[:, 0])),
        'positive_rule': 'candidate distance < 4 source pixels',
        'far_rule': 'candidate distance >= max(16, l_raw + 4) source pixels',
        'ambiguous_rule': 'all remaining candidates excluded from loss denominator',
    }
    return positive, positive | far, eligible, preserve, stats


def fixed_candidate_coverage(rows, xy):
    gt = np.array([[r['x_raw'], r['y_raw']] for r in rows], dtype=np.float64)
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    error = np.linalg.norm(xy - gt[:, None], axis=2)
    result = {'visible': int(visible.sum())}
    for k in (1, 16):
        best = error[:, :k].min(axis=1)
        for radius in (4, 16):
            count = int(np.sum(visible & (best < radius)))
            result[f'K{k}@{radius}'] = {
                'covered': count,
                'coverage': count / result['visible'],
            }
    return result


def _feature_batch(features, ids, device):
    # 第0列GT锚点在CPU侧即被切掉，训练和推理都不会传给模型。
    values = np.array(features[ids, 1:17], dtype=np.float32, copy=True)
    return torch.from_numpy(values).to(device)


@torch.no_grad()
def predict(model, features, peak_logits, xy, device, batch_size=256):
    model.eval()
    selected = []
    for start in range(0, len(xy), batch_size):
        ids = np.arange(start, min(start + batch_size, len(xy)))
        descriptors = _feature_batch(features, ids, device)
        peaks = torch.from_numpy(np.array(peak_logits[ids], copy=True)).to(device)
        selected.append(candidate_scores(model, descriptors, peaks).argmax(1).cpu().numpy())
    selected = np.concatenate(selected)
    return xy[np.arange(len(xy)), selected], selected


def load_data(features_root, layer):
    manifest = json.loads((features_root / 'manifest.json').read_text())
    if layer not in manifest['descriptors'] or manifest['point_order'] != (
            'column0=GT anchor (valid only for V1); columns1:17=unchanged automatic K16'):
        raise ValueError('点特征缓存层或列语义不符合协议')
    run = Path(manifest['source_run'])
    config = json.loads((run / 'config.json').read_text())
    if (manifest['adapted_checkpoint_epoch'] != 8 or config['interaction'] != 'cross_address'
            or config['protocol'] != 'blurball-centered-hflip-v1'
            or config['window'] != 'center5' or config['target_slot'] != 2
            or config['augmentation'] != 'hflip' or config['augmentation_probability'] != .5
            or config['train_targets'] != 37590 or config['val_targets'] != 13912):
        raise ValueError('点特征不是固定增强center5 best8来源')
    cache = Path(config['rgb_cache'])
    metadata = json.loads((cache / 'metadata.json').read_text())
    windows, all_rows, slot, _ = prepare_training_windows(
        metadata['frames'], metadata['windows'], config['continuity_boundaries'],
        config['window'], config['temporal_input'])
    if slot != 2:
        raise ValueError('目标帧位置不是center5的slot2')

    result = {}
    for split, expected_n in (('train', 37590), ('val', 13912)):
        mask = np.array([r['split'] == split for r in all_rows])
        rows = [r for r, keep in zip(all_rows, mask) if keep]
        frame_ids = windows[mask, slot]
        if len(rows) != expected_n:
            raise ValueError(f'{split}目标数不符合固定协议')
        feature_ids = np.load(features_root / split / 'frame_ids.npy', mmap_mode='r')
        np.testing.assert_array_equal(feature_ids, frame_ids)
        point_features = np.load(features_root / split / f'{layer}.npy', mmap_mode='r')
        expected_shape = (expected_n, 17, manifest['descriptors'][layer])
        if point_features.shape != expected_shape or point_features.dtype != np.float32:
            raise ValueError(f'{split}点特征形状或精度不符: {point_features.shape}')

        source = Path(manifest['splits'][split]['candidate_source'])
        source_manifest = json.loads((source / 'manifest.json').read_text())
        if (source_manifest['checkpoint_epoch'] != 8
                or Path(source_manifest['source_run']).resolve() != run.resolve()
                or source_manifest['target_split'] != split
                or source_manifest['target_count'] != expected_n):
            raise ValueError(f'{split}候选来源不符合固定协议')
        with np.load(source / 'candidates.npz') as saved:
            candidates = {key: saved[key] for key in
                          ('current_frame_ids', 'local_xy', 'peak_logits', 'q')}
        np.testing.assert_array_equal(candidates['current_frame_ids'], frame_ids)
        if (candidates['local_xy'].shape != (expected_n, 16, 2)
                or candidates['peak_logits'].shape != (expected_n, 16)
                or candidates['q'].shape != (expected_n,)):
            raise ValueError(f'{split}候选数组不是固定K16')
        if not (np.isfinite(candidates['local_xy']).all()
                and np.isfinite(candidates['peak_logits']).all()
                and np.isfinite(candidates['q']).all()):
            raise ValueError(f'{split}候选含非有限值')
        if np.any(np.argmax(candidates['peak_logits'], axis=1) != 0):
            raise ValueError(f'{split}候选第0列不是原峰值top1')
        _validate_saved_rows(rows, [metadata['frames'][int(i)] for i in frame_ids], source)
        positive, allowed, eligible, preserve, stats = supervision(
            rows, candidates['local_xy'])
        expected_stats = {
            'train': (32134, 31430, 30921, 509, 109, 400),
            'val': (12689, 11461, 10813, 648, 515, 133),
        }[split]
        actual_stats = tuple(stats[key] for key in (
            'visible', 'eligible', 'eligible_preserve', 'eligible_recovery',
            'k1_far_recoverable', 'k1_ambiguous_recoverable'))
        if actual_stats != expected_stats:
            raise ValueError(f'{split}监督群体偏离锁定协议: {actual_stats}')
        result[split] = {
            'rows': rows, 'features': point_features, **candidates,
            'positive': positive, 'allowed': allowed, 'eligible': eligible,
            'preserve': preserve, 'supervision': stats,
        }
    return manifest, config, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--layer', choices=LAYERS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('不覆盖已有层点读出目录')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device('cuda')
    loaded_at = time.perf_counter()
    manifest, _, data = load_data(args.features.resolve(), args.layer)
    train, val = data['train'], data['val']
    weights, group_counts = eligible_frame_weights(train['preserve'], train['eligible'])
    model = LayerPointResidual(manifest['descriptors'][args.layer]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    args.output.mkdir(parents=True)
    config = {
        'protocol': 'blurball-layer-points-v1',
        'features': str(args.features.resolve()), 'layer': args.layer,
        'source_run': str(Path(manifest['source_run']).resolve()),
        'source_checkpoint_epoch': 8, 'seed': 0, 'epochs': 30, 'batch_size': 256,
        'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': .01},
        'readout': f'{manifest["descriptors"][args.layer]} -> 32 -> GELU -> 1 residual; final linear zero initialized',
        'score': 'fixed peak_logits + learned residual',
        'input': 'automatic candidates columns1:17 only; GT anchor column0 excluded before H2D',
        'q': 'fixed source q', 'coordinates': 'fixed source local_xy',
        'objective': 'mean candidate-set loss over all eligible frames; no preserve/recovery reweighting',
        'weighting_decision': ('each eligible frame has weight 1/n_eligible; recovery already dominates '
                               'the zero-residual initial loss, so sample-count balancing is not used'),
        'training_groups': group_counts,
        'supervision': {split: values['supervision'] for split, values in data.items()},
        'targets': {split: len(values['rows']) for split, values in data.items()},
        'selection': 'fixed epoch30; epoch0 and validation are diagnostics, no validation selection',
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'precision': 'float32 cached descriptors, residual MLP and peak logits; float64 coordinates',
        'device': torch.cuda.get_device_name(),
        'cache_load_seconds': time.perf_counter() - loaded_at,
        'code_revision': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['scripts/train_blurball_layer_points.py'],
    }
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    baseline_xy = val['local_xy'][:, 0]
    baseline_metrics = evaluate_blurball(val['rows'], baseline_xy, val['q'], grouped=False)
    initial_xy, initial_ids = predict(
        model, val['features'], val['peak_logits'], val['local_xy'], device)
    if np.any(initial_ids) or not np.array_equal(initial_xy, baseline_xy):
        raise ValueError('epoch0没有精确保留原候选top1')
    history = [{'epoch': 0, 'epoch0_top1_exact': True, 'val': baseline_metrics,
                'changed_targets': 0}]
    (args.output / 'history.jsonl').write_text(json.dumps(history[-1]) + '\n')
    torch.save({'epoch': 0, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict()}, args.output / 'last.pt')

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, 31):
        model.train()
        weighted_total = 0.
        group_loss = {'preserve': 0., 'recovery': 0.}
        visited = {'preserve': 0, 'recovery': 0}
        skipped = 0
        permutation = rng.permutation(len(train['rows']))
        epoch_started = time.perf_counter()
        for start in range(0, len(permutation), 256):
            batch_ids = permutation[start:start + 256]
            active = batch_ids[train['eligible'][batch_ids]]
            if not len(active):
                skipped += 1
                continue
            descriptors = _feature_batch(train['features'], active, device)
            peaks = torch.from_numpy(np.array(train['peak_logits'][active], copy=True)).to(device)
            positive = torch.from_numpy(train['positive'][active]).to(device)
            allowed = torch.from_numpy(train['allowed'][active]).to(device)
            frame_weights = torch.from_numpy(weights[active]).to(device)
            optimizer.zero_grad(set_to_none=True)
            losses = candidate_set_losses(
                candidate_scores(model, descriptors, peaks), positive, allowed)
            # 全训练目标均匀抽取的无偏mini-batch估计；零权重帧仍在permutation中。
            loss = (losses * frame_weights).sum() * len(train['rows']) / len(batch_ids)
            if not torch.isfinite(loss):
                raise ValueError('层点残差训练loss非有限')
            loss.backward()
            optimizer.step()
            detached = losses.detach().cpu().numpy()
            weighted_total += float(np.sum(detached * weights[active]))
            active_preserve = train['preserve'][active]
            for name, selected in (('preserve', active_preserve),
                                   ('recovery', ~active_preserve)):
                group_loss[name] += float(detached[selected].sum())
                visited[name] += int(selected.sum())
        if visited != group_counts:
            raise ValueError(f'训练组未完整访问: {visited} != {group_counts}')
        xy, selected = predict(model, val['features'], val['peak_logits'],
                               val['local_xy'], device)
        metrics = evaluate_blurball(val['rows'], xy, val['q'], grouped=False)
        record = {
            'epoch': epoch, 'train_eligible_mean_loss': weighted_total,
            'train_group_mean_loss': {name: group_loss[name] / visited[name]
                                      for name in visited},
            'visited_targets': len(permutation), 'visited_eligible': visited,
            'skipped_zero_supervision_batches': skipped,
            'val': metrics, 'changed_targets': int(np.sum(selected != 0)),
            'seconds': time.perf_counter() - epoch_started,
        }
        history.append(record)
        with (args.output / 'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record) + '\n')
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, args.output / 'last.pt')
        print(f'{args.layer} epoch {epoch}/30 loss={weighted_total:.6f} '
              f'val_PCK4={metrics["location"]["pck4"]:.6f} '
              f'val_F1_4={metrics["detection4"]["f1"]:.6f} '
              f'changed={record["changed_targets"]}', flush=True)

    training_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    # 最后一轮已对同一权重完成全验证预测，直接复用，避免重复forward。
    final_xy = xy
    final_metrics = evaluate_blurball(val['rows'], final_xy, val['q'], grouped=True)
    write_predictions(args.output / 'val_predictions.csv', val['rows'], final_xy, val['q'])
    train_xy, _ = predict(model, train['features'], train['peak_logits'],
                           train['local_xy'], device)
    train_metrics = evaluate_blurball(train['rows'], train_xy, train['q'], grouped=True)
    write_predictions(args.output / 'train_predictions.csv', train['rows'], train_xy, train['q'])
    error_masks = baseline_error_masks(val['rows'], val['local_xy'])
    comparisons = comparison_groups(
        val['rows'], baseline_xy, val['q'], final_xy, val['q'], error_masks,
        before_name='baseline_top1', after_name=args.layer)
    result = {
        'protocol': config['protocol'], 'completed_epochs': 30,
        'primary_checkpoint': 'last.pt', 'primary_epoch': 30,
        'baseline_val': baseline_metrics, 'last_val': final_metrics, 'last_train': train_metrics,
        'changed_targets': int(np.sum(selected != 0)),
        'supervision': config['supervision'],
        'fixed_candidate_coverage': fixed_candidate_coverage(val['rows'], val['local_xy']),
        'comparisons': comparisons,
        'training_and_validation_seconds': training_seconds,
        'final_prediction_comparison_and_write_seconds': time.perf_counter() - prediction_started,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
    }
    (args.output / 'summary.json').write_text(
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'layer': args.layer, 'last_val': final_metrics,
                      'changed_targets': result['changed_targets']}, indent=2))


if __name__ == '__main__':
    main()
