"""同一冻结K16上的当前/同址/软对应残差读出，全部目标训练与评价。"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from ballmotion.blurball import evaluate_blurball
from ballmotion.candidate_readout import CandidateResidualReadout, candidate_targets
from compare_blurball_temporal import _group_masks, continuous_windows
from rerank_blurball_candidates import groups
from train_tennis_heatmap import write_predictions


@torch.no_grad()
def predict(model, data):
    model.eval()
    ids = []
    for start in range(0, len(data['xy']), 256):
        sl = slice(start, start+256)
        ids.append(model(data['query'][sl], data['history'][sl], data['scores'][sl]).argmax(1).cpu().numpy())
    ids = np.concatenate(ids)
    return data['xy'][np.arange(len(ids)), ids], ids


def rank(metrics):
    return metrics['frame_counts4']['tp'], round(metrics['location']['pck4']*metrics['location']['n'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', choices=('current', 'stationary', 'correspondence'), required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise ValueError('已有训练目录需要明确resume，不能覆盖')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device('cuda')
    manifest = json.loads((args.features/'manifest.json').read_text())
    assert manifest['checkpoint_epoch'] == 3 and manifest['channels'] == 96 and manifest['temperature'] == .1
    run = Path(manifest['source_run'])
    source_config = json.loads((run/'config.json').read_text())
    metadata = json.loads((ROOT/manifest['rgb_cache']/'metadata.json').read_text())
    frames = metadata['frames']
    legal, _ = continuous_windows(frames, metadata['windows'], source_config['continuity_boundaries'])
    windows = {s: np.array([w for w in legal if frames[w[-1]]['split'] == s]) for s in ('train', 'val')}
    rows = {s: [frames[w[-1]] for w in ws] for s, ws in windows.items()}
    data = {}
    loaded = time.perf_counter()
    for split in ('train', 'val'):
        directory = args.features/split
        np.testing.assert_array_equal(np.load(directory/'current_frame_ids.npy'), windows[split][:, -1])
        with np.load(manifest['candidate_sources'][split]) as f:
            np.testing.assert_array_equal(f['current_frame_ids'], windows[split][:, -1])
            data[split] = {'xy': f['local_xy'], 'q': f['q'],
                           'scores': torch.as_tensor(f['peak_logits'], device=device)}
        d = data[split]
        d['query'] = torch.from_numpy(np.load(directory/'query.npy')).to(device)
        d['history'] = (torch.zeros((len(rows[split]), 2, 16, 96), device=device)
                        if args.condition == 'current'
                        else torch.from_numpy(np.load(directory/f'{args.condition}.npy')).to(device))
        gt = np.array([[r['x_raw'], r['y_raw']] for r in rows[split]])
        vis = np.array([r['visibility_raw'] == 1 for r in rows[split]])
        mask = vis & (np.linalg.norm(d['xy']-gt[:, None], axis=-1).min(1) < 4)
        d['supervised'] = torch.as_tensor(mask, device=device)
        if split == 'train':
            d['targets'] = candidate_targets(torch.from_numpy(d['xy']), torch.from_numpy(gt)).float().to(device)
    assert len(rows['train']) == 38854 and len(rows['val']) == 14192
    model = CandidateResidualReadout().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    config = {'protocol': 'blurball-candidate-residual-v1', 'features': str(args.features.resolve()),
        'condition': args.condition, 'seed': 0, 'epochs': 30, 'batch_size': 256,
        'optimizer': {'name': 'AdamW', 'lr': 3e-4, 'weight_decay': .01},
        'candidate_source': manifest['candidate_sources'], 'source_epoch': 3,
        'visual_support': ['t-2', 't-1', 't'], 'q': 'fixed cross', 'coordinate': 'fixed candidate local_xy',
        'loss': 'candidate soft-target CE, sigma4px, current V1 and min_error<4 only',
        'selection': 'maximum val TP4, then raw correct4, earliest incl epoch0',
        'targets': {s: len(rs) for s, rs in rows.items()},
        'supervised_targets': {s: int(d['supervised'].sum()) for s, d in data.items()},
        'trainable_parameters': sum(p.numel() for p in model.parameters()),
        'device': torch.cuda.get_device_name(), 'precision': 'float32 features/MLP; float64 original xy',
        'cache_load_seconds': time.perf_counter()-loaded,
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'code_changes': ['src/ballmotion/candidate_readout.py', 'scripts/train_blurball_candidate_residual.py']}
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    start_epoch = 1
    baseline = evaluate_blurball(rows['val'], data['val']['xy'][:, 0], data['val']['q'], grouped=False)
    best_metrics, best_epoch = baseline, 0
    elapsed_prior = 0.

    def checkpoint(epoch):
        return {'epoch': epoch, 'model': copy.deepcopy(model.state_dict()),
                'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
                'torch_rng': torch.get_rng_state(), 'best_epoch': best_epoch,
                'best_metrics': best_metrics, 'elapsed_seconds': elapsed_prior+time.perf_counter()-started}

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if args.resume:
        previous = json.loads((args.output/'config.json').read_text())
        for key in ('features', 'condition', 'seed', 'epochs', 'batch_size', 'optimizer', 'loss'):
            assert previous[key] == config[key], f'resume配置不符: {key}'
        state = torch.load(args.output/'last.pt', map_location=device, weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        start_epoch, best_epoch = state['epoch']+1, state['best_epoch']
        best_metrics = state['best_metrics']
        elapsed_prior = state['elapsed_seconds']
        records = [json.loads(line) for line in (args.output/'history.jsonl').read_text().splitlines()
                   if json.loads(line)['epoch'] <= state['epoch']]
        (args.output/'history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    else:
        initial_xy, ids = predict(model, data['val'])
        assert not ids.any(), 'epoch0未保留原候选top1'
        np.testing.assert_array_equal(initial_xy, data['val']['xy'][:, 0])
        (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
        records.append({'epoch': 0, 'val': baseline, 'epoch0_top1_exact': True})
        (args.output/'history.jsonl').write_text(json.dumps(records[-1])+'\n')
        torch.save(checkpoint(0), args.output/'best.pt')
        torch.save(checkpoint(0), args.output/'last.pt')
    train = data['train']
    for epoch in range(start_epoch, 31):
        model.train()
        total_loss = 0.
        count = skipped = 0
        permutation = rng.permutation(len(rows['train']))
        epoch_start = time.perf_counter()
        for start in range(0, len(permutation), 256):
            ids = torch.as_tensor(permutation[start:start+256], device=device)
            mask = train['supervised'][ids]
            valid = int(mask.sum())
            if not valid:
                skipped += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(train['query'][ids], train['history'][ids], train['scores'][ids])
            loss = -(train['targets'][ids][mask]*F.log_softmax(logits[mask], dim=1)).sum(1).mean()
            if not torch.isfinite(loss):
                raise ValueError('candidate训练loss非有限')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()*valid
            count += valid
        assert count == config['supervised_targets']['train']
        xy, selected = predict(model, data['val'])
        metrics = evaluate_blurball(rows['val'], xy, data['val']['q'], grouped=False)
        improved = rank(metrics) > rank(best_metrics)
        if improved:
            best_epoch, best_metrics = epoch, metrics
        record = {'epoch': epoch, 'train_loss': total_loss/count, 'supervised_n': count,
                  'visited_targets': len(permutation), 'skipped_no_label_batches': skipped,
                  'val': metrics, 'changed_targets': int((selected != 0).sum()),
                  'seconds': time.perf_counter()-epoch_start, 'best_epoch': best_epoch}
        records.append(record)
        with (args.output/'history.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        state = checkpoint(epoch)
        if improved:
            torch.save(state, args.output/'best.pt')
        torch.save(state, args.output/'last.pt')
        print(f'{args.condition} epoch {epoch}/30 loss={record["train_loss"]:.5f} '
              f'val TP4={rank(metrics)[0]} raw4={rank(metrics)[1]} best={best_epoch}', flush=True)
    best = torch.load(args.output/'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(best['model'])
    val_xy, selected = predict(model, data['val'])
    final_metrics = evaluate_blurball(rows['val'], val_xy, data['val']['q'], grouped=False)
    assert final_metrics == best_metrics, 'best模型未复现选优指标'
    for split in ('train', 'val'):
        xy, _ = predict(model, data[split]) if split == 'train' else (val_xy, selected)
        write_predictions(args.output/f'{split}_predictions.csv', rows[split], xy, data[split]['q'])
    masks = _group_masks(rows['val'], frames, windows['val'])
    result = {'config': config, 'completed_epochs': 30, 'best_epoch': best_epoch,
              'best_reproduced': True, 'best_val': final_metrics,
              'elapsed_seconds_training_validation_and_final_predictions': elapsed_prior+time.perf_counter()-started,
              'peak_allocated_mib_including_loaded_cache': torch.cuda.max_memory_allocated()/2**20,
              'groups': groups(rows['val'], val_xy, data['val']['q'], data['val']['xy'][:, 0], masks,
                               after=args.condition)}
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
