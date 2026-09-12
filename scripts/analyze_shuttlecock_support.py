"""训练侧近两帧联合几何支持，并按位移预选四个原视频核对样本。"""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from analyze_shuttlecock_displacement import ROOT, split_for
from analyze_tennis_history_age import describe_window, summarize
from ballmotion.tennis import causal_windows, center_pairs, grid_targets


def frame_rows(raw_rows, record):
    rows = []
    for raw in raw_rows:
        visibility = int(raw['Visibility'])
        if visibility not in (0, 1):
            raise ValueError(f'Unexpected Shuttlecock visibility: {visibility}')
        rows.append(dict(game=f"{record['source']}/match{record['match']}", clip=record['rally'],
                         original_frame_id=int(raw['Frame']), visibility_raw=visibility,
                         x_raw=float(raw['X']), y_raw=float(raw['Y']),
                         label_state='located' if visibility == 1 else 'not_visible'))
    return rows


def support_records(rows, original_hw):
    cells = grid_targets([[r['x_raw'], r['y_raw']] for r in rows],
                         [r['label_state'] == 'located' for r in rows], (36, 64), original_hw)
    windows, skipped = causal_windows(rows, target_step=1, history=2)
    return [describe_window(rows, cells, window, 1) for window in windows], skipped


def self_check():
    # 无可见历史、唯一可见历史不覆盖、仅远历史覆盖；缺 Frame6 不得重编号。
    raw = [dict(Frame=i, Visibility=v, X=x, Y=100) for i, v, x in
           ((0, 0, 0), (1, 0, 0), (2, 1, 100), (3, 1, 300),
            (4, 1, 110), (5, 1, 115), (7, 1, 115))]
    rows = frame_rows(raw, dict(source='Professional', match=1, rally='example'))
    records, skipped = support_records(rows, (720, 1280))
    assert [r['current_original_frame_id'] for r in records] == [2, 3, 4, 5]
    assert [r['vc1_support'] for r in records] == [
        'no_visible_history', 'visible_history_uncovered',
        'visible_history_covered', 'visible_history_covered']
    assert not records[2]['near_covered'] and records[2]['far_covered']
    assert skipped == 1


def analyze(output):
    manifest = ROOT / 'outputs/shuttlecock/development_motion/development_rallies.csv'
    data_root = ROOT / 'data/shuttlecock/original/TrackNetV2'
    if (output / 'summary.json').exists():
        raise ValueError('保留已有诊断，请指定新的输出目录')
    with manifest.open(newline='') as handle:
        rallies = [r for r in csv.DictReader(handle) if r['split'] == 'train']
    records, candidates, skipped = [], [], 0
    by_match = defaultdict(list)
    for rally in rallies:
        if split_for(rally['source'], int(rally['match'])) != 'train':
            raise ValueError(f'训练manifest超出既定开发范围: {rally["video"]}')
        with (data_root / rally['csv']).open(newline='') as handle:
            rows = frame_rows(csv.DictReader(handle), rally)
        current, missing = support_records(rows, (int(rally['height']), int(rally['width'])))
        skipped += missing
        for item in current:
            item['video'] = rally['video']
        records.extend(current)
        by_match[f"{rally['source']}/match{rally['match']}"].extend(current)
        pair = max(center_pairs(rows, 1), default=None,
                   key=lambda ab: math.hypot(ab[1]['x_raw']-ab[0]['x_raw'],
                                            ab[1]['y_raw']-ab[0]['y_raw']))
        if pair is None:
            continue
        previous, target = pair
        t = target['original_frame_id']
        lookup = {r['original_frame_id']: r for r in rows}
        candidates.append(dict(
            video=rally['video'], csv=rally['csv'], width=int(rally['width']),
            height=int(rally['height']), avg_frame_rate=rally['avg_frame_rate'],
            nb_frames=int(rally['nb_frames']), current_original_frame_id=t,
            displacement_px=math.hypot(target['x_raw']-previous['x_raw'],
                                       target['y_raw']-previous['y_raw']),
            original_frame_ids=list(range(max(0, t-2), min(int(rally['nb_frames']), t+3))),
            annotations=[lookup[i] for i in range(t-2, t+3) if i in lookup],
            h2_support=next((r for r in current if r['current_original_frame_id'] == t), None)))
    candidates.sort(key=lambda r: (-r['displacement_px'], r['video'], r['current_original_frame_id']))
    selected = candidates[:4]
    uncovered = [r for r in records if r['vc1_support'] == 'visible_history_uncovered']
    result = dict(
        manifest=str(manifest.relative_to(ROOT)), analysis_script='scripts/analyze_shuttlecock_support.py',
        scope='Professional match1-20 and Amateur match1-3 training labels only; current V1 only',
        label_version='original TrackNetV2; no corrected Test labels',
        window='exact original Frame [t-2,t-1,t] within published rally; V0 is not physical absence',
        geometry='36x64 cells at actual original size; near radius2, far radius4; GT geometry only',
        train_rallies=len(rallies), visible_targets_without_full_label_window=skipped,
        overall=summarize(records), by_match={key: summarize(value) for key, value in sorted(by_match.items())},
        selection='largest V1 adjacent displacement per rally, then four largest distinct rallies; before image inspection',
        uncovered_windows=len(uncovered))
    output.mkdir(parents=True, exist_ok=True)
    (output/'selected_pairs.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2)+'\n')
    if uncovered:
        with (output/'uncovered_windows.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(uncovered[0]))
            writer.writeheader()
            writer.writerows(uncovered)
    (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(overall=result['overall'], selected=[
        {k: r[k] for k in ('video','current_original_frame_id','displacement_px','h2_support')}
        for r in selected]), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print('Shuttlecock support self-check passed.')
    else:
        if args.output is None:
            parser.error('--output is required unless --self-check is used')
        analyze(args.output)
