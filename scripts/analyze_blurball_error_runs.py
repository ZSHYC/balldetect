"""按真实帧号统计 BlurBall 已保存预测的连续输出状态，不运行模型。"""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from compare_blurball_temporal import _identity, read_predictions


STATES = ('v1_correct_emitted', 'v1_near_emitted', 'v1_far_emitted',
          'v1_rejected', 'v0_emitted', 'v0_rejected')


def frame_state(row):
    v = row['visibility_raw']
    if v not in (0, 1):
        raise ValueError(f'Unexpected BlurBall visibility: {v}')
    if row['presence_probability'] < .5:
        return f'v{v}_rejected'
    if v == 0:
        return 'v0_emitted'
    error = math.hypot(row['pred_x'] - row['x_raw'], row['pred_y'] - row['y_raw'])
    return 'v1_correct_emitted' if error < 4 else (
        'v1_near_emitted' if error < 16 else 'v1_far_emitted')


def bbox_diagonal(rows, x_key, y_key):
    return math.hypot(max(r[x_key] for r in rows) - min(r[x_key] for r in rows),
                      max(r[y_key] for r in rows) - min(r[y_key] for r in rows))


def build_runs(rows):
    groups = []
    previous = None
    for row in sorted(rows, key=_identity):
        state = frame_state(row)
        identity = _identity(row)
        if (previous is None or identity[:2] != previous[:2]
                or identity[2] != previous[2] + 1 or state != previous_state):
            groups.append([])
        groups[-1].append(row)
        previous, previous_state = identity, state
    runs = []
    for group in groups:
        first, last = group[0], group[-1]
        state = frame_state(first)
        emitted = state.endswith('_emitted')
        runs.append(dict(
            game=first['game'], clip=first['clip'], state=state,
            start_frame=first['original_frame_id'], end_frame=last['original_frame_id'],
            length=len(group),
            pred_bbox_diagonal_px=bbox_diagonal(group, 'pred_x', 'pred_y') if emitted else None,
            gt_bbox_diagonal_px=bbox_diagonal(group, 'x_raw', 'y_raw')
            if emitted and first['visibility_raw'] == 1 else None))
    return runs


def summarize(runs):
    result = {}
    for state in STATES:
        lengths = [r['length'] for r in runs if r['state'] == state]
        n = sum(lengths)
        long_frames = sum(length for length in lengths if length >= 4)
        result[state] = dict(
            frames=n, runs=len(lengths), max_length=max(lengths, default=0),
            length_histogram=dict(sorted(Counter(lengths).items())),
            frames_in_runs_ge4=long_frames,
            fraction_frames_in_runs_ge4=long_frames / n if n else None)
    return result


def select_visual_runs(runs):
    selected = []
    for game in sorted({r['game'] for r in runs}):
        candidates = [r for r in runs if r['game'] == game and r['state'] == 'v1_far_emitted']
        if candidates:
            run = min(candidates, key=lambda r: (-r['length'], r['clip'], r['start_frame']))
            selected.append(dict(run, target_frames=sorted({run['start_frame'],
                (run['start_frame'] + run['end_frame']) // 2, run['end_frame']})))
    return selected


def self_check():
    def row(frame, error, **changes):
        return dict(dict(game='match18', clip='001', original_frame_id=frame,
                         visibility_raw=1, x_raw=0., y_raw=0., pred_x=error, pred_y=0.,
                         presence_probability=.5), **changes)
    rows = [row(0, 3.999), row(1, 4), row(2, 15.999)]
    rows += [row(i, 16) for i in range(3, 7)] + [row(8, 16), row(9, 0, presence_probability=.49)]
    rows += [row(10, 0, visibility_raw=0, x_raw=float('nan')),
             row(11, 0, visibility_raw=0, presence_probability=.49),
             row(12, 0, clip='002', visibility_raw=0, presence_probability=.49),
             row(13, 0, game='match19', clip='002', visibility_raw=0, presence_probability=.49)]
    runs = build_runs(list(reversed(rows)))
    assert [r['length'] for r in runs] == [1, 2, 4, 1, 1, 1, 1, 1, 1]
    assert [r['state'] for r in runs] == [*STATES[:3], 'v1_far_emitted', *STATES[3:],
                                       'v0_rejected', 'v0_rejected']
    assert runs[5]['gt_bbox_diagonal_px'] is None
    assert runs[4]['pred_bbox_diagonal_px'] is None
    assert summarize(runs)['v1_far_emitted']['fraction_frames_in_runs_ge4'] == .8
    assert select_visual_runs(runs)[0]['target_frames'] == [3, 4, 6]
    assert select_visual_runs([runs[3]])[0]['target_frames'] == [8]


def analyze(predictions, output):
    rows, _, _ = read_predictions(predictions)
    if {r['game'] for r in rows} != {f'match{i}' for i in range(18, 22)}:
        raise ValueError('本诊断限定完整 match18–21 验证预测')
    if len({_identity(r) for r in rows}) != len(rows):
        raise ValueError('重复目标帧不能作为新的连续观察')
    runs = build_runs(rows)
    result = dict(
        predictions=str(predictions), n_targets=len(rows),
        definition='q>=0.5 emitted; V1 original-pixel error <4 / [4,16) / >=16; V0 has no position error',
        continuity='same game, clip, state and consecutive original_frame_id; no gap filling',
        spatial_extent='bounding-box diagonal is an upper bound on pairwise point distance, not exact diameter',
        overall=summarize(runs), by_game={game: summarize([r for r in runs if r['game'] == game])
                                         for game in sorted({r['game'] for r in runs})})
    output.mkdir(parents=True, exist_ok=True)
    # 样例按长度与身份确定；先保存选择，再由独立渲染步骤看 RGB。
    (output / 'selected_runs.json').write_text(json.dumps(select_visual_runs(runs), indent=2) + '\n')
    with (output / 'runs.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(n_targets=len(rows), overall=result['overall'],
                         selected=select_visual_runs(runs)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print('BlurBall error-run self-check passed.')
    else:
        if args.predictions is None or args.output is None:
            parser.error('--predictions and --output are required unless --self-check is used')
        analyze(args.predictions, args.output)
