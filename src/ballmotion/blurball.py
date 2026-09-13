"""BlurBall可见streak中点的原图评价；不把不可见等同于物理无球。"""
import numpy as np

from .probe import _counts


def temporal_windows(frames, target_indices, offsets, boundaries):
    """从原始帧号构造指定目标的输入；返回合法窗口和被排除的目标帧索引。"""
    offsets = tuple(offsets)
    if not offsets or tuple(sorted(set(offsets))) != offsets or 0 not in offsets:
        raise ValueError('时间offset需递增、无重复且含目标0')
    target_indices = np.asarray(target_indices, dtype=np.int64)
    lookup = {(r['match'], r['rally'], int(r['original_frame_id'])): i
              for i, r in enumerate(frames)}
    windows, positions = [], []
    keep = np.zeros(len(target_indices), dtype=bool)
    for position, target in enumerate(target_indices):
        row = frames[target]
        frame = int(row['original_frame_id'])
        window = [lookup.get((row['match'], row['rally'], frame + offset), -1)
                  for offset in offsets]
        if min(window) >= 0:
            windows.append(window)
            positions.append(position)
            keep[position] = True
    windows = np.asarray(windows, dtype=np.int64).reshape(-1, len(offsets))
    windows, crossing = continuous_windows(frames, windows, boundaries)
    if len(crossing):
        keep[np.asarray(positions)[crossing]] = False
    return windows, target_indices[~keep]


def continuous_windows(frames, windows, boundaries):
    """在按rally/原Frame排列的既有缓存上剔除跨已确认内部时间边界的窗口。"""
    windows = np.asarray(windows, dtype=np.int64)
    lookup = {(r['match'], r['rally'], int(r['original_frame_id'])): i
              for i, r in enumerate(frames)}
    keep = np.ones(len(windows), dtype=bool)
    for row in boundaries:
        key = (row['match'], row['rally'], int(row['new_segment_start']))
        if key not in lookup or key[-1] <= 0:
            raise ValueError(f'缓存中不存在合法的内部时间边界: {key}')
        index = lookup[key]
        keep &= ~((windows[:, 0] < index) & (windows[:, -1] >= index))
    return windows[keep], np.flatnonzero(~keep)


def source_coordinates(reference_xy, rows):
    dimensions = np.array([[r['width'], r['height']] for r in rows], dtype=float)
    return (np.asarray(reference_xy) + .5) * dimensions / [1280., 720.] - .5


def position_summary(errors):
    return {'n': len(errors),
            'mean_px': float(np.mean(errors)) if len(errors) else None,
            'median_px': float(np.median(errors)) if len(errors) else None,
            'rmse_px': float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            **{f'pck{r}': float(np.mean(errors < r)) if len(errors) else None
               for r in (4, 8, 16)}}


def summarize(visible, errors, emitted):
    result = {'n_frames': len(visible), 'location': position_summary(errors[visible]),
              'emitted_location': position_summary(errors[visible & emitted]),
              'visibility': _counts(np.sum(visible & emitted), np.sum(~visible & emitted),
                                    np.sum(visible & ~emitted))}
    for radius in (4, 8, 16):
        tp = int(np.sum(visible & emitted & (errors < radius)))
        fp1 = int(np.sum(visible & emitted & (errors >= radius)))
        fp2 = int(np.sum(~visible & emitted))
        fn = int(np.sum(visible & ~emitted))
        tn = int(np.sum(~visible & ~emitted))
        result[f'frame_counts{radius}'] = dict(tp=tp, fp1=fp1, fp2=fp2, fn_visible=fn, tn=tn)
        result[f'detection{radius}'] = _counts(tp, fp1 + fp2, fn + fp1)
        result[f'author_detection{radius}'] = _counts(tp, fp1 + fp2, fn)
    return result


def evaluate_blurball(rows, xy, confidence, grouped=True):
    target = np.array([[r['x_raw'], r['y_raw']] for r in rows], dtype=float).reshape(-1, 2)
    xy, confidence = np.asarray(xy), np.asarray(confidence)
    if xy.shape != target.shape or confidence.shape != (len(rows),):
        raise ValueError('预测必须与BlurBall目标逐帧对齐')
    if not np.isfinite(xy).all() or not np.isfinite(confidence).all():
        raise ValueError('预测位置或可见输出概率非有限')
    visible = np.array([r['visibility_raw'] == 1 for r in rows])
    # V0占位坐标没有几何意义；不参与位置误差。
    errors = np.full(len(rows), np.nan)
    errors[visible] = np.linalg.norm(xy[visible] - target[visible], axis=1)
    emitted = confidence >= .5
    result = summarize(visible, errors, emitted)
    if not grouped:
        return result

    lengths = np.array([r['l_raw'] for r in rows], dtype=float)
    valid_length = visible & np.isfinite(lengths) & (lengths >= 0)
    bins = {'l0': valid_length & (lengths == 0),
            '0_2': valid_length & (lengths > 0) & (lengths <= 2),
            '2_5': valid_length & (lengths > 2) & (lengths <= 5),
            '5_10': valid_length & (lengths > 5) & (lengths <= 10),
            'gt10': valid_length & (lengths > 10)}
    result['visible_without_valid_half_length'] = int(np.sum(visible & ~valid_length))

    def subset(mask):
        return summarize(visible[mask], errors[mask], emitted[mask])

    result['by_half_length'] = {name: subset(mask) for name, mask in bins.items()}
    for name, values in (
            ('match', np.array([r['game'] for r in rows])),
            ('resolution', np.array([f"{r['width']}x{r['height']}" for r in rows]))):
        result[f'by_{name}'] = {value: subset(values == value) for value in sorted(set(values))}
        result[f'by_{name}_half_length'] = {
            value: {label: subset((values == value) & mask) for label, mask in bins.items()}
            for value in sorted(set(values))}
    return result
