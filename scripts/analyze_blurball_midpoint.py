"""保存预测的探索性GT拖影轴诊断；不修改正式指标或运行模型。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def components(error_xy, theta_degrees):
    angle = np.deg2rad(theta_degrees)
    dx, dy = np.asarray(error_xy).T
    return np.column_stack((dx * np.cos(angle) + dy * np.sin(angle),
                            -dx * np.sin(angle) + dy * np.cos(angle)))


def describe(values):
    if not len(values):
        return {'n': 0}
    absolute = np.abs(values)
    return {'n': len(values),
            'median_abs_parallel_px': float(np.median(absolute[:, 0])),
            'median_abs_perpendicular_px': float(np.median(absolute[:, 1])),
            'mean_abs_parallel_px': float(np.mean(absolute[:, 0])),
            'mean_abs_perpendicular_px': float(np.mean(absolute[:, 1])),
            'parallel_dominant_fraction': float(np.mean(absolute[:, 0] > absolute[:, 1])),
            'perpendicular_lt4_fraction': float(np.mean(absolute[:, 1] < 4))}


def analyze(run):
    config = json.loads((run / 'config.json').read_text())
    metadata = json.loads((Path(config['rgb_cache']) / 'metadata.json').read_text())
    key = lambda r: (r['game'], r['clip'], int(r['original_frame_id']))
    lookup = {key(r): r for r in metadata['frames']}
    output = {'scope': 'exploratory GT-axis decomposition; original pixels; no model input',
              'groups': {}}
    for split in ('train', 'val'):
        with (run / f'{split}_predictions.csv').open(newline='') as handle:
            predictions = list(csv.DictReader(handle))
        rows = [lookup[key(p)] for p in predictions]
        error = np.array([[float(p['pred_x']) - r['x_raw'],
                           float(p['pred_y']) - r['y_raw']] for p, r in zip(predictions, rows)])
        lengths = np.array([r['l_raw'] for r in rows])
        theta = np.array([r['theta_raw'] for r in rows])
        visible = np.array([r['visibility_raw'] == 1 for r in rows])
        emitted = np.array([float(p['presence_probability']) >= .5 for p in predictions])
        valid = visible & np.isfinite(theta) & np.isfinite(lengths) & (lengths > 0)
        along = components(error, theta)
        distance = np.linalg.norm(error, axis=1)
        matches = np.array([r['game'] for r in rows])
        bands = {'all': np.ones(len(rows), dtype=bool), 'lt4': distance < 4,
                 '4_8': (distance >= 4) & (distance < 8),
                 '8_16': (distance >= 8) & (distance < 16), 'ge16': distance >= 16,
                 '4_16': (distance >= 4) & (distance < 16)}
        bins = {'0_2': lengths <= 2, '2_5': (lengths > 2) & (lengths <= 5),
                '5_10': (lengths > 5) & (lengths <= 10), 'gt10': lengths > 10}
        groups = {}
        for match in ('all', *sorted(set(matches))):
            for label, length_mask in bins.items():
                base = valid & length_mask & ((matches == match) if match != 'all' else True)
                for status, status_mask in [('raw', True), ('emitted', emitted), ('rejected', ~emitted)]:
                    for band, band_mask in bands.items():
                        groups[f'{match}/{label}/{status}/{band}'] = describe(
                            along[base & status_mask & band_mask])
        output['groups'][split] = groups
    (run / 'axis_errors.json').write_text(json.dumps(output, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    result = analyze(args.run)
    for match in ('all', 'match18', 'match19', 'match20', 'match21'):
        label = f'{match}/gt10/raw/4_16'
        print(label, result['groups']['val'][label])
