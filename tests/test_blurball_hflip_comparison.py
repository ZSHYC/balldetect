"""增强比较不能同时改变模型、目标或训练预算；位移只统计双端V1。"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_blurball_hflip import validate_configs, displacement_groups


def configs(window='center3'):
    baseline = dict(window=window,
                    protocol=('blurball-centered-length-v1' if window == 'center3'
                              else 'blurball-five-frame-context-v1'),
                    augmentation=None, epochs=12, batch_size=4, seed=0,
                    train_targets=37590, val_targets=13912,
                    interaction='cross_address', temporal_input='history',
                    head={'num_frames': 3 if window == 'center3' else 5},
                    optimizer={'head_lr': .0003, 'prefix_lr': .00001},
                    output='baseline', code_revision='before')
    augmented = deepcopy(baseline)
    augmented.update(protocol='blurball-centered-hflip-v1', augmentation='hflip',
                     augmentation_probability=.5, output='augmented', code_revision='after')
    return baseline, augmented


@pytest.mark.parametrize('window', ['center3', 'center5'])
def test_same_window_allows_only_augmentation_change(window):
    validate_configs(*configs(window))


@pytest.mark.parametrize('field,value', [
    ('window', 'center5'), ('batch_size', 8), ('seed', 1),
    ('train_targets', 38854), ('head', {'num_frames': 5}),
    ('optimizer', {'head_lr': .003, 'prefix_lr': .00001}),
    ('augmentation_probability', .75), ('augmentation', None),
])
def test_other_training_changes_are_not_attributed_to_hflip(field, value):
    baseline, augmented = configs()
    augmented[field] = value
    with pytest.raises(ValueError):
        validate_configs(baseline, augmented)


def test_displacement_uses_original_neighbors_and_excludes_unknown_motion():
    frames = [dict(game='match18', clip='001', original_frame_id=i,
                   visibility_raw=v, x_raw=x, y_raw=0.)
              for i, v, x in [(10, 1, 0), (11, 1, 4), (12, 1, 20),
                              (13, 0, 0), (14, 1, 40)]]
    rows = [frames[i] for i in (4, 2, 1, 3)]
    groups = displacement_groups(rows, frames)
    np.testing.assert_array_equal(groups['displacement/d1_lt4'], [False]*4)
    np.testing.assert_array_equal(groups['displacement/d1_4_to16'], [False, False, True, False])
    np.testing.assert_array_equal(groups['displacement/d1_ge16'], [False, True, False, False])
