"""激活顺序比较只允许机制变化，候选覆盖必须逐目标配对。"""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_blurball_temporal_activation import (
    candidate_groups,
    validate_candidate_arrays,
    validate_configs,
)


def configs():
    baseline = {
        'interaction': 'cross_address', 'protocol': 'blurball-centered-hflip-v1',
        'window': 'center5', 'input_slots': ['t-2', 't-1', 't', 't+1', 't+2'],
        'target_slot': 2, 'temporal_input': 'history', 'augmentation': 'hflip',
        'augmentation_probability': .5, 'seed': 0, 'batch_size': 4, 'epochs': 12,
        'train_targets': 37590, 'val_targets': 13912, 'total_parameters': 1279169,
        'head': {'input_channels': 960, 'hidden_channels': 32, 'upscale': 8,
                 'num_frames': 5, 'appearance_channels': 960},
        'optimizer': {'head_lr': .0003}, 'output': 'baseline', 'code_revision': 'before',
    }
    challenger = deepcopy(baseline)
    challenger.update(interaction='target_activation',
                      protocol='blurball-temporal-activation-v1',
                      output='challenger', code_revision='after')
    return baseline, challenger


def test_configs_allow_only_activation_order_change():
    validate_configs(*configs())


@pytest.mark.parametrize('field,value', [
    ('batch_size', 8), ('augmentation_probability', .75), ('target_slot', 4),
    ('input_slots', ['t-4', 't-3', 't-2', 't-1', 't']),
    ('head', {'input_channels': 960, 'hidden_channels': 64, 'upscale': 8,
              'num_frames': 5, 'appearance_channels': 960}),
])
def test_configs_reject_other_training_or_target_changes(field, value):
    baseline, challenger = configs()
    challenger[field] = value
    with pytest.raises(ValueError):
        validate_configs(baseline, challenger)


def test_candidate_identity_mismatch_is_rejected_before_pairing():
    rows = [dict(game='match18', clip='001', original_frame_id=10,
                 visibility_raw=1, x_raw=2., y_raw=3.)]
    frames = [dict(game='match18', clip='001', original_frame_id=11,
                   visibility_raw=1, x_raw=2., y_raw=3.)]
    arrays = {'current_frame_ids': np.array([0]), 'local_xy': np.zeros((1, 16, 2)),
              'q': np.array([.8])}
    with pytest.raises(ValueError, match='身份或顺序'):
        validate_candidate_arrays(rows, frames, arrays, np.zeros((1, 2)), np.array([.8]), 'x')


def test_candidate_coverage_uses_paired_rows_and_strict_radius():
    rows = [dict(game='match18', visibility_raw=1, x_raw=0., y_raw=0.),
            dict(game='match19', visibility_raw=1, x_raw=0., y_raw=0.)]
    baseline = np.full((2, 16, 2), 100.)
    challenger = baseline.copy()
    baseline[0, 0] = [3., 0.]
    challenger[0, 0] = [4., 0.]  # strict <4: breaks
    baseline[1, 0] = [4., 0.]
    challenger[1, 1] = [3., 0.]  # K16 rescues
    masks = {'baseline_error/example': np.array([False, True])}
    groups = candidate_groups(rows, baseline, challenger, masks)
    assert groups['all']['paired_K16']['4']['rescued'] == 1
    assert groups['all']['paired_K16']['4']['broken'] == 1
    assert groups['all']['paired_K16']['4']['net_covered'] == 0
    assert groups['match/match18']['coverage']['baseline']['K1@4']['covered'] == 1
    assert groups['match/match18']['coverage']['challenger']['K1@4']['covered'] == 0
    assert groups['baseline_error/example']['paired_K16']['4']['rescued'] == 1
