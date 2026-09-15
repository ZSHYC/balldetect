import numpy as np
import torch

from scripts.train_blurball_layer_points import (
    LayerPointResidual,
    _feature_batch,
    eligible_frame_weights,
    candidate_scores,
    candidate_set_losses,
)


def test_zero_initialization_preserves_original_ranking():
    model = LayerPointResidual(3)
    descriptors = torch.randn(4, 16, 3)
    peaks = torch.randn(4, 16)
    torch.testing.assert_close(candidate_scores(model, descriptors, peaks), peaks)


def test_gt_anchor_cannot_affect_automatic_inference():
    model = LayerPointResidual(2)
    with torch.no_grad():
        model.layers[-1].weight.fill_(1)
    points = np.random.default_rng(0).normal(size=(2, 17, 2)).astype(np.float32)
    changed = points.copy()
    changed[:, 0] = 1e6
    first = _feature_batch(points, np.arange(2), torch.device('cpu'))
    second = _feature_batch(changed, np.arange(2), torch.device('cpu'))
    peaks = torch.randn(2, 16)
    torch.testing.assert_close(candidate_scores(model, first, peaks),
                               candidate_scores(model, second, peaks))


def test_ambiguous_candidate_has_zero_loss_effect_and_gradient():
    logits = torch.tensor([[1., 100., -2.]], requires_grad=True)
    positive = torch.tensor([[True, False, False]])
    allowed = torch.tensor([[True, False, True]])
    loss = candidate_set_losses(logits, positive, allowed).sum()
    loss.backward()
    first = loss.detach().clone()
    assert logits.grad[0, 1].item() == 0

    changed = torch.tensor([[1., -100., -2.]])
    torch.testing.assert_close(candidate_set_losses(changed, positive, allowed).sum(), first)


def test_every_eligible_frame_has_the_same_weight():
    preserve = np.array([True, True, False, False, False, True])
    eligible = np.array([True, True, True, True, True, False])
    weights, counts = eligible_frame_weights(preserve, eligible)
    assert counts == {'preserve': 2, 'recovery': 3}
    np.testing.assert_allclose(weights[eligible], np.full(5, 1 / 5))
    assert np.isclose(weights[eligible & preserve].sum(), 2 / 5)
    assert np.isclose(weights[eligible & ~preserve].sum(), 3 / 5)
    assert weights[~eligible].sum() == 0
