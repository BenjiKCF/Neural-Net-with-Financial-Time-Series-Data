"""Mechanism tests for the small independently initialized TimesFM adaptation."""

import torch

from financial_ts.architectures.timesfm import (
    TimesFMMiniEncoder,
    _MixingLayer,
    _patchify,
    _refine_forecast,
    _running_revin,
)


def test_patch_alignment_preserves_the_prediction_cutoff():
    values = torch.arange(1, 6, dtype=torch.float32).reshape(1, 1, -1)
    missing = torch.zeros_like(values, dtype=torch.bool)
    patches, mask = _patchify(values, missing, 4)
    torch.testing.assert_close(
        patches, torch.tensor([[[[0.0, 0.0, 0.0, 1.0], [2.0, 3.0, 4.0, 5.0]]]])
    )
    assert mask.tolist() == [[[[True, True, True, False], [False] * 4]]]


def test_prefix_revin_is_reversible_and_does_not_use_later_patches():
    values = torch.tensor([[[[0.0, 1.0], [3.0, 5.0], [7.0, 9.0]]]])
    missing = torch.tensor([[[[True, False], [False, False], [False, False]]]])
    normalized, count, mean, variance, scale = _running_revin(values, missing)
    reconstructed = normalized * scale.unsqueeze(-1) + mean.unsqueeze(-1)
    torch.testing.assert_close(reconstructed[~missing], values[~missing])
    torch.testing.assert_close(count, torch.tensor([[[1.0, 3.0, 5.0]]]))
    changed = values.clone()
    changed[:, :, -1] = 1000
    changed_normalized, _, changed_mean, changed_variance, _ = _running_revin(changed, missing)
    torch.testing.assert_close(changed_normalized[:, :, :-1], normalized[:, :, :-1])
    torch.testing.assert_close(changed_mean[:, :, :-1], mean[:, :, :-1])
    torch.testing.assert_close(changed_variance[:, :, :-1], variance[:, :, :-1])


def test_mixing_layer_is_causal_even_with_cross_variate_attention():
    torch.manual_seed(2)
    layer = _MixingLayer(d_model=16, nhead=2, dropout=0).eval()
    tokens = torch.randn(2, 3, 5, 16)
    padding = torch.zeros(2, 3, 5, dtype=torch.bool)
    first = layer(tokens, padding)
    changed = tokens.clone()
    changed[:, :, 3:] = torch.randn_like(changed[:, :, 3:]) * 10
    second = layer(changed, padding)
    torch.testing.assert_close(first[:, :, :3], second[:, :, :3])
    assert not torch.allclose(first[:, :, 3:], second[:, :, 3:])


def test_forecast_refinement_uses_preceding_estimates_for_next_patch():
    # Observed history [1,3]: count=2, mean=2, population variance=1.
    # The first prediction [3,3] changes the extended history's mean to 2.5.
    raw = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    count, mean, variance = (torch.tensor([[value]]) for value in (2.0, 2.0, 1.0))
    refined = _refine_forecast(raw, count, mean, variance)
    torch.testing.assert_close(refined, torch.tensor([[[[3.0, 3.0], [2.5, 2.5]]]]))
    changed = raw.clone()
    changed[:, :, 0] = -1
    different = _refine_forecast(changed, count, mean, variance)
    torch.testing.assert_close(different, torch.tensor([[[[1.0, 1.0], [1.5, 1.5]]]]))


def tiny_model():
    return TimesFMMiniEncoder(
        d_feat=5,
        d_model=16,
        nhead=2,
        num_layers=1,
        dropout=0,
        patch_len=4,
        num_variates=3,
        horizon_patches=2,
    )


def test_singleton_batch_trains_temporal_variate_and_forecast_paths():
    torch.manual_seed(4)
    model = tiny_model()
    values = torch.randn(1, 9, 5, requires_grad=True)
    result = model(values, torch.zeros(1, 9, dtype=torch.bool))
    assert result.shape == (1,)
    assert torch.isfinite(result).all()
    (result - 1.0).square().sum().backward()
    for parameter in (
        model.variate_projection.weight,
        model.layers[0].temporal_attention.qkv.weight,
        model.layers[0].variate_attention.qkv.weight,
        model.patch_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_padding_values_cannot_affect_predictions_and_empty_history_is_finite():
    torch.manual_seed(6)
    model = tiny_model().eval()
    values = torch.randn(2, 9, 5)
    padding = torch.zeros(2, 9, dtype=torch.bool)
    padding[0, :5] = True
    padding[1, :] = True
    expected = model(values, padding)
    changed = values.clone()
    changed[padding] = torch.nan
    actual = model(changed, padding)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)


def test_one_context_patch_can_predict_a_contiguous_masked_horizon():
    model = tiny_model()
    values = torch.randn(2, 1, 5)
    output = model(values, torch.zeros(2, 1, dtype=torch.bool))
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
