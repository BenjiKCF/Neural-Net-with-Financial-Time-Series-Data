"""Behavioral checks for the fixed low-capacity financial forecasting candidates."""

import copy

import pytest
import torch

from financial_ts.architectures.financial import (
    MultiScaleResidualMixerEncoder,
    RegimeLinearMoEEncoder,
    _trailing_moments,
)

MODELS = (MultiScaleResidualMixerEncoder, RegimeLinearMoEEncoder)


def make_model(cls):
    return cls(d_feat=4, d_model=16, nhead=2, num_layers=2, dropout=0)


def test_trailing_statistics_exclude_missing_values_and_preserve_time_gaps():
    values = torch.tensor([[[2.0], [1000.0], [6.0], [10.0], [14.0]]])
    padding = torch.tensor([[False, True, False, False, False]])
    means, variances, counts = _trailing_moments(
        values.masked_fill(padding.unsqueeze(-1), 0), padding
    )
    # Four-session window at the last date contains [missing,6,10,14], not [2,6,10,14].
    torch.testing.assert_close(means[0, -1, 1, 0], torch.tensor(10.0))
    torch.testing.assert_close(counts[0, -1, 1, 0], torch.tensor(3.0))
    torch.testing.assert_close(variances[0, -1, 1, 0], torch.tensor(32.0 / 3))
    torch.testing.assert_close(means[0, 1, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(means[0, -1, -1, 0], torch.tensor(8.0))


@pytest.mark.parametrize("cls", MODELS)
def test_singleton_and_empty_history_start_at_normalized_training_mean(cls):
    torch.manual_seed(3)
    model = make_model(cls).eval()
    for length in (1, 9, 32):
        features = torch.randn(1, length, 4)
        padding = torch.zeros(1, length, dtype=torch.bool)
        output = model(features, padding)
        assert output.shape == (1,)
        torch.testing.assert_close(output, torch.zeros(1), rtol=0, atol=0)
        missing = torch.ones_like(padding)
        result = model(torch.full_like(features, torch.nan), missing)
        assert torch.isfinite(result).all()
        assert torch.isfinite(model.forward_features(features, missing)).all()


@pytest.mark.parametrize("cls", MODELS)
def test_masked_values_and_left_padding_cannot_change_a_forecast(cls):
    torch.manual_seed(5)
    model = make_model(cls).eval()
    # Activate the output so the comparison cannot pass only because of zero heads.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "head" in name or "experts" in name:
                parameter.normal_(0, 0.1)
    values = torch.randn(2, 11, 4)
    padding = torch.zeros(2, 11, dtype=torch.bool)
    padding[0, :4] = True
    expected = model(values, padding)
    changed = values.clone()
    changed[padding] = torch.nan
    torch.testing.assert_close(model(changed, padding), expected)
    padded_values = torch.cat((torch.full((2, 5, 4), torch.nan), values), dim=1)
    padded_mask = torch.cat((torch.ones(2, 5, dtype=torch.bool), padding), dim=1)
    torch.testing.assert_close(model(padded_values, padded_mask), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("cls", MODELS)
def test_hidden_prefix_and_routing_are_causal_after_heads_are_activated(cls):
    torch.manual_seed(8)
    model = make_model(cls).eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "head" in name or "experts" in name:
                parameter.normal_(0, 0.1)
    features = torch.randn(2, 17, 4)
    padding = torch.zeros(2, 17, dtype=torch.bool)
    original = model.forward_features(features, padding)
    perturbed = features.clone()
    perturbed[:, 9:] = torch.randn_like(perturbed[:, 9:]) * 30
    actual = model.forward_features(perturbed, padding)
    torch.testing.assert_close(original[:, :9], actual[:, :9], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(original[:, 9:], actual[:, 9:])
    if isinstance(model, RegimeLinearMoEEncoder):
        torch.testing.assert_close(
            model.routing_weights(features, padding)[:, :9],
            model.routing_weights(perturbed, padding)[:, :9],
        )


def test_regime_gate_is_a_normalized_data_dependent_mixture():
    torch.manual_seed(11)
    model = make_model(RegimeLinearMoEEncoder).eval()
    features = torch.randn(2, 13, 4)
    features[1] = features[0] * 7
    weights = model.routing_weights(features, torch.zeros(2, 13, dtype=torch.bool))
    assert weights.shape == (2, 13, 3)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 13))
    assert not torch.allclose(weights[0, -1], weights[1, -1])


@pytest.mark.parametrize("cls", MODELS)
def test_components_learn_and_state_dict_replays_exactly(cls):
    torch.manual_seed(14)
    model = make_model(cls)
    features = torch.randn(16, 12, 4)
    padding = torch.zeros(16, 12, dtype=torch.bool)
    target = 0.7 * features[:, -1, 0] - 0.3 * features[:, -4:, 1].mean(dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(4):
        optimizer.zero_grad()
        (model(features, padding) - target).square().mean().backward()
        optimizer.step()
    paths = (
        (
            "projection.weight",
            "scale_gate.weight",
            "blocks.0.temporal.weight",
            "residual_head.weight",
        )
        if cls is MultiScaleResidualMixerEncoder
        else ("gate.0.weight", "gate.6.weight", "experts.weight")
    )
    parameters = dict(model.named_parameters())
    for name in paths:
        gradient = parameters[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name
        assert gradient.abs().sum() > 0, name
    model.eval()
    expected = model(features, padding)
    restored = make_model(cls).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(features, padding), expected, rtol=0, atol=0)


@pytest.mark.parametrize("cls", MODELS)
def test_fixed_candidates_learn_a_known_historical_multiscale_signal(cls):
    torch.manual_seed(21)
    torch.set_num_threads(2)
    model = make_model(cls)
    features = torch.randn(96, 12, 4)
    padding = torch.zeros(96, 12, dtype=torch.bool)
    target = (
        0.6 * features[:, -1, 0]
        - 0.3 * features[:, -4:, 1].mean(dim=1)
        + 0.2 * features[:, :, 2].mean(dim=1)
    )
    initial = (model(features, padding) - target).square().mean().item()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(80):
        optimizer.zero_grad()
        loss = (model(features, padding) - target).square().mean()
        loss.backward()
        optimizer.step()
    final = (model(features, padding) - target).square().mean().item()
    assert final < initial * 0.1


@pytest.mark.parametrize("cls", MODELS)
def test_invalid_dimensions_masks_and_observed_nonfinite_values_are_rejected(cls):
    with pytest.raises(ValueError, match="positive integer"):
        cls(d_feat=0)
    with pytest.raises(ValueError, match="dropout"):
        cls(dropout=float("nan"))
    model = make_model(cls)
    features = torch.randn(2, 3, 4)
    with pytest.raises(ValueError, match="padding"):
        model(features, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="Observed"):
        model(features.fill_(float("nan")), torch.zeros(2, 3, dtype=torch.bool))
